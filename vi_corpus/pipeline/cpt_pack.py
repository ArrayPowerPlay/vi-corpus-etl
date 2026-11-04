"""
Bước chuẩn bị huấn luyện CPT (T-07, T-08, T-09): từ cpt_segments.parquet (stage finalize) cắt + ghép thành chuỗi độ dài cố định.

Vì sao tách khỏi pipeline: cỡ model và độ dài ngữ cảnh chưa chốt (R-21), nên pipeline chỉ ghi các segment liền mạch không
giới hạn độ dài; bước này chạy lại rẻ (CPU, đọc từng phần) mỗi khi đổi độ dài chuỗi hay bật / tắt tiền tố metadata.

Cách làm:
- Cắt (split_segment): segment dài hơn chuỗi được cắt thành các mảnh ĐÚNG ranh giới đoạn (chunk_token_counts /
  chunk_char_offsets), không cắt giữa đoạn. Đoạn đơn lẻ dài hơn chuỗi (hiếm, đoạn tối đa 2.048 token theo D-10) thành một
  mảnh riêng và được đếm là "oversize" (trainer sẽ cắt cụt).
- EOS chỉ ở cuối THẬT của văn bản: mảnh cuối của segment có doc_end = True mới mang eos = True. Mảnh giữa văn bản không có
  EOS (văn bản chưa hết).
- Ghép chuỗi (best_fit_pack): best-fit decreasing (Ding et al. 2024, "Fewer Truncations") trong từng vùng đệm khoảng
  buffer_tokens token: mảnh lớn xếp trước, mỗi mảnh vào chuỗi còn chỗ trống NHỎ NHẤT vừa đủ; không mảnh nào bị cắt ngang
  để lấp chỗ trống. Trainer phải che attention giữa các mảnh trong cùng chuỗi (intra-document masking: reset position_ids /
  cu_seqlens ở đầu mỗi mảnh).
- Tiền tố metadata (T-09): TẮT mặc định. Bật (use_metadata_prefix) thì mỗi mảnh có cột "prefix" (vd "Tiêu đề: ...\\nTạp chí:
  ...\\n\\n") và prefix_tokens; tiền tố nằm ở cột riêng để trainer không tính loss trên nó (MeCo) và bỏ được ở đoạn cuối.

Đầu ra phẳng (mỗi dòng một mảnh, các mảnh cùng seq_id ghép theo seq_pos thành một chuỗi), xem PACKED_SCHEMA.
Số token của mảnh = tổng token các đoạn + 1 token cho mỗi dấu nối "\\n\\n" giữa hai đoạn (ước lượng an toàn theo Qwen3).
"""

from __future__ import annotations

import bisect
import json
import logging
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vi_corpus.pipeline.knowledge import SEGMENT_SEP

logger = logging.getLogger(__name__)

DEFAULT_SEQ_LEN = 8192  # T-08: ngữ cảnh 8K, không train trực tiếp ở 32K
DEFAULT_BUFFER_TOKENS = 20_000_000  # cỡ vùng đệm best-fit (~80 MB chữ), đủ lớn để chuỗi đầy, đủ nhỏ để không tràn RAM
SEGMENT_COLUMNS = ["segment_id", "source_key", "dedup_family_id", "split_bucket", "chunk_token_counts", "chunk_char_offsets",
                   "doc_end", "meta_json", "text"]
# Nhãn tiếng Việt cho các khoá của meta_json khi dựng tiền tố, theo thứ tự in.
PREFIX_LABELS = (("title", "Tiêu đề"), ("journal_title", "Tạp chí"), ("year", "Năm"), ("doi", "DOI"), ("nganh", "Ngành"),
                 ("mon", "Môn"), ("category", "Danh mục"), ("source", "Nguồn"))

PACKED_SCHEMA = pa.schema([
    ("seq_id", pa.int64()),  # số thứ tự chuỗi
    ("seq_pos", pa.int64()),  # vị trí mảnh trong chuỗi (0, 1, ...)
    ("seq_tokens", pa.int64()),  # tổng token cả chuỗi (mảnh + tiền tố + EOS), <= seq_len trừ khi có mảnh oversize
    ("segment_id", pa.string()),
    ("piece_index", pa.int64()),  # mảnh thứ mấy của segment
    ("n_pieces", pa.int64()),  # segment bị cắt thành bao nhiêu mảnh
    ("source_key", pa.string()),
    ("dedup_family_id", pa.string()),
    ("split_bucket", pa.int64()),
    ("prefix", pa.string()),  # tiền tố metadata (null khi tắt); không tính loss
    ("prefix_tokens", pa.int64()),
    ("text", pa.string()),
    ("token_count", pa.int64()),  # token của text (không gồm tiền tố, EOS)
    ("eos", pa.bool_()),  # True = mảnh kết thúc văn bản thật, trainer thêm EOS sau mảnh
])


def render_prefix(meta_json: str | None) -> str:
    """Tiền tố metadata của một segment từ meta_json: mỗi khoá có giá trị một dòng "Nhãn: giá trị", kết thúc bằng một dòng trống."""
    try:
        meta = json.loads(meta_json or "{}")
    except (TypeError, ValueError):
        meta = {}
    lines = [f"{label}: {meta[k]}" for k, label in PREFIX_LABELS if isinstance(meta, dict) and meta.get(k) not in (None, "")]
    return "\n".join(lines) + "\n\n" if lines else ""


def split_segment(seg: dict, seq_len: int, prefix_tokens: int = 0) -> list[dict]:
    """
    Cắt một segment thành các mảnh vừa chuỗi seq_len, đúng ranh giới đoạn (xem docstring module).

    Mỗi mảnh: {"text", "token_count", "eos", "oversize"}; "cost" = token_count + prefix_tokens + (1 nếu eos) là chỗ mảnh
    chiếm trong chuỗi. Đoạn đầu của mảnh luôn được nhận dù vượt seq_len (oversize), để không bao giờ lặp vô hạn.
    """
    counts = list(seg["chunk_token_counts"] or [])
    offsets = list(seg["chunk_char_offsets"] or [])
    text = seg["text"] or ""
    if not counts:  # segment không có thông tin đoạn: cả segment là một đoạn
        counts, offsets = [_approx_tokens(text)], [0]
    pieces: list[dict] = []
    i, n = 0, len(counts)
    while i < n:
        j, tokens = i + 1, counts[i]
        while j < n:
            last = j + 1 == n
            add = counts[j] + 1  # +1 cho dấu nối "\n\n"
            if tokens + add + prefix_tokens + (1 if last and seg["doc_end"] else 0) > seq_len:
                break
            tokens += add
            j += 1
        start = offsets[i]
        end = offsets[j] - len(SEGMENT_SEP) if j < n else len(text)
        eos = j == n and bool(seg["doc_end"])
        cost = tokens + prefix_tokens + (1 if eos else 0)
        pieces.append({"text": text[start:end], "token_count": tokens, "eos": eos, "cost": cost, "oversize": cost > seq_len})
        i = j
    return pieces


def _approx_tokens(text: str) -> int:
    """Ước lượng token khi segment thiếu chunk_token_counts: 1,4 token mỗi từ (M-01: Qwen3 1,366 token / âm tiết)."""
    return max(1, round(1.4 * len(text.split())))


def best_fit_pack(items: list[dict], seq_len: int) -> list[list[dict]]:
    """
    Best-fit decreasing: xếp mảnh theo "cost" giảm dần, mỗi mảnh vào chuỗi còn chỗ trống nhỏ nhất mà vừa; không vừa chuỗi
    nào thì mở chuỗi mới. Mảnh oversize luôn ở chuỗi riêng. Thứ tự ổn định (cùng đầu vào cho cùng kết quả).

    Returns:
        Danh sách chuỗi, mỗi chuỗi là danh sách mảnh theo thứ tự được xếp vào.
    """
    order = sorted(range(len(items)), key=lambda k: (-items[k]["cost"], k))
    bins: list[list[dict]] = []
    free: list[tuple[int, int]] = []  # (chỗ trống còn lại, số thứ tự chuỗi), luôn sắp tăng dần
    for k in order:
        item = items[k]
        pos = bisect.bisect_left(free, (item["cost"], -1))
        if pos < len(free):
            room, b = free.pop(pos)
            bins[b].append(item)
            bisect.insort(free, (room - item["cost"], b))
        else:
            bins.append([item])
            room = seq_len - item["cost"]
            if room > 0:
                bisect.insort(free, (room, len(bins) - 1))
    return bins


def parse_buckets(spec: str | None) -> tuple[int, int] | None:
    """"A-B" -> (A, B) (split_bucket A..B, gồm hai đầu); None / "" = mọi bucket."""
    if not spec:
        return None
    lo, _, hi = spec.partition("-")
    lo_i, hi_i = int(lo), int(hi or lo)
    if not 0 <= lo_i <= hi_i <= 99:
        raise ValueError(f"--buckets phải dạng A-B với 0 <= A <= B <= 99, nhận {spec!r}")
    return lo_i, hi_i


def iter_segments(paths: Iterable[Path], buckets: tuple[int, int] | None, batch_rows: int = 2000) -> Iterator[dict]:
    """Đọc lần lượt các segment từ các file cpt_segments.parquet theo lô (không nạp cả file vào RAM), lọc theo split_bucket."""
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=SEGMENT_COLUMNS):
            for seg in batch.to_pylist():
                if buckets and not buckets[0] <= (seg["split_bucket"] or 0) <= buckets[1]:
                    continue
                yield seg


def pack(paths: list[Path], out: Path, seq_len: int = DEFAULT_SEQ_LEN, use_metadata_prefix: bool = False,
         count_tokens: Callable[[list[str]], list[int]] | None = None, buckets: tuple[int, int] | None = None,
         buffer_tokens: int = DEFAULT_BUFFER_TOKENS) -> dict:
    """
    Cắt + ghép các segment của `paths` thành chuỗi seq_len, ghi Parquet phẳng `out` (PACKED_SCHEMA) nguyên tử.

    Args:
        count_tokens: hàm đếm token theo lô, chỉ cần khi use_metadata_prefix (đếm token của tiền tố).
        buffer_tokens: best-fit trong từng vùng đệm cỡ này rồi ghi ra, nên RAM không phụ thuộc cỡ corpus.

    Returns:
        Thống kê: segments, pieces, sequences, tokens (chữ), prefix_tokens, fill (tỷ lệ lấp chuỗi), split_segments
        (segment bị cắt), oversize_pieces, eos_pieces.
    """
    if use_metadata_prefix and count_tokens is None:
        raise ValueError("use_metadata_prefix cần count_tokens để đếm token của tiền tố")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    stats = {"segments": 0, "pieces": 0, "sequences": 0, "tokens": 0, "prefix_tokens": 0, "used": 0, "split_segments": 0,
             "oversize_pieces": 0, "eos_pieces": 0}
    buf: list[dict] = []
    buf_tokens = 0
    seq_id = 0

    def flush(writer: pq.ParquetWriter) -> None:
        """Ghép chuỗi cho vùng đệm hiện tại rồi ghi ra."""
        nonlocal seq_id, buf, buf_tokens
        rows = []
        for seq in best_fit_pack(buf, seq_len):
            total = sum(p["cost"] for p in seq)
            stats["used"] += total
            for pos, p in enumerate(seq):
                rows.append({"seq_id": seq_id, "seq_pos": pos, "seq_tokens": total, **{k: p[k] for k in (
                    "segment_id", "piece_index", "n_pieces", "source_key", "dedup_family_id", "split_bucket", "prefix",
                    "prefix_tokens", "text", "token_count", "eos")}})
            seq_id += 1
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=PACKED_SCHEMA))
        buf, buf_tokens = [], 0

    with pq.ParquetWriter(tmp, PACKED_SCHEMA) as writer:
        for seg in iter_segments(paths, buckets):
            prefix = render_prefix(seg["meta_json"]) if use_metadata_prefix else ""
            n_prefix = count_tokens([prefix])[0] if prefix else 0
            pieces = split_segment(seg, seq_len, n_prefix)
            stats["segments"] += 1
            stats["split_segments"] += len(pieces) > 1
            for idx, p in enumerate(pieces):
                p.update({"segment_id": seg["segment_id"], "piece_index": idx, "n_pieces": len(pieces),
                          "source_key": seg["source_key"], "dedup_family_id": seg["dedup_family_id"],
                          "split_bucket": seg["split_bucket"], "prefix": prefix or None, "prefix_tokens": n_prefix})
                stats["pieces"] += 1
                stats["tokens"] += p["token_count"]
                stats["prefix_tokens"] += n_prefix
                stats["oversize_pieces"] += p["oversize"]
                stats["eos_pieces"] += p["eos"]
                buf.append(p)
                buf_tokens += p["cost"]
            if buf_tokens >= buffer_tokens:
                flush(writer)
        flush(writer)
    tmp.replace(out)
    stats["sequences"] = seq_id
    stats["fill"] = round(stats.pop("used") / (seq_id * seq_len), 4) if seq_id else 0.0
    logger.info("[cpt_pack] %d segment -> %d mảnh -> %d chuỗi %d token, lấp %.1f%%, %d segment bị cắt, %d mảnh oversize",
                stats["segments"], stats["pieces"], seq_id, seq_len, 100 * stats["fill"], stats["split_segments"],
                stats["oversize_pieces"])
    return stats
