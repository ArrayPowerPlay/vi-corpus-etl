"""
Stage ingest: lấy mẫu từ data/raw và đưa về schema chung, không sửa file gốc.

- Nguồn text (SEA, parquet / jsonl.gz): chọn ngẫu nhiên (theo seed) một số file, mỗi file lấy `rows_per_file` dòng
  ngẫu nhiên, cho đến khi đủ số mẫu. Parquet chỉ đọc cột text và chỉ chuyển thành dict các dòng được chọn; jsonl.gz
  dùng reservoir sampling một lượt (gzip không truy cập ngẫu nhiên được). Bộ nhớ không phụ thuộc cỡ file.
- Nguồn sách (stbook, giao_trinh): đọc cả sách (đã OCR / trích text ở interim/); số mẫu được cắt xuống ở stage chunk,
  vì "mẫu" của nguồn này là một đoạn chứ không phải cả cuốn.
"""

import gzip
import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vi_corpus.common.registry import SourceSpec
from vi_corpus.pipeline.config import RunConfig
from vi_corpus.sea.reader import row_to_record, source_files

logger = logging.getLogger("vi_corpus")

# Phiên bản code của stage ingest, nằm trong vân tay stage (F-04): nâng khi đổi iter_records / row_to_record / cột meta
# để 01_ingest.parquet cũ tự chạy lại. 2 = lần đầu đưa vào vân tay (meta stbook có page_lines_removed, commit 24bc92a).
INGEST_VERSION = "2"


def _sample_parquet(path: Path, col: str, k: int, rng: random.Random) -> list[tuple[int, dict]]:
    """
    Lấy k dòng ngẫu nhiên của một file parquet: chọn chỉ số từ metadata, rồi đọc từng lô và chỉ chuyển các dòng đã chọn.

    Returns:
        Danh sách (số thứ tự dòng trong file, {col: giá trị}), tăng dần theo số thứ tự.
    """
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    wanted = sorted(rng.sample(range(total), min(k, total)))
    out, offset, w = [], 0, 0
    for batch in pf.iter_batches(batch_size=1000, columns=[col]):
        local = []
        while w < len(wanted) and wanted[w] < offset + batch.num_rows:
            local.append(wanted[w] - offset)
            w += 1
        if local:
            rows = batch.take(pa.array(local)).to_pylist()
            out.extend((offset + i, r) for i, r in zip(local, rows))
        offset += batch.num_rows
        if w == len(wanted):
            break
    return out


def _sample_jsonl_gz(path: Path, k: int, rng: random.Random) -> list[tuple[int, dict]]:
    """Lấy k dòng ngẫu nhiên của file jsonl.gz bằng reservoir sampling; chỉ parse JSON các dòng được chọn."""
    reservoir: list[tuple[int, str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        i = -1
        for line in f:
            if not line.strip():
                continue
            i += 1
            if len(reservoir) < k:
                reservoir.append((i, line))
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = (i, line)
    return sorted((i, json.loads(line)) for i, line in reservoir)


def sample_text_source(spec: SourceSpec, data_root: Path, n: int, cfg: RunConfig) -> list[dict]:
    """
    Lấy n bản ghi ngẫu nhiên (theo cfg.seed) từ một nguồn text, trải đều trên nhiều file.

    Nếu nguồn có ít hơn n dòng thì trả về tất cả những gì có (log cảnh báo).
    """
    files = source_files(spec, data_root)
    if not files:
        logger.warning("Nguồn %s: không thấy file dữ liệu ở %s", spec.key, spec.path)
        return []
    rng = random.Random(f"{cfg.seed}:{spec.key}")
    rng.shuffle(files)
    records: list[dict] = []
    for path in files:
        if len(records) >= n:
            break
        k = min(cfg.rows_per_file, n - len(records))
        picked = (_sample_parquet(path, spec.text_col, k, rng) if spec.fmt == "parquet"
                  else _sample_jsonl_gz(path, k, rng))
        rel = str(path.relative_to(data_root)) if path.is_relative_to(data_root) else str(path)
        records.extend(row_to_record(spec, rel, i, row, with_meta=True) for i, row in picked)
    if len(records) < n:
        logger.warning("Nguồn %s: chỉ lấy được %d/%d mẫu", spec.key, len(records), n)
    return records


def iter_book_source(spec: SourceSpec, data_root: Path) -> Iterator[dict]:
    """Sinh bản ghi từng cuốn sách / tài liệu của nguồn dài (stbook đã làm sạch theo trang; giao_trinh đã làm sạch sẵn)."""
    if spec.key == "stbook":
        from vi_corpus.stbook.ocr_books import iter_records
        yield from iter_records(spec, data_root, clean=True, with_meta=True)
    elif spec.key == "giao_trinh":
        from vi_corpus.giao_trinh.records import iter_records
        yield from iter_records(spec, data_root, with_meta=True)
    else:
        raise ValueError(f"Chưa có adapter nguồn sách cho {spec.key}")


def input_files(spec: SourceSpec, data_root: Path, chunked: bool) -> list[Path]:
    """
    Các file đầu vào stage ingest đọc cho một nguồn (dùng tính vân tay đầu vào, G-02): file parquet / jsonl.gz của nguồn
    text; file kết quả OCR (stbook) hoặc checkpoint trích text (giao_trinh) của nguồn sách.
    """
    if not chunked:
        return source_files(spec, data_root)
    if spec.key == "stbook":
        from vi_corpus.stbook.ocr_books import OCR_DIR
        return sorted((data_root / OCR_DIR).glob("*/*.json"))
    if spec.key == "giao_trinh":
        from vi_corpus.giao_trinh.extract import TEXT_DIR
        return sorted((data_root / TEXT_DIR).glob("*.json"))
    return []


def ingest(specs: dict[str, SourceSpec], data_root: Path, cfg: RunConfig) -> list[dict]:
    """
    Chạy stage ingest cho mọi nguồn trong cfg.mix, trả về bản ghi (chưa chuẩn hóa).

    Nguồn sách trả về nguyên cuốn (xem chunk). Nguồn không có trong registry thì báo lỗi ngay.

    Raises:
        KeyError: nếu cfg.mix có nguồn không có trong registry.
    """
    records: list[dict] = []
    for key, n in cfg.mix.items():
        spec = specs[key]
        if cfg.profile(key).chunked:
            rows = list(iter_book_source(spec, data_root))
            if not rows:
                logger.warning("Nguồn %s: chưa có sách nào (đã OCR / trích text chưa?)", key)
        else:
            rows = sample_text_source(spec, data_root, n, cfg)
        logger.info("ingest %-18s %6d bản ghi", key, len(rows))
        records.extend(rows)
    return records
