"""
Stage finalize (knowledge unit): từ clean corpus sinh các đơn vị tri thức cho CPT / SFT.

Thuật ngữ (R-09): KU loại "cpt" = một candidate (một đoạn CPT còn sống sau mọi bước lọc và dedup, có đủ provenance; các
chỉ tiêu "≥100k / ≥500k / ≥1,2M candidates" đếm đơn vị này); KU loại "sft" = một instruction (cặp hỏi - đáp).

- CPT: mỗi bản ghi giữ lại (status = kept) là một đơn vị văn bản thuần (SEA-Pile, đoạn sách).
- SFT: mỗi cuộc hội thoại SEA-Instruct có >= 1 cặp (human, assistant) cho các đơn vị (instruction, response), mỗi cặp liền
  kề là một đơn vị; vai trò "gpt" / "assistant" / "model" là phía trả lời.
- Hybrid: ghép từ hai loại trên ở bước huấn luyện (theo tỷ lệ), nên không sinh riêng ở đây.
Mọi đơn vị mang dedup_family_id của bản ghi gốc; split_bucket (0..99) là băm của dedup_family_id: cắt train / val / test
sau này chỉ cần so split_bucket với ngưỡng, nên các bản gần trùng luôn rơi cùng một phía (không chia ở tầng corpus).
Theo D-09, knowledge unit chính thức sinh SAU dedup vòng 2 (RunConfig.global_verdict); không có verdict thì họ trùng
chỉ là của lần chạy này.

Đoạn liền mạch CPT (T-07, build_cpt_segments → cpt_segments.parquet): các đoạn của cùng một văn bản được nhóm theo
parent_doc_id, sắp theo chunk_index, rồi nối MỌI đoạn LIỀN NHAU còn giữ thành một "segment", KHÔNG giới hạn độ dài (bản
cũ cpt_blocks cắt ở 4.096 token, khoá pipeline vào một độ dài ngữ cảnh). Gặp đoạn bị loại hoặc đoạn không có trong lần chạy
(lấy mẫu) thì ngắt segment. Cắt segment theo độ dài ngữ cảnh và ghép chuỗi (best-fit packing) là việc của bước chuẩn bị
huấn luyện vi_corpus.pipeline.cpt_pack (scripts/cpt_pack.py), nên đổi cỡ model / độ dài ngữ cảnh không phải chạy lại
pipeline. Metadata (tiêu đề, tạp chí, năm, ngành, môn...) được thu sẵn vào cột riêng (T-09), không trộn vào text.
"""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vi_corpus.pipeline.tokens import TokenCounter

KU_SCHEMA = pa.schema([
    ("unit_id", pa.string()),
    ("doc_id", pa.string()),
    ("source_key", pa.string()),
    ("dedup_family_id", pa.string()),
    ("split_bucket", pa.int64()),
    ("unit_type", pa.string()),  # "cpt" | "sft"
    ("text", pa.string()),  # cpt: văn bản; sft: instruction
    ("response", pa.string()),  # sft: câu trả lời; cpt: null
    ("token_count", pa.int64()),
])
SEGMENT_SCHEMA = pa.schema([
    ("segment_id", pa.string()),  # "<parent_doc_id>:s<chunk_start>", ổn định (không phụ thuộc thứ tự đầu vào)
    ("parent_doc_id", pa.string()),
    ("source_key", pa.string()),
    ("source_path", pa.string()),  # lineage của văn bản gốc
    ("source_sha256", pa.string()),
    ("language", pa.string()),
    ("domain", pa.string()),
    ("license", pa.string()),
    ("rights_status", pa.string()),
    ("dedup_family_id", pa.string()),  # của đoạn đầu
    ("split_bucket", pa.int64()),
    ("doc_ids", pa.list_(pa.string())),  # các đoạn đã nối, theo thứ tự
    ("quality_bands", pa.list_(pa.string())),  # band chất lượng từng đoạn, cùng thứ tự doc_ids
    ("chunk_start", pa.int64()),  # chunk_index của đoạn đầu / đoạn cuối
    ("chunk_end", pa.int64()),
    ("chunk_token_counts", pa.list_(pa.int64())),  # số token từng đoạn: cpt_pack cắt segment dài đúng ranh giới đoạn
    ("chunk_char_offsets", pa.list_(pa.int64())),  # vị trí ký tự bắt đầu của từng đoạn trong text
    ("doc_start", pa.bool_()),  # segment bắt đầu ở đoạn đầu văn bản (chunk_index 0)
    ("doc_end", pa.bool_()),  # segment kết thúc ở đoạn cuối văn bản có trong lần chạy
    ("doc_complete", pa.bool_()),  # segment phủ cả văn bản (doc_start và doc_end)
    ("break_reason", pa.string()),  # vì sao segment dừng: "doc_end" | "rejected" (đoạn sau bị loại) | "gap" (đoạn sau không có)
    ("title", pa.string()),  # tiêu đề văn bản nếu nguồn có (bài báo, sách, file giáo trình)
    ("meta_json", pa.string()),  # JSON metadata cho tiền tố (title, journal, year, nganh, mon, category...); không nằm trong text
    ("text_format", pa.string()),  # "plain" (chữ thuần); "markdown" khi pipeline đọc được bản markdown của OCR (T-06)
    ("text", pa.string()),  # các đoạn nối bằng SEGMENT_SEP
    ("token_count", pa.int64()),
])
SEGMENT_SEP = "\n\n"
# Khoá metadata lấy từ cột "meta" (JSON) của bản ghi cho meta_json, theo nguồn: vjol (journal_title, year, doi), giáo trình
# (nganh, mon), stbook (category). Khoá không có thì bỏ qua.
SEGMENT_META_KEYS = ("title", "journal_title", "year", "doi", "nganh", "mon", "category")
_ASSISTANT_ROLES = frozenset({"gpt", "assistant", "model", "bot"})


def split_bucket(family: str) -> int:
    """Số 0..99 ổn định từ dedup_family_id, để chia train / val / test theo họ trùng."""
    return int(hashlib.sha1(family.encode("utf-8")).hexdigest()[:8], 16) % 100


def sft_pairs(turns: list[dict]) -> list[tuple[str, str]]:
    """Các cặp (lời hỏi, lời đáp) từ các lượt thoại liền kề: một lượt không phải trợ lý, theo sau là một lượt trợ lý."""
    pairs = []
    for prev, cur in zip(turns, turns[1:]):
        if cur["role"].lower() in _ASSISTANT_ROLES and prev["role"].lower() not in _ASSISTANT_ROLES:
            pairs.append((prev["content"], cur["content"]))
    return pairs


def build_units(rows: list[dict], counter: TokenCounter) -> list[dict]:
    """
    Sinh các knowledge unit từ các bản ghi có status = "kept".

    Hội thoại không có cặp hợp lệ nào thì được dùng như văn bản CPT. token_count của SFT đếm bằng cùng bộ đếm token
    với corpus (D-01).
    """
    units = []
    for row in rows:
        if row["status"] != "kept":
            continue
        base = {"doc_id": row["doc_id"], "source_key": row["source_key"], "dedup_family_id": row["dedup_family_id"],
                "split_bucket": split_bucket(row["dedup_family_id"])}
        turns = (json.loads(row["meta"]).get("turns") or []) if row["source_key"] == "sea_instruct_2602" and row["meta"] else []
        pairs = sft_pairs(turns)
        if pairs:
            counts = counter.count_batch([q + "\n" + a for q, a in pairs])
            for i, ((q, a), n) in enumerate(zip(pairs, counts)):
                units.append({**base, "unit_id": f"{row['doc_id']}:sft{i}", "unit_type": "sft", "text": q,
                              "response": a, "token_count": n})
        else:
            units.append({**base, "unit_id": f"{row['doc_id']}:cpt", "unit_type": "cpt", "text": row["text"],
                          "response": None, "token_count": row["token_count"]})
    return units


def _segment_meta(row: dict) -> tuple[str | None, str]:
    """(tiêu đề, meta_json) của văn bản từ cột "meta" của một đoạn; chỉ giữ SEGMENT_META_KEYS có giá trị, cộng nguồn."""
    try:
        meta = json.loads(row.get("meta") or "{}")
    except (TypeError, ValueError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    picked = {k: meta[k] for k in SEGMENT_META_KEYS if meta.get(k) not in (None, "")}
    picked["source"] = row["source_key"]
    title = picked.get("title")
    return (str(title) if title is not None else None), json.dumps(picked, ensure_ascii=False)


def build_cpt_segments(rows: list[dict]) -> list[dict]:
    """
    Nối MỌI đoạn CPT liền nhau còn giữ của cùng văn bản thành một segment, không giới hạn độ dài (xem docstring module).

    Bỏ qua hội thoại SEA-Instruct (là SFT). Văn bản không bị cắt (không có parent_doc_id) là một segment riêng. "doc_end" so
    với chunk_index lớn nhất của văn bản CÓ TRONG lần chạy (chạy mẫu có thể thiếu đoạn cuối thật). Segment mang
    dedup_family_id của đoạn đầu.
    """
    by_parent: dict[str, list[dict]] = {}
    for row in rows:
        if row["source_key"] == "sea_instruct_2602":
            continue
        by_parent.setdefault(row.get("parent_doc_id") or row["doc_id"], []).append(row)
    segments: list[dict] = []

    def flush(parent: str, cur: list[dict], last_idx: int, reason: str) -> None:
        """Ghi segment đang gom (nếu có) vào danh sách; reason = vì sao dừng."""
        if not cur:
            return
        first = cur[0]
        start, end = cur[0].get("chunk_index") or 0, cur[-1].get("chunk_index") or 0
        offsets, pos = [], 0
        for r in cur:
            offsets.append(pos)
            pos += len(r["text"] or "") + len(SEGMENT_SEP)
        title, meta_json = _segment_meta(first)
        fam = first["dedup_family_id"]
        doc_start, doc_end = start == 0, end == last_idx
        segments.append({
            "segment_id": f"{parent}:s{start}", "parent_doc_id": parent, "source_key": first["source_key"],
            "source_path": first.get("source_path"), "source_sha256": first.get("source_sha256"),
            "language": first.get("language"), "domain": first.get("domain"), "license": first.get("license"),
            "rights_status": first.get("rights_status"), "dedup_family_id": fam, "split_bucket": split_bucket(fam),
            "doc_ids": [r["doc_id"] for r in cur], "quality_bands": [r.get("quality_band") for r in cur],
            "chunk_start": start, "chunk_end": end, "chunk_token_counts": [r["token_count"] or 0 for r in cur],
            "chunk_char_offsets": offsets, "doc_start": doc_start, "doc_end": doc_end,
            "doc_complete": doc_start and doc_end, "break_reason": reason, "title": title, "meta_json": meta_json,
            "text_format": "plain", "text": SEGMENT_SEP.join(r["text"] or "" for r in cur),
            "token_count": sum(r["token_count"] or 0 for r in cur)})

    for parent, group in by_parent.items():
        group.sort(key=lambda r: r.get("chunk_index") or 0)
        last_idx = group[-1].get("chunk_index") or 0
        cur: list[dict] = []
        prev_idx = None
        for row in group:
            idx = row.get("chunk_index") or 0
            gap = prev_idx is not None and idx != prev_idx + 1
            prev_idx = idx
            if row["status"] != "kept":
                flush(parent, cur, last_idx, "rejected")
                cur = []
                continue
            if cur and gap:
                flush(parent, cur, last_idx, "gap")
                cur = []
            cur.append(row)
        flush(parent, cur, last_idx, "doc_end")
    return segments


def _write(path: Path, items: list[dict], schema: pa.Schema) -> int:
    """Ghi các dòng ra Parquet theo schema, nguyên tử. Trả về số dòng."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(items, schema=schema), tmp)
    tmp.replace(path)
    return len(items)


def write_units(path: Path, units: list[dict]) -> int:
    """Ghi các knowledge unit ra Parquet (KU_SCHEMA), nguyên tử. Trả về số đơn vị."""
    return _write(path, units, KU_SCHEMA)


def write_segments(path: Path, segments: list[dict]) -> int:
    """Ghi các segment CPT ra Parquet (SEGMENT_SCHEMA), nguyên tử. Trả về số segment."""
    return _write(path, segments, SEGMENT_SCHEMA)
