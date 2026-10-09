"""
Đọc / ghi dữ liệu giữa các stage: schema Parquet chung của pipeline, ghi nguyên tử và manifest của lần chạy.

PIPELINE_SCHEMA = CORPUS_SCHEMA + các cột do pipeline điền (meta, đoạn, đếm từ, điểm ngôn ngữ / chất lượng,
trạng thái giữ / loại, thông tin trùng lặp). Mọi stage ghi cùng một schema; cột chưa tới stage điền thì để null.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vi_corpus.common.schema import CORPUS_SCHEMA
from vi_corpus.common.state import read_json, write_json_atomic

PIPELINE_SCHEMA = pa.schema(
    list(CORPUS_SCHEMA)
    + [
        ("meta", pa.string()),  # JSON: lượt thoại (SEA-Instruct), tên sách (stbook)
        ("parent_doc_id", pa.string()),  # doc_id của tài liệu gốc nếu bản ghi là một đoạn
        ("chunk_index", pa.int64()),
        ("char_count", pa.int64()),
        ("word_count", pa.int64()),
        ("lines_removed", pa.int64()),  # số dòng bị xoá ở bước xoá dòng lặp (D-04)
        ("chars_removed", pa.int64()),
        ("doc_sha256", pa.string()),  # sha256 của cả văn bản gốc (sau làm sạch dòng, trước khi cắt đoạn)
        ("doc_minhash", pa.binary()),  # chữ ký MinHash của cả văn bản gốc (dedup theo văn bản, R-19)
        ("lang_score", pa.float64()),
        ("lang_mix", pa.string()),  # JSON {ngôn ngữ: phần độ dài} (D-02)
        ("lang_answer", pa.string()),  # ngôn ngữ phía trả lời của hội thoại SFT (F-07); null với bản ghi khác
        ("quality_score", pa.float64()),
        ("quality_metrics", pa.string()),  # JSON các số đo thô (để vẽ biểu đồ và gỡ lỗi ngưỡng)
        ("rights_gate", pa.string()),  # "pass" | "quarantine" (trục quyền, tách biệt với chất lượng)
        ("status", pa.string()),  # "kept" | "rejected:quality" | "rejected:rights" | "rejected:duplicate"
        ("dup_kind", pa.string()),  # "exact" | "fuzzy" | "contained" | "global_exact" | "global_fuzzy" (vòng 2, D-09) | null
        ("dup_of", pa.string()),  # doc_id bản được giữ lại trong họ trùng
    ]
)
BATCH = 500  # số bản ghi ghi mỗi lần, để không nhân đôi bộ nhớ khi ghi


def fill(row: dict) -> dict:
    """Trả về bản ghi đủ mọi cột của PIPELINE_SCHEMA (cột thiếu = None, reason_codes = [])."""
    out = {name: row.get(name) for name in PIPELINE_SCHEMA.names}
    out["reason_codes"] = list(row.get("reason_codes") or [])
    return out


def write_rows(path: Path, rows: Iterable[dict]) -> int:
    """
    Ghi các bản ghi ra Parquet (PIPELINE_SCHEMA), nguyên tử (file tạm rồi os.replace).

    Returns:
        Số bản ghi đã ghi.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    total, batch = 0, []
    with pq.ParquetWriter(tmp, PIPELINE_SCHEMA) as writer:
        for row in rows:
            batch.append(fill(row))
            if len(batch) == BATCH:
                writer.write_table(pa.Table.from_pylist(batch, schema=PIPELINE_SCHEMA))
                total, batch = total + len(batch), []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=PIPELINE_SCHEMA))
            total += len(batch)
        elif total == 0:  # file rỗng vẫn phải có schema hợp lệ
            writer.write_table(pa.Table.from_pylist([], schema=PIPELINE_SCHEMA))
    tmp.replace(path)
    return total


def read_rows(path: Path) -> list[dict]:
    """Đọc toàn bộ file Parquet của một stage thành list[dict] (stage dùng cho vài chục nghìn bản ghi)."""
    return pq.read_table(path).to_pylist()


def read_manifest(run_dir: Path) -> dict:
    """Đọc manifest.json của lần chạy ({} nếu chưa có)."""
    return read_json(run_dir / "manifest.json") or {}


def update_manifest(run_dir: Path, key: str, value: dict) -> None:
    """Ghi (nguyên tử) mục `key` của manifest.json, giữ nguyên các mục khác."""
    manifest = read_manifest(run_dir)
    manifest[key] = value
    write_json_atomic(run_dir / "manifest.json", manifest)


def loads(value: str | None) -> dict:
    """Parse cột JSON dạng chuỗi (meta, quality_metrics); rỗng hoặc hỏng thì trả {}."""
    try:
        return json.loads(value) if value else {}
    except json.JSONDecodeError:
        return {}
