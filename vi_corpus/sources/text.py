"""
Adapter đọc nguồn đã là text (parquet / jsonl.gz) về schema chung.

Chỉ đọc, không sửa file gốc trong data/raw/. Tên cột text lấy từ SourceSpec.text_col;
với SEA-Instruct-2602, cột `conversations` là chuỗi Python-repr nên phải dùng
ast.literal_eval (không dùng json.loads).
"""

import ast
import gzip
import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from vi_corpus.schema import make_doc_id, text_sha256
from vi_corpus.sources.registry import SourceSpec


def conversations_to_text(raw: str) -> str:
    """
    Ghép chuỗi `conversations` (Python-repr của list các lượt thoại) thành một đoạn text.

    Mỗi lượt có dạng {'from': ..., 'value': ...} hoặc {'role': ..., 'content': ...};
    kết quả là các dòng "vai trò: nội dung". Trả về "" nếu không parse được.
    """
    try:
        turns = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return ""
    lines = []
    for turn in turns if isinstance(turns, list) else []:
        role = turn.get("from") or turn.get("role") or ""
        content = turn.get("value") or turn.get("content") or ""
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _iter_rows(path: Path, fmt: str) -> Iterator[dict]:
    """Đọc từng dòng của một file parquet hoặc jsonl.gz dưới dạng dict."""
    if fmt == "parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
            yield from batch.to_pylist()
    else:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def iter_records(spec: SourceSpec, data_root: Path, limit_files: int | None = None) -> Iterator[dict]:
    """
    Sinh các bản ghi theo schema chung (vi_corpus.schema.CORPUS_SCHEMA) cho một nguồn text.

    Các trường language, quality_band, dedup_family_id để None: các stage sau sẽ điền.
    Dòng không có text (rỗng) bị bỏ qua ở stage quality, không bị bỏ ở đây để giữ số liệu funnel.

    Args:
        spec:        Nguồn cần đọc.
        data_root:   Thư mục gốc dữ liệu.
        limit_files: Chỉ đọc N file đầu (chạy thử).
    """
    base = Path(spec.path) if Path(spec.path).is_absolute() else data_root / spec.path
    suffix = ".parquet" if spec.fmt == "parquet" else ".jsonl.gz"
    files = sorted(p for p in base.rglob(f"*{suffix}") if ".cache" not in p.parts)
    for path in files[:limit_files]:
        rel = str(path.relative_to(data_root)) if path.is_relative_to(data_root) else str(path)
        for i, row in enumerate(_iter_rows(path, spec.fmt)):
            raw = row.get(spec.text_col) or ""
            text = conversations_to_text(raw) if spec.text_col == "conversations" else raw
            yield {
                "doc_id": make_doc_id(spec.key, rel, i),
                "source_key": spec.key,
                "source_path": rel,
                "source_sha256": text_sha256(text),
                "text": text,
                "language": None,
                "domain": spec.domain,
                "license": spec.license,
                "owner": spec.owner,
                "rights_status": spec.rights_status,
                "quality_band": None,
                "reason_codes": [],
                "token_count": None,
                "dedup_family_id": None,
            }
