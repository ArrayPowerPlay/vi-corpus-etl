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

from vi_corpus.common.schema import make_doc_id, text_sha256
from vi_corpus.common.registry import SourceSpec


def conversations_to_turns(raw: str) -> list[dict]:
    """
    Parse chuỗi `conversations` (Python-repr của list các lượt thoại) thành list {"role", "content"}.

    Mỗi lượt có dạng {'from': ..., 'value': ...} hoặc {'role': ..., 'content': ...}; lượt rỗng bị bỏ.
    Trả về [] nếu không parse được.
    """
    try:
        turns = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    out = []
    for turn in turns if isinstance(turns, list) else []:
        if not isinstance(turn, dict):
            continue
        content = turn.get("value") or turn.get("content") or ""
        if content:
            out.append({"role": turn.get("from") or turn.get("role") or "", "content": content})
    return out


def conversations_to_text(raw: str) -> str:
    """
    Ghép chuỗi `conversations` thành một đoạn text: mỗi lượt một dòng "vai trò: nội dung".

    Trả về "" nếu không parse được (xem conversations_to_turns).
    """
    return "\n".join(f"{t['role']}: {t['content']}" for t in conversations_to_turns(raw))


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


def row_to_record(spec: SourceSpec, rel: str, row_index: int, row: dict, with_meta: bool = False) -> dict:
    """
    Chuyển một dòng gốc (dict) của nguồn text về bản ghi schema chung.

    Các trường language, quality_band, dedup_family_id để None: các stage sau sẽ điền.

    Args:
        spec:      Nguồn của dòng.
        rel:       Đường dẫn file chứa dòng, tương đối so với data_root.
        row_index: Số thứ tự dòng trong file (dùng cho doc_id ổn định).
        row:       Dòng gốc đã đọc.
        with_meta: True thì thêm khoá "meta" (JSON); với SEA-Instruct là {"turns": [{"role", "content"}]}.
    """
    raw = row.get(spec.text_col) or ""
    text = conversations_to_text(raw) if spec.text_col == "conversations" else raw
    rec = {
        "doc_id": make_doc_id(spec.key, rel, row_index),
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
    if with_meta:
        turns = conversations_to_turns(raw) if spec.text_col == "conversations" else []
        rec["meta"] = json.dumps({"turns": turns}, ensure_ascii=False) if turns else None
    return rec


def source_files(spec: SourceSpec, data_root: Path) -> list[Path]:
    """Danh sách file dữ liệu (parquet / jsonl.gz) của một nguồn text, đã sắp xếp, bỏ thư mục .cache của HF."""
    base = Path(spec.path) if Path(spec.path).is_absolute() else data_root / spec.path
    suffix = ".parquet" if spec.fmt == "parquet" else ".jsonl.gz"
    return sorted(p for p in base.rglob(f"*{suffix}") if ".cache" not in p.parts)


def iter_records(spec: SourceSpec, data_root: Path, limit_files: int | None = None,
                 with_meta: bool = False) -> Iterator[dict]:
    """
    Sinh các bản ghi theo schema chung (vi_corpus.common.schema.CORPUS_SCHEMA) cho một nguồn text.

    Dòng không có text (rỗng) bị bỏ qua ở stage quality, không bị bỏ ở đây để giữ số liệu funnel.

    Args:
        spec:        Nguồn cần đọc.
        data_root:   Thư mục gốc dữ liệu.
        limit_files: Chỉ đọc N file đầu (chạy thử).
        with_meta:   Thêm khoá "meta" vào bản ghi (xem row_to_record).
    """
    for path in source_files(spec, data_root)[:limit_files]:
        rel = str(path.relative_to(data_root)) if path.is_relative_to(data_root) else str(path)
        for i, row in enumerate(_iter_rows(path, spec.fmt)):
            yield row_to_record(spec, rel, i, row, with_meta)
