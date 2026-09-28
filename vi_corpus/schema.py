"""
Schema chung của clean corpus (xem docs/PIPELINE.md, mục 3).

Mọi adapter nguồn đều chuyển bản ghi của mình về schema này để các stage sau
(language, normalize, quality, dedup) không phải biết nguồn gốc.
"""

import hashlib

import pyarrow as pa

# Các trường bắt buộc để tính KPI truy vết nguồn (>= 95% bản ghi phải có đủ 3 trường này).
LINEAGE_FIELDS = ("source_key", "source_path", "source_sha256")

CORPUS_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("source_key", pa.string()),
        ("source_path", pa.string()),  # đường dẫn file gốc, tương đối so với data_root
        ("source_sha256", pa.string()),  # sha256 của nội dung text gốc
        ("text", pa.string()),
        ("language", pa.string()),  # điền ở stage language id
        ("domain", pa.string()),
        ("license", pa.string()),
        ("owner", pa.string()),
        ("rights_status", pa.string()),
        ("quality_band", pa.string()),  # điền ở stage quality
        ("reason_codes", pa.list_(pa.string())),
        ("token_count", pa.int64()),
        ("dedup_family_id", pa.string()),  # điền ở stage dedup
    ]
)


def text_sha256(text: str) -> str:
    """Trả về sha256 (hex) của text, mã hóa UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_doc_id(source_key: str, source_path: str, row_index: int) -> str:
    """Tạo doc_id ổn định từ nguồn, file và số thứ tự dòng (chạy lại cho kết quả giống hệt)."""
    return f"{source_key}:{hashlib.sha1(source_path.encode('utf-8')).hexdigest()[:10]}:{row_index}"


def lineage_rate(rows: list[dict]) -> float:
    """Tỷ lệ bản ghi có đủ các trường trong LINEAGE_FIELDS (KPI >= 95%). Rỗng thì trả 0.0."""
    if not rows:
        return 0.0
    ok = sum(1 for r in rows if all(r.get(f) for f in LINEAGE_FIELDS))
    return ok / len(rows)
