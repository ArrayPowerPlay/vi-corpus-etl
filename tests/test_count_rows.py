"""Kiểm tra scripts/count_rows.py: đếm đúng số dòng parquet/jsonl.gz và bỏ qua .cache/."""

import gzip

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.count_rows import count_rows, find_files


def test_count_rows_and_skip_cache(tmp_path):
    """Đếm đúng cả khi dòng cuối jsonl.gz thiếu '\\n'; file trong .cache/ không được liệt kê."""
    ds = tmp_path / "ds" / "vi"
    ds.mkdir(parents=True)
    pq.write_table(pa.table({"text": ["a", "b", "c"]}), ds / "x.parquet")
    with gzip.open(ds / "y.jsonl.gz", "wb") as f:
        f.write(b'{"t":1}\n{"t":2}')  # dòng cuối không có "\n"
    cache = tmp_path / "ds" / ".cache"
    cache.mkdir()
    (cache / "z.parquet").write_bytes(b"")

    files = find_files(tmp_path)
    assert [p.name for p in files] == ["x.parquet", "y.jsonl.gz"]
    assert [count_rows(p) for p in files] == [3, 2]
