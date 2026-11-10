"""
Xuất bản ghi CORPUS_SCHEMA ra Parquet, dùng chung cho các nguồn xử lý theo lô file (giáo trình, VJOL).
"""

from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vi_corpus.common.schema import CORPUS_SCHEMA

BATCH = 64  # số bản ghi ghi mỗi lần, để không giữ cả kho trong RAM


def export_parquet(records: Iterable[dict], out_path: Path) -> int:
    """
    Ghi các bản ghi ra Parquet (CORPUS_SCHEMA), từng lô BATCH bản ghi, nguyên tử (file tạm rồi os.replace).

    Returns:
        Số bản ghi đã ghi.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    total, batch = 0, []
    with pq.ParquetWriter(tmp, CORPUS_SCHEMA) as writer:
        for rec in records:
            batch.append(rec)
            if len(batch) == BATCH:
                writer.write_table(pa.Table.from_pylist(batch, schema=CORPUS_SCHEMA))
                total, batch = total + len(batch), []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=CORPUS_SCHEMA))
            total += len(batch)
    tmp.replace(out_path)
    return total
