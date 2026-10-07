"""
In N dòng đầu của clean.parquet (mặc định 100) để đọc thử nội dung, dùng pyarrow đọc theo batch nên không nạp cả file.

Cách dùng:
    uv run python scripts/head_clean.py <run_dir> [-n 100] [--max-chars 400] [--status kept]
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

COLS = ["doc_id", "source_key", "status", "quality_band", "language", "word_count", "text"]


def main() -> int:
    """Đọc tham số, in N bản ghi đầu của <run_dir>/clean.parquet; trả về 1 nếu không thấy file."""
    p = argparse.ArgumentParser(description="Xem N bản ghi đầu của clean.parquet.")
    p.add_argument("run_dir", type=Path, help="Thư mục run, vd data/processed/pipeline_runs/run_10000_seed42")
    p.add_argument("-n", type=int, default=100, help="Số bản ghi cần in.")
    p.add_argument("--max-chars", type=int, default=400, help="Cắt văn bản ở ngần này ký tự (0 = in đủ).")
    p.add_argument("--status", default=None, help="Chỉ lấy bản ghi có status này, vd kept.")
    args = p.parse_args()
    path = args.run_dir / "clean.parquet"
    if not path.exists():
        print(f"Không thấy {path}")
        return 1
    pf = pq.ParquetFile(path)
    cols = [c for c in COLS if c in pf.schema_arrow.names]
    shown = 0
    for batch in pf.iter_batches(batch_size=500, columns=cols):
        for r in batch.to_pylist():
            if args.status and r.get("status") != args.status:
                continue
            shown += 1
            text = (r.get("text") or "").replace("\n", " ")
            if args.max_chars and len(text) > args.max_chars:
                text = text[:args.max_chars] + "…"
            meta = " | ".join(f"{k}={r.get(k)}" for k in cols if k != "text")
            print(f"[{shown}] {meta}\n    {text}\n")
            if shown >= args.n:
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
