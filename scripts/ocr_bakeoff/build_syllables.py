"""
Lập tập âm tiết tham chiếu từ SEA-PILE v2 cho số đo "âm tiết lạ" của bộ B (F-11). Chỉ dùng CPU.

Đọc tối đa --max-files file parquet trong <data-root>/raw/sea_vi/sea_pile_v2 (chọn rải đều theo thứ tự tên), dừng sau
--max-docs văn bản, giữ từ chữ cái xuất hiện >= --min-freq lần. Ghi <bakeoff-dir>/syllables.json.
Ví dụ:
    uv run python scripts/ocr_bakeoff/build_syllables.py --data-root /duong/dan/data
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])

import pyarrow.parquet as pq  # noqa: E402

from vi_corpus.ocr_bakeoff.syllables import build_syllables  # noqa: E402


def main() -> int:
    """Đọc tham số, đếm âm tiết, ghi file."""
    p = argparse.ArgumentParser(description="Lập tập âm tiết từ SEA-PILE v2 (số đo bộ B của so sánh OCR).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")),
                   help="Thư mục gốc dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).")
    p.add_argument("--bakeoff-dir", type=Path, default=None,
                   help="Thư mục đợt so sánh (mặc định <data-root>/processed/ocr_bakeoff).")
    p.add_argument("--source-dir", type=Path, default=None,
                   help="Thư mục parquet (mặc định <data-root>/raw/sea_vi/sea_pile_v2).")
    p.add_argument("--max-files", type=int, default=10)
    p.add_argument("--max-docs", type=int, default=300_000)
    p.add_argument("--min-freq", type=int, default=20)
    args = p.parse_args()
    src = args.source_dir or args.data_root / "raw/sea_vi/sea_pile_v2"
    files = sorted(src.rglob("*.parquet"))
    if not files:
        p.error(f"Không thấy file parquet trong {src}")
    step = max(1, len(files) // args.max_files)
    files = files[::step][: args.max_files]
    names = pq.ParquetFile(files[0]).schema_arrow.names
    text_col = "text" if "text" in names else next((n for n in names if "text" in n.lower()), None)
    if text_col is None:
        p.error(f"Không thấy cột văn bản trong {files[0]} (các cột: {names})")
    res = build_syllables(files, text_col, max_docs=args.max_docs, min_freq=args.min_freq)
    out = (args.bakeoff_dir or args.data_root / "processed/ocr_bakeoff") / "syllables.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"{len(res['syllables'])} âm tiết (tần suất >= {args.min_freq}) từ {res['docs']} văn bản, "
          f"{res['files']} file -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
