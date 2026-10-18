"""
Kiểm kết quả OCR stbook bằng chuỗi VLM sau khi chạy xong (CPU, không cần GPU / vLLM); logic ở vi_corpus/stbook/ocr_check.py.

Ghi <data-root>/processed/ocr_vlm_check/ (summary.json, books.csv, suspect_pages.csv, foreign_books.csv, odd_pages_in_vi_books.csv,
baseline_pages.csv, samples/, logs/). Chỉ cần tải thư mục này về. Nhận diện ngôn ngữ theo trang bằng fastText lid.176 (tải một lần), tỷ lệ
âm tiết thật dùng <data-root>/processed/ocr_bakeoff/syllables.json nếu có.
Ví dụ:
    uv run --no-sync python scripts/stbook/check_ocr_vlm.py --data-root data
    tar czf ocr_vlm_check.tar.gz -C data/processed ocr_vlm_check
"""

import argparse
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # để import được vi_corpus

from vi_corpus.common.state import setup_logging  # noqa: E402
from vi_corpus.stbook.ocr_check import DEFAULT_LANG_MODEL, check  # noqa: E402


def main() -> int:
    """Đọc tham số, chạy phần kiểm và in đường dẫn thư mục kết quả."""
    p = argparse.ArgumentParser(description="Kiểm kết quả OCR stbook bằng chuỗi VLM (CPU).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục dữ liệu gốc.")
    p.add_argument("--per-tier", type=int, default=20, help="Số trang mẫu ngẫu nhiên cho mỗi engine (có ảnh).")
    p.add_argument("--suspect-samples", type=int, default=20, help="Số trang đáng nghi lấy mẫu thêm.")
    p.add_argument("--seed", type=int, default=0, help="Hạt giống chọn mẫu.")
    p.add_argument("--lang-model", default=DEFAULT_LANG_MODEL, help="Nhận diện ngôn ngữ theo trang: fasttext:lid.176 | fasttext:<file.bin> | heuristic.")
    p.add_argument("--syllables", type=Path, default=None, help="Tập âm tiết (mặc định <data-root>/processed/ocr_bakeoff/syllables.json).")
    args = p.parse_args()
    setup_logging(args.data_root / "logs", "check_ocr_vlm")
    out = check(args.data_root, args.per_tier, args.suspect_samples, args.seed, args.lang_model, args.syllables)
    print(f"Xong: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
