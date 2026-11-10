"""
Đo chữ hỏng do font PDF trên checkpoint VJOL đã có (J-10 (5)); chỉ CPU, logic ở vi_corpus/vjol/garbled_check.py.

Ghi <data-root>/processed/vjol_garbled_check/ (summary.json, docs.csv, pages.csv, examples.txt, new_signals.txt, blind_spots.txt). Chạy trước khi tốn GPU chạy lại mẫu để
chốt ngưỡng: summary.json có số trang / bài bị nghi (tổng và theo bốn tín hiệu tone / case / foreign / unknown), quét ngưỡng từng tín hiệu,
phân bố theo năm, ước số phút OCR lại; new_signals.txt và blind_spots.txt để xem bằng mắt trang bị bắt và trang bị sót (J-13, J-16). Tín hiệu
unknown cần processed/ocr_bakeoff/syllables.json (thiếu thì bỏ). Chỉ cần tải thư mục kết quả về.
Ví dụ:
    uv run python scripts/vjol/check_garbled.py --data-root data
    tar czf vjol_garbled_check.tar.gz -C data/processed vjol_garbled_check
"""

import argparse
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common.state import setup_logging  # noqa: E402
from vi_corpus.vjol.garbled_check import check  # noqa: E402


def main() -> int:
    """Đọc tham số, chạy phần đo và in đường dẫn thư mục kết quả."""
    p = argparse.ArgumentParser(description="Đo chữ hỏng do font PDF trên checkpoint VJOL (CPU).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục dữ liệu gốc.")
    p.add_argument("--seed", type=int, default=0, help="Hạt giống chọn trang ngẫu nhiên cho examples.txt.")
    p.add_argument("--syllables", type=Path, default=None, help="Tập âm tiết (mặc định <data-root>/processed/ocr_bakeoff/syllables.json).")
    args = p.parse_args()
    setup_logging(args.data_root / "logs", "vjol_check_garbled")
    print(f"Xong: {check(args.data_root, args.seed, args.syllables)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
