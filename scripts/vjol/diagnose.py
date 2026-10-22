"""
Chẩn đoán VJOL cho J-22 (1) và (3) A (chỉ CPU, không sửa dữ liệu); logic ở vi_corpus/vjol/diagnose.py.

    # (1) chữ đảo ngược đến từ lớp chữ nhúng hay từ OCR VLM: in từng trang của hai bài mẫu
    uv run python scripts/vjol/diagnose.py --data-root data inspect 8053925644 26f746692e
    # (3) A: phân bố producer của PDF để quyết định OCR lại cả nhóm có lớp chữ OCR cũ lỗi nhẹ; ghi data/processed/vjol_diagnose/
    uv run python scripts/vjol/diagnose.py --data-root data producers
"""

import argparse
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common.state import setup_logging  # noqa: E402
from vi_corpus.vjol.diagnose import inspect_articles, pdf_producers  # noqa: E402


def main() -> int:
    """Đọc tham số rồi chạy lệnh con `inspect` (in ra màn hình) hoặc `producers` (ghi thư mục kết quả)."""
    p = argparse.ArgumentParser(description="Chẩn đoán VJOL (J-22): trang của bài mẫu, phân bố producer PDF.")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục dữ liệu gốc.")
    p.add_argument("--syllables", type=Path, default=None, help="Tập âm tiết (mặc định <data-root>/processed/ocr_bakeoff/syllables.json).")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect", help="In từng trang của các bài chỉ định.")
    s.add_argument("ids", nargs="+", help="article_id hoặc tiền tố sha256 của PDF (vd 8053925644).")
    sub.add_parser("producers", help="Đếm producer / creator của PDF kèm tỷ lệ âm tiết lạ của lớp chữ.")
    args = p.parse_args()
    setup_logging(args.data_root / "logs", "vjol_diagnose")
    if args.cmd == "inspect":
        print(inspect_articles(args.data_root, args.ids, args.syllables))
    else:
        print(f"Xong: {pdf_producers(args.data_root, args.syllables)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
