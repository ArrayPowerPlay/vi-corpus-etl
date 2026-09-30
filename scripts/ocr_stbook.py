"""
OCR sách stbook.vn (PDF dạng ảnh do stbook-crawler tải) thành text, mỗi cuốn một file JSON.

Đọc <stbook-root> (mặc định <data-root>/raw/stbook), ghi <data-root>/interim/stbook_ocr/.
Có checkpoint theo cuốn: bị ngắt thì chạy lại đúng lệnh cũ. Cần `uv sync --group ocr`.
Ví dụ:
    uv run python scripts/ocr_stbook.py --data-root /duong/dan/data --limit-books 1 --device cpu
    uv run python scripts/ocr_stbook.py --data-root /duong/dan/data
    uv run python scripts/ocr_stbook.py --data-root /duong/dan/data --status
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được vi_corpus

from vi_corpus.download.downloader import acquire_run_lock, setup_logging  # noqa: E402
from vi_corpus.sources.registry import SOURCES  # noqa: E402
from vi_corpus.sources.stbook import ocr_status, run_ocr  # noqa: E402


def main() -> int:
    """Đọc tham số, rồi in tiến độ (--status) hoặc chạy OCR. Trả về 1 nếu có cuốn lỗi."""
    parser = argparse.ArgumentParser(description="OCR sách stbook.vn thành text.")
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")),
                        help="Thư mục gốc dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).")
    parser.add_argument("--stbook-root", type=Path, default=None,
                        help="Thư mục data/ của stbook-crawler (mặc định <data-root>/raw/stbook).")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Thiết bị chạy VietOCR (mặc định cuda).")
    parser.add_argument("--limit-books", type=int, default=None, help="Chỉ OCR N cuốn (chạy thử).")
    parser.add_argument("--status", action="store_true", help="Chỉ in tiến độ rồi thoát.")
    args = parser.parse_args()
    stbook_root = args.stbook_root or args.data_root / SOURCES["stbook"].path
    if not stbook_root.is_dir():
        parser.error(f"Không thấy thư mục {stbook_root}")

    if args.status:
        done, total = ocr_status(stbook_root, args.data_root)
        print(f"[stbook] OCR xong {done}/{total} cuốn có PDF")
        return 0
    _lock = acquire_run_lock(args.data_root / "state" / "stbook_ocr")  # noqa: F841 — giữ khoá
    setup_logging(args.data_root / "logs", "stbook_ocr")
    return 1 if run_ocr(stbook_root, args.data_root, args.device, args.limit_books) else 0


if __name__ == "__main__":
    sys.exit(main())
