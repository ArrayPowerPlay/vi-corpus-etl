"""
Bước 1 của đợt so sánh engine OCR (F-11): chọn và chuẩn bị bộ trang.

Bộ A: ~300 trang PDF giáo trình có lớp chữ hợp lệ, render cỡ ảnh stbook rồi làm xấu nhẹ, đáp án = lớp chữ (phân tầng
hai cột / bảng / ký tự đặc biệt). Bộ B: ~40 trang ảnh quét thật từ >= 10 sách stbook (ép trang 1248:150). Chi tiết:
vi_corpus/ocr_bakeoff/pages.py. Chỉ dùng CPU.

Đọc <data-root>/raw/giao_trinh, <data-root>/raw/stbook, interim/giao_trinh_text (đếm trang cần OCR của kho). Ghi vào
--bakeoff-dir (mặc định <data-root>/processed/ocr_bakeoff): pages/, gt/, pages.jsonl, selection.json. Không ghi đè bộ
trang đã có (kết quả các engine gắn với page_id) trừ khi có --overwrite.
Ví dụ:
    uv run python scripts/ocr_bakeoff/select_pages.py --data-root /duong/dan/data
    uv run python scripts/ocr_bakeoff/select_pages.py --data-root /duong/dan/data --force-b 1248:150 --force-b 1271:30,31
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.common.state import setup_logging  # noqa: E402
from vi_corpus.ocr_bakeoff.pages import DEFAULT_FORCED_B, prepare_pages  # noqa: E402


def main() -> int:
    """Đọc tham số, chọn trang, in tóm tắt selection.json."""
    p = argparse.ArgumentParser(description="Chọn bộ trang A (có đáp án) và B (ảnh quét thật) cho so sánh OCR.")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")),
                   help="Thư mục gốc dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).")
    p.add_argument("--bakeoff-dir", type=Path, default=None,
                   help="Thư mục đợt so sánh (mặc định <data-root>/processed/ocr_bakeoff).")
    p.add_argument("--n-a", type=int, default=300, help="Số trang bộ A (mặc định 300).")
    p.add_argument("--n-b", type=int, default=40, help="Số trang bộ B (mặc định 40).")
    p.add_argument("--min-books", type=int, default=10, help="Số sách stbook tối thiểu của bộ B (mặc định 10).")
    p.add_argument("--force-b", action="append", default=None,
                   help="Trang ép vào bộ B, dạng product_id:trang[,trang] (số trang từ 1); lặp được. "
                        f"Mặc định {', '.join(DEFAULT_FORCED_B)}.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-height", type=int, default=None,
                   help="Chiều cao ảnh bộ A (px); mặc định trung vị chiều cao ảnh bộ B.")
    p.add_argument("--max-files", type=int, default=2000, help="Số PDF giáo trình tối đa được xét (mặc định 2000).")
    p.add_argument("--scan-per-doc", type=int, default=8, help="Số trang thử mỗi PDF (mặc định 8).")
    p.add_argument("--max-per-doc", type=int, default=3, help="Số trang nhận tối đa mỗi PDF (mặc định 3).")
    p.add_argument("--overwrite", action="store_true", help="Xoá bộ trang cũ (và runs/, report/) rồi chọn lại.")
    args = p.parse_args()
    out = args.bakeoff_dir or args.data_root / "processed/ocr_bakeoff"
    if (out / "pages.jsonl").exists():
        if not args.overwrite:
            p.error(f"{out}/pages.jsonl đã có; thêm --overwrite để chọn lại (xoá cả kết quả engine trong runs/)")
        for sub in ("pages", "gt", "runs", "report"):
            shutil.rmtree(out / sub, ignore_errors=True)
    setup_logging(args.data_root / "logs", "ocr_bakeoff_select")
    sel = prepare_pages(args.data_root, out, n_a=args.n_a, n_b=args.n_b, min_books=args.min_books,
                        forced_b=tuple(args.force_b or DEFAULT_FORCED_B), seed=args.seed,
                        target_height=args.target_height, max_files=args.max_files,
                        scan_per_doc=args.scan_per_doc, max_per_doc=args.max_per_doc)
    print(json.dumps({k: sel[k] for k in ("set_a", "set_b", "target_height", "strata", "shortfall", "books_b",
                                          "corpus_pages")}, ensure_ascii=False, indent=2))
    if sel["shortfall"]:
        print(f"Thiếu trang ở tầng {sel['shortfall']}: tăng --max-files / --scan-per-doc rồi chạy lại với --overwrite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
