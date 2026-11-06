"""
Đếm số file PDF và số trang phải OCR của VJOL và VISTA (chỉ CPU, không GPU); logic ở vi_corpus/vjol/ocr_count.py.

Đọc checkpoint đã có (mỗi lần chạy chỉ đọc lại file mới hoặc đã đổi, bộ nhớ đệm ở <data-root>/state/ocr_count/), kiểm kê PDF trong raw, in tóm tắt và ghi
<data-root>/processed/ocr_count/<nguồn>.json và <nguồn>_by_year.csv. Nhanh hơn nhiều so với `extract.py --status` vì không đọc tuần tự toàn bộ nội dung mọi checkpoint.

PDF chưa có checkpoint (vd VISTA chưa trích) được báo riêng và ước theo năm; thêm `--scan-missing N` để chạy N tiến trình `extract.py --no-ocr` quét nốt (CPU) rồi đếm lại
ra số thật. KHÔNG dùng --scan-missing cho nguồn mà lượt GPU đang chạy.

Ví dụ:
    uv run --no-sync python scripts/count_ocr_pages.py --data-root $SEA_DATA_ROOT                         # cả hai nguồn, chỉ đọc
    uv run --no-sync python scripts/count_ocr_pages.py --data-root $SEA_DATA_ROOT --sources vista --scan-missing 8
"""

import argparse
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common.state import setup_logging  # noqa: E402
from vi_corpus.vjol.ocr_count import LAYOUTS, count_source, format_result, scan_missing  # noqa: E402


def main() -> int:
    """Đọc tham số, đếm từng nguồn (quét nốt PDF chưa quét nếu được yêu cầu), in tóm tắt. Trả về 0."""
    p = argparse.ArgumentParser(description="Đếm số file và số trang cần OCR của VJOL / VISTA (CPU).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục dữ liệu gốc.")
    p.add_argument("--sources", default="vjol,vista", help="Các nguồn cần đếm, cách nhau dấu phẩy (mặc định %(default)s).")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4), help="Số luồng kiểm kê và tiến trình đọc checkpoint (mặc định %(default)s).")
    p.add_argument("--no-cache", action="store_true", help="Bỏ bộ nhớ đệm: đọc lại mọi checkpoint (chậm, dùng khi nghi bộ nhớ đệm sai).")
    p.add_argument("--scan-missing", type=int, default=0, metavar="N", help="Quét nốt PDF chưa có checkpoint bằng N tiến trình extract.py --no-ocr (CPU), rồi đếm lại.")
    p.add_argument("--gpus", type=int, default=4, help="Số GPU để ước giờ OCR (mặc định %(default)s).")
    p.add_argument("--pages-per-sec", type=float, default=1.55, help="Trang/giây/GPU để ước giờ (mặc định %(default)s, đo trên stbook, O-22).")
    args = p.parse_args()
    keys = [k.strip() for k in args.sources.split(",") if k.strip()]
    unknown = [k for k in keys if k not in LAYOUTS]
    if unknown:
        p.error(f"Nguồn không biết: {unknown}; chọn trong {sorted(LAYOUTS)}")
    setup_logging(args.data_root / "logs", "count_ocr_pages")
    repo_root = Path(__file__).resolve().parents[1]
    for key in keys:
        result = count_source(key, args.data_root, args.workers, not args.no_cache)
        if args.scan_missing and result["total"]["missing_files"]:
            print(f"[{key}] {result['total']['missing_files']:,} PDF chưa quét: chạy {args.scan_missing} tiến trình extract.py --no-ocr (log: logs/{key}_count_cpu*.log)", flush=True)
            bad = scan_missing(key, args.data_root, args.scan_missing, repo_root)
            if bad:
                print(f"[{key}] {bad} tiến trình thoát mã khác 0 (xem log; mã 1 thường chỉ là có PDF lỗi)")
            result = count_source(key, args.data_root, args.workers, not args.no_cache)
        print(format_result(result, args.gpus, args.pages_per_sec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
