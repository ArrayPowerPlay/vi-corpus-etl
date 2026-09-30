"""
Xử lý thô giáo trình: kiểm kê file, trích text từng trang (PDF, pptx; trang scan thì OCR), rồi xuất Parquet.

Đọc <data-root>/raw/giao_trinh (hoặc --root), ghi:
    interim/giao_trinh_text/<sha256>.json          text từng trang + method (checkpoint theo file)
    processed/giao_trinh/giao_trinh.parquet        bản ghi CORPUS_SCHEMA, mỗi tài liệu một dòng
Có checkpoint theo file: bị ngắt thì chạy lại đúng lệnh cũ. Xuất Parquet chạy tự động ở cuối mỗi lần
chạy (không cần cờ riêng). OCR cần `uv sync --group ocr`; thiếu thì trang scan để needs_ocr.
Ví dụ:
    uv run python scripts/giao_trinh/extract.py --data-root /duong/dan/data --limit-files 5 --no-ocr
    uv run --group ocr python scripts/giao_trinh/extract.py --data-root /duong/dan/data --device cuda
    uv run python scripts/giao_trinh/extract.py --data-root /duong/dan/data --status
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Thay thư mục script (sys.path[0]) bằng gốc repo: profile.py cạnh đây sẽ che module `profile` của stdlib
# (paddle/cProfile import nó khi nạp OCR) và làm hỏng việc nạp mô hình.
sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.common.registry import SOURCES  # noqa: E402
from vi_corpus.common.state import acquire_run_lock, setup_logging  # noqa: E402
from vi_corpus.giao_trinh.extract import make_ocr_getter, run_extract, status  # noqa: E402
from vi_corpus.giao_trinh.records import OUT_PATH, export_parquet, iter_records  # noqa: E402


def main() -> int:
    """Đọc tham số, rồi in tiến độ (--status) hoặc trích text + xuất Parquet. Trả về 1 nếu có file lỗi."""
    parser = argparse.ArgumentParser(description="Xử lý thô giáo trình: trích text từng trang và xuất Parquet.")
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")),
                        help="Thư mục gốc dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).")
    parser.add_argument("--root", type=Path, default=None,
                        help="Thư mục giáo trình (mặc định <data-root>/raw/giao_trinh).")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Thiết bị chạy VietOCR cho trang scan (mặc định cuda).")
    parser.add_argument("--no-ocr", action="store_true", help="Không OCR: trang scan để needs_ocr.")
    parser.add_argument("--limit-files", type=int, default=None, help="Chỉ xử lý N file đầu (chạy thử).")
    parser.add_argument("--status", action="store_true", help="Chỉ in tiến độ rồi thoát.")
    args = parser.parse_args()
    spec = SOURCES["giao_trinh"]
    root = args.root or args.data_root / spec.path
    if not root.is_dir():
        parser.error(f"Không thấy thư mục {root}")

    if args.status:
        s = status(root, args.data_root)
        pages = ", ".join(f"{m}={n}" for m, n in sorted(s["pages"].items())) or "chưa có"
        print(f"[giao_trinh] {s['files']} file trong raw; {s['checkpoints']} đã trích text "
              f"({s['errors']} lỗi, {s['skipped']} bỏ qua); trang theo method: {pages}")
        return 0
    _lock = acquire_run_lock(args.data_root / "state" / "giao_trinh_extract")  # noqa: F841 — giữ khoá
    setup_logging(args.data_root / "logs", "giao_trinh_extract")
    failed = run_extract(root, args.data_root, make_ocr_getter(args.device, not args.no_ocr),
                         ocr_enabled=not args.no_ocr, limit_files=args.limit_files)
    out = args.data_root / OUT_PATH
    n = export_parquet(iter_records(spec, args.data_root), out)
    logging.getLogger("vi_corpus").info("Đã xuất %d bản ghi ra %s", n, out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
