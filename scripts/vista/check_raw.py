"""
Kiểm tra bản chép VISTA trên server (bước V0): so <data-root>/raw/vista với metadata/pdf_download_log.jsonl (thiếu file, sai kích thước, tuỳ chọn sai sha256).

Chạy trên CPU, không cần mạng. Thoát với mã 1 nếu có file thiếu / sai kích thước / sai sha256 (bản chép chưa đủ hoặc dở dang: chạy lại rclone copy).

Ví dụ:
    uv run python scripts/vista/check_raw.py --data-root /duong/dan/data              # nhanh: đếm file và so số byte
    uv run python scripts/vista/check_raw.py --data-root /duong/dan/data --sha256     # chậm: băm lại cả kho (~234 GB)
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.vista.raw_check import check_raw  # noqa: E402


def main() -> int:
    """Đọc tham số, chạy check_raw, in kết quả JSON. Trả về 1 nếu bản chép không khớp nhật ký."""
    p = argparse.ArgumentParser(description="Kiểm tra raw/vista so với pdf_download_log.jsonl.")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục gốc dữ liệu (mặc định SEA_DATA_ROOT hoặc ./data).")
    p.add_argument("--root", type=Path, default=None, help="Thư mục VISTA (mặc định <data-root>/raw/vista).")
    p.add_argument("--log", type=Path, default=None, help="File pdf_download_log.jsonl (mặc định <root>/metadata/pdf_download_log.jsonl).")
    p.add_argument("--sha256", action="store_true", help="Băm lại mọi file và so với nhật ký (rất chậm).")
    p.add_argument("--workers", type=int, default=8, help="Số luồng băm.")
    args = p.parse_args()
    root = args.root or args.data_root / "raw/vista"
    if not root.is_dir():
        p.error(f"Không thấy thư mục {root}")
    result = check_raw(root, args.log, verify_sha=args.sha256, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 1 if result["missing"] or result["size_mismatch"] or result["sha_mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
