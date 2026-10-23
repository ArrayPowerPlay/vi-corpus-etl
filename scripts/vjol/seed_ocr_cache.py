"""
Nạp bộ nhớ đệm OCR từ checkpoint trích text hiện có (J-25 Bước 2) trước khi trích lại sau khi đổi EXTRACT_VERSION: các trang đã OCR không phải OCR lại.

Chữ ký phải khớp lệnh trích text sẽ chạy: cùng --tiers và cùng việc có xoay thử trang hay không (mặc định có; --no-rotate nếu lần trích sau chạy --no-rotate). Chạy MỘT LẦN, trước
`run_extract_vlm.sh`; chạy lại vô hại (ghi đè bằng đúng nội dung). Chỉ CPU, vài phút.
    uv run python scripts/vjol/seed_ocr_cache.py --data-root data
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common.ocr_cache import ocr_signature  # noqa: E402
from vi_corpus.ocr_bakeoff.cascade import DEFAULT_TIERS  # noqa: E402
from vi_corpus.vjol.ocr_seed import seed_cache  # noqa: E402


def main() -> int:
    """Đọc tham số, nạp cache, in số trang đã nạp. Trả về 0."""
    p = argparse.ArgumentParser(description="Nạp cache OCR VJOL từ checkpoint hiện có.")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục gốc dữ liệu.")
    p.add_argument("--tiers", default=",".join(DEFAULT_TIERS), help="Chuỗi engine của lần trích cũ và lần trích sắp chạy (mặc định %(default)s).")
    p.add_argument("--no-rotate", action="store_true", help="Lần trích sắp chạy KHÔNG xoay thử trang (chữ ký không có :rot, nạp cả trang baseline / flagged).")
    args = p.parse_args()
    signature = ocr_signature([t.strip() for t in args.tiers.split(",") if t.strip()], not args.no_rotate)
    stats = seed_cache(args.data_root, signature)
    print(f"Chữ ký: {signature}")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
