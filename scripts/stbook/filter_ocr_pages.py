"""
Lọc trang OCR stbook sau khi chạy chuỗi VLM xong (O-30 mục 2 và 3; CPU, không chạy lại GPU); logic ở vi_corpus/stbook/ocr_filter.py.

Chuyển sang `flagged` (văn bản giữ ở `page_info[i].rejected_text`, `pages[i] == ""`):
    * trang baseline có tỷ lệ âm tiết thật dưới --min-real-rate (mặc định 0,85);
    * trang rp110 bị fastText gắn ngôn ngữ khác tiếng Việt trong sách tiếng Việt.
Chạy lại được và đổi ngưỡng được (mỗi lần chạy trả lại trang đã lọc trước rồi lọc lại). Báo cáo ở <data-root>/processed/ocr_page_filter/.
Cần <data-root>/processed/ocr_bakeoff/syllables.json và fastText lid.176 (không tự rơi về heuristic). Không chạy lúc đang OCR.
Ví dụ:
    uv run --no-sync python scripts/stbook/filter_ocr_pages.py --data-root data --dry-run   # xem trước, không sửa file
    uv run --no-sync python scripts/stbook/filter_ocr_pages.py --data-root data
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # để import được vi_corpus

from vi_corpus.common.state import setup_logging  # noqa: E402
from vi_corpus.stbook.ocr_check import DEFAULT_LANG_MODEL  # noqa: E402
from vi_corpus.stbook.ocr_filter import BASELINE_MIN_REAL_RATE, filter_pages  # noqa: E402


def main() -> int:
    """Đọc tham số, lọc trang và in tóm tắt."""
    p = argparse.ArgumentParser(description="Lọc trang OCR stbook (baseline chữ rác, rp110 bịa) sau khi chạy chuỗi VLM.")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục dữ liệu gốc.")
    p.add_argument("--min-real-rate", type=float, default=BASELINE_MIN_REAL_RATE, help="Trang baseline dưới tỷ lệ âm tiết thật này bị loại.")
    p.add_argument("--lang-model", default=DEFAULT_LANG_MODEL, help="fasttext:lid.176 | fasttext:<file.bin> | heuristic (chỉ chọn heuristic khi chủ động muốn).")
    p.add_argument("--syllables", type=Path, default=None, help="Tập âm tiết (mặc định <data-root>/processed/ocr_bakeoff/syllables.json).")
    p.add_argument("--dry-run", action="store_true", help="Chỉ đếm và ghi báo cáo, không sửa file OCR.")
    args = p.parse_args()
    setup_logging(args.data_root / "logs", "filter_ocr_pages")
    summary = filter_pages(args.data_root, args.lang_model, args.syllables, args.min_real_rate, args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
