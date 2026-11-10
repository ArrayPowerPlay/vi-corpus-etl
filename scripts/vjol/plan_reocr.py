"""
Đo CPU trước khi OCR lại cả bài cho VJOL (J-42 (2), P1): bao nhiêu bài / trang lệch dấu, rơi chữ, dính liền, rác; OCR lại hết thì mất bao nhiêu giờ GPU (mẫu và cả kho). Logic ở vi_corpus/vjol/plan_reocr.py.

Ghi <data-root>/processed/vjol_reocr_plan/ (summary.json, reocr_plan.csv). Không cần GPU hay vLLM; mở PDF gốc để lấy (producer, creator) (--no-pdf-meta để bỏ). Nếu `summary.json` có `warning`
(OCR lại > 15% số trang) thì báo lại chủ dự án chọn ngưỡng trước khi chạy GPU.
    uv run python scripts/vjol/plan_reocr.py --data-root data
    tar czf vjol_reocr_plan.tar.gz -C data/processed vjol_reocr_plan
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common.state import setup_logging  # noqa: E402
from vi_corpus.vjol.plan_reocr import CORPUS_PDFS, plan  # noqa: E402


def main() -> int:
    """Đọc tham số, chạy phép đo, in tóm tắt. Trả về 1 nếu có cảnh báo OCR lại quá nhiều trang (để lệnh sau `;` vẫn chạy, không nối bằng &&)."""
    p = argparse.ArgumentParser(description="Đo CPU kế hoạch OCR lại cả bài cho VJOL (J-42 (2)).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    p.add_argument("--root", type=Path, default=None, help="Thư mục raw/vjol (mặc định <data-root>/raw/vjol), để đọc metadata tạp chí.")
    p.add_argument("--syllables", type=Path, default=None, help="Tập âm tiết (mặc định <data-root>/processed/ocr_bakeoff/syllables.json).")
    p.add_argument("--no-pdf-meta", action="store_true", help="Không mở PDF gốc để lấy producer / creator.")
    p.add_argument("--corpus-pdfs", type=int, default=CORPUS_PDFS, help="Số PDF của cả kho để nhân ước lượng.")
    args = p.parse_args()
    setup_logging(args.data_root / "logs", "vjol_plan_reocr")
    out = plan(args.data_root, args.root, args.syllables, not args.no_pdf_meta, args.corpus_pdfs)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    print(f"Xong: {out}")
    print(f"{summary['docs']} bài, {summary['text_pages']} trang chữ; bài / trang trúng từng tín hiệu: {summary['docs_with_signal']} / {summary['pages_with_signal']}")
    print(f"Mặc định {summary['assumptions']}: OCR lại {summary['default']}")
    print(f"Bài chỉ có gạch nối thay ư (OCR không sửa được, J-32): {summary['hyphen_only_docs']}")
    if summary["warning"]:
        print(summary["warning"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
