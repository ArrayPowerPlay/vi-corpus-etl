"""
Xử lý thô VISTA (bài báo khoa học sti.vista.gov.vn): chọn mẫu PDF chia đều theo năm, trích text từng trang (trang scan OCR bằng chuỗi engine VLM), rồi xuất Parquet.

Đọc <data-root>/raw/vista (thư mục năm <publication_id>.pdf, unknown/, metadata/), ghi interim/vista_text/<sha256>.json (checkpoint theo PDF) và processed/vista/vista.parquet.
Dùng chung toàn bộ logic với VJOL (vi_corpus/vjol/cli.py); file này chỉ truyền bố cục VISTA. Xem tham số bằng --help; nhiều GPU: scripts/vista/run_extract_vlm.sh.

Ví dụ:
    uv run python scripts/vista/extract.py --data-root /duong/dan/data --sample-docs 20 --no-ocr    # thử nhanh, không GPU
    uv run python scripts/vista/extract.py --data-root /duong/dan/data --status
"""

import sys
from pathlib import Path

# Thay thư mục script (sys.path[0]) bằng gốc repo, như scripts/vjol/extract.py.
sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.vista.layout import VISTA_LAYOUT  # noqa: E402
from vi_corpus.vista.records import iter_records  # noqa: E402
from vi_corpus.vjol.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(VISTA_LAYOUT, iter_records))
