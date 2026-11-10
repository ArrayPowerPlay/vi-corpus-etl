"""
Xử lý thô VJOL: chọn mẫu PDF chia đều theo năm, trích text từng trang (trang scan OCR bằng chuỗi engine VLM, O-20), rồi xuất Parquet.

Đọc <data-root>/raw/vjol (hoặc --root: thư mục năm + metadata/), ghi interim/vjol_text/<sha256>.json (checkpoint theo PDF) và processed/vjol/vjol.parquet.
Toàn bộ logic nằm ở vi_corpus/vjol/cli.py (dùng chung với VISTA, scripts/vista/extract.py); file này chỉ truyền bố cục VJOL. Xem tham số bằng --help.

Ví dụ:
    uv run python scripts/vjol/extract.py --data-root /duong/dan/data --sample-docs 20 --no-ocr          # thử nhanh, không GPU
    uv run --no-sync python scripts/vjol/extract.py --data-root /duong/dan/data --sample-docs 3000 --seed 0 \
        --gpu 0 --serve --vllm-bin $(which vllm)                                                         # một GPU
    uv run python scripts/vjol/extract.py --data-root /duong/dan/data --status
    uv run python scripts/vjol/extract.py --data-root /duong/dan/data --export-only
Nhiều GPU: scripts/vjol/run_extract_vlm.sh.
"""

import sys
from pathlib import Path

# Thay thư mục script (sys.path[0]) bằng gốc repo: profile.py cạnh đây (giáo trình) sẽ che module `profile` của stdlib.
sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.vjol.cli import main  # noqa: E402
from vi_corpus.vjol.extract import VJOL_LAYOUT  # noqa: E402
from vi_corpus.vjol.records import iter_records  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(VJOL_LAYOUT, iter_records))
