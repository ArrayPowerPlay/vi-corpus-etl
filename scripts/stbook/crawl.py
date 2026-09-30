"""
Crawl sách stbook.vn theo danh mục (metadata, ảnh bìa, PDF sách miễn phí) về <data-root>/raw/stbook/.

Mọi tham số được chuyển thẳng cho vi_corpus.stbook.crawler.main (xem --help). Ví dụ:
    uv run python scripts/stbook/crawl.py --category kinh-dien --download-pdf --max-pages 2
    uv run python scripts/stbook/crawl.py --download-pdf --out /duong/dan/data/raw/stbook
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # để import được vi_corpus

from vi_corpus.stbook.crawler.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
