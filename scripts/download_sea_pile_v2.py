"""
Tải phần tiếng Việt của SEA-PILE-v2 (aisingapore/SEA-PILE-v2, thư mục vi/) về data/raw/sea_pile_v2/.

Có checkpoint: nếu bị ngắt giữa chừng, chạy lại đúng lệnh này để tải tiếp.
Ví dụ:
    python scripts/download_sea_pile_v2.py --data-root /duong/dan/data --workers 8
    python scripts/download_sea_pile_v2.py --data-root /duong/dan/data --status
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được vi_corpus

from vi_corpus.download.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["sea_pile_v2"], "Tải phần tiếng Việt của SEA-PILE-v2 (aisingapore/SEA-PILE-v2, thư mục vi/)."))
