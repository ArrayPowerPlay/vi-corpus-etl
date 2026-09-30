"""
Tải phần tiếng Việt của SEA-LION-Pile v1 (aisingapore/SEA-PILE-v1, thư mục sea-pile-mc4/vi/) về data/raw/sea_lion_pile_v1/.

Có checkpoint: nếu bị ngắt giữa chừng, chạy lại đúng lệnh này để tải tiếp.
Ví dụ:
    python scripts/sea/download_sea_lion_pile_v1.py --data-root /duong/dan/data --workers 8
    python scripts/sea/download_sea_lion_pile_v1.py --data-root /duong/dan/data --status
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # để import được vi_corpus

from vi_corpus.sea.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["sea_lion_pile_v1"], "Tải phần tiếng Việt của SEA-LION-Pile v1 (aisingapore/SEA-PILE-v1, thư mục sea-pile-mc4/vi/)."))
