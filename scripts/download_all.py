"""
Tải phần tiếng Việt của cả 3 bộ: SEA-Instruct-2602 → SEA-PILE-v2 → SEA-LION-Pile v1.

Chạy lần lượt từng bộ (bộ nhỏ trước). Có checkpoint: nếu bị ngắt, chạy lại đúng lệnh này,
các file đã xong ở mọi bộ đều được bỏ qua.
Ví dụ:
    python scripts/download_all.py --data-root /duong/dan/data --workers 8
    python scripts/download_all.py --data-root /duong/dan/data --status
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được vi_corpus

from vi_corpus.download.cli import main  # noqa: E402
from vi_corpus.download.datasets import DATASETS  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(list(DATASETS), "Tải phần tiếng Việt của cả 3 bộ SEA."))
