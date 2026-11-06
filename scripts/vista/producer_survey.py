"""
Khảo sát producer của PDF VISTA (VS-05), CPU, không cần GPU: tìm producer có lớp chữ mất dấu / hỏng mà luật "OCR cả bài" chưa phủ.

Đọc mẫu PDF chia đều theo năm ở <data-root>/raw/vista, ghi processed/vista_producer_survey/{producers.csv,survey.json}. Dòng đầu của bảng là producer đáng thêm
vào OLD_OCR_PRODUCERS (vi_corpus/vista/layout.py) nhất (cột uncovered_suspect lớn). Nên chạy với mẫu lớn (20.000) trước khi trích text cả kho.

Ví dụ:
    uv run --no-sync python scripts/vista/producer_survey.py --data-root /duong/dan/data --sample 20000 --workers 16
"""

import argparse
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.vista.producer_survey import run_survey  # noqa: E402


def main() -> int:
    """Đọc tham số, chạy khảo sát, in tóm tắt và 15 producer đầu bảng. Trả về 0."""
    p = argparse.ArgumentParser(description="Khảo sát producer PDF VISTA so với luật OCR cả bài.")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục gốc dữ liệu (mặc định SEA_DATA_ROOT hoặc ./data).")
    p.add_argument("--root", type=Path, default=None, help="Thư mục VISTA (mặc định <data-root>/raw/vista).")
    p.add_argument("--sample", type=int, default=20000, help="Số PDF trong mẫu chia đều theo năm (mặc định 20000).")
    p.add_argument("--seed", type=int, default=0, help="Hạt giống chọn mẫu.")
    p.add_argument("--workers", type=int, default=None, help="Số tiến trình (mặc định số CPU).")
    args = p.parse_args()
    root = args.root or args.data_root / "raw/vista"
    if not root.is_dir():
        p.error(f"Không thấy thư mục {root}")
    res = run_survey(root, args.data_root, args.sample, args.seed, args.workers)
    print({k: v for k, v in res.items() if k != "table"})
    print(f"{'producer':32} {'docs':>6} {'years':>10} {'text':>6} {'suspect':>8} {'covered':>8} {'uncovered_suspect':>18}")
    for t in res["table"][:15]:
        sus = "-" if t["suspect_share"] is None else f"{t['suspect_share']:.2f}"
        print(f"{t['producer']:32} {t['docs']:>6} {t['years']:>10} {t['text_share']:>6.2f} {sus:>8} {str(t['covered']):>8} {t['uncovered_suspect']:>18}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
