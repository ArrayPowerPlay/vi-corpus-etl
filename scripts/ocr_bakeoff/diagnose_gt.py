"""
Chẩn đoán đáp án bộ A của đợt so sánh engine OCR (O-10, O-11): CER cao là do engine hay do đáp án?

Chạy trên CPU sau khi đã có runs/*/outputs.jsonl (chỉ đọc, không gọi engine). Kiểm trang đôi, số trang lẫn giữa trang,
tiêu đề / chân trang trong đáp án mà đa số engine bỏ; tính lại CER trên đáp án đã bỏ các phần nghi, liệt kê trang mà mọi
engine đều CER cao. Ghi <bakeoff-dir>/report/gt_diag.json và gt_diag.csv. Chi tiết: vi_corpus/ocr_bakeoff/gt_diag.py.
Ví dụ:
    uv run python scripts/ocr_bakeoff/diagnose_gt.py --data-root /duong/dan/data
"""

import argparse
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.ocr_bakeoff.gt_diag import HIGH_CER, diagnose_gt  # noqa: E402


def main() -> int:
    """Đọc tham số, chẩn đoán, in tóm tắt."""
    p = argparse.ArgumentParser(description="Chẩn đoán đáp án bộ A của đợt so sánh OCR (O-11).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")),
                   help="Thư mục gốc dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).")
    p.add_argument("--bakeoff-dir", type=Path, default=None,
                   help="Thư mục đợt so sánh (mặc định <data-root>/processed/ocr_bakeoff).")
    p.add_argument("--engine", action="append", default=None, help="Chỉ dùng các engine này (mặc định mọi runs/*).")
    p.add_argument("--high-cer", type=float, default=HIGH_CER, help=f"Ngưỡng CER của mục trang khó (mặc định {HIGH_CER}).")
    args = p.parse_args()
    bakeoff = args.bakeoff_dir or args.data_root / "processed/ocr_bakeoff"
    if not (bakeoff / "pages.jsonl").exists():
        p.error(f"Chưa có {bakeoff}/pages.jsonl: chạy select_pages.py trước")
    r = diagnose_gt(bakeoff, args.engine, args.high_cer)
    if not r["engines"]:
        print(f"Chưa có kết quả engine nào trong {bakeoff}/runs: chạy run_engine.py trước")
        return 1
    print(f"{r['pages']} trang bộ A; trang nghi theo lý do: {r['reasons'] or 'không có'}; "
          f"{r['trimmed_lines']} dòng đáp án bị coi là thừa")
    print(f"{'engine':16s} {'CER nguyên':>11s} {'bỏ trang ngang':>15s} {'đã sửa đáp án':>14s}")
    for n, c in r["cer"].items():
        print(f"{n:16s} {100 * c['original']:10.2f}% {100 * c['without_landscape']:14.2f}% {100 * c['trimmed']:13.2f}%")
    if r["ground_truth_share"] is not None:
        print(f"Engine tốt nhất ({r['best_engine']}): sửa đáp án giảm {100 * r['ground_truth_share']:.1f}% CER (tương đối)")
    print(f"{len(r['hard_pages'])} trang mọi engine đều CER > {args.high_cer:.0%} (đáp án sai hoặc trang khó), "
          f"xem {bakeoff / 'report/gt_diag.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
