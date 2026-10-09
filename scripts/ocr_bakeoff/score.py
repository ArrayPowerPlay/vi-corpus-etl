"""
Bước 3 của đợt so sánh engine OCR (F-11): chấm điểm và áp quy tắc chọn đã chốt trước khi chạy.

Đọc <bakeoff-dir>/runs/*/outputs.jsonl, ghi <bakeoff-dir>/report/{summary.json, per_page.csv, report.html}. Chấm lại
bao nhiêu lần cũng được (không gọi engine). --ppl tính thêm perplexity Qwen3-0.6B cho bộ B (cần GPU, lưu vào
runs/<engine>/ppl.jsonl, lần sau không tính lại). Đặt --gpu-hour-budget khi chủ dự án đã chốt ngân sách, nếu không cổng
tốc độ để "chưa kiểm" và kết luận chỉ là tạm.
Ví dụ:
    uv run python scripts/ocr_bakeoff/score.py --data-root /duong/dan/data
    uv run python scripts/ocr_bakeoff/score.py --data-root /duong/dan/data --ppl --gpu-hour-budget 30
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.common.state import setup_logging  # noqa: E402
from vi_corpus.ocr_bakeoff.perplexity import PPL_MODEL  # noqa: E402
from vi_corpus.ocr_bakeoff.scoring import Gates, score_bakeoff  # noqa: E402
from vi_corpus.ocr_bakeoff.syllables import load_syllables  # noqa: E402


def main() -> int:
    """Đọc tham số, (tuỳ chọn) tính perplexity, chấm điểm, in kết luận."""
    p = argparse.ArgumentParser(description="Chấm điểm so sánh engine OCR (F-11).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")),
                   help="Thư mục gốc dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).")
    p.add_argument("--bakeoff-dir", type=Path, default=None,
                   help="Thư mục đợt so sánh (mặc định <data-root>/processed/ocr_bakeoff).")
    p.add_argument("--engine", action="append", default=None, help="Chỉ chấm các engine này (mặc định mọi runs/*).")
    p.add_argument("--baseline", default="baseline", help="Tên engine baseline (mặc định baseline).")
    p.add_argument("--syllables", type=Path, default=None,
                   help="Tập âm tiết (mặc định <bakeoff-dir>/syllables.json nếu có).")
    p.add_argument("--ppl", action="store_true", help="Tính perplexity Qwen3-0.6B trước khi chấm.")
    p.add_argument("--ppl-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--ppl-model", default=PPL_MODEL, help=f"Mô hình đo perplexity (mặc định {PPL_MODEL}).")
    p.add_argument("--num-gpus", type=int, default=4, help="Số GPU dùng để quy ra giờ GPU toàn kho (mặc định 4).")
    p.add_argument("--gpu-hour-budget", type=float, default=None, help="Ngân sách giờ GPU toàn kho.")
    p.add_argument("--corpus-pages", type=int, default=None,
                   help="Tổng số trang cần OCR (mặc định lấy từ selection.json: stbook + giáo trình cần OCR).")
    p.add_argument("--special-recall-min", type=float, default=Gates.special_recall_min)
    p.add_argument("--two-col-gap-max", type=float, default=Gates.two_col_gap_max)
    p.add_argument("--bad-rate-max", type=float, default=Gates.bad_rate_max)
    p.add_argument("--trunc-rate-max", type=float, default=Gates.trunc_rate_max)
    args = p.parse_args()
    bakeoff = args.bakeoff_dir or args.data_root / "processed/ocr_bakeoff"
    if not (bakeoff / "pages.jsonl").exists():
        p.error(f"Chưa có {bakeoff}/pages.jsonl: chạy select_pages.py trước")
    setup_logging(args.data_root / "logs", "ocr_bakeoff_score")
    engines = args.engine or sorted(d.name for d in (bakeoff / "runs").glob("*") if d.is_dir())
    if not engines:
        p.error(f"Chưa có kết quả engine nào trong {bakeoff}/runs: chạy run_engine.py trước")
    if args.ppl:
        from vi_corpus.ocr_bakeoff.perplexity import score_runs  # nạp torch / transformers chỉ khi cần

        try:
            score_runs(bakeoff, engines, model_id=args.ppl_model, device=args.ppl_device)
        except Exception as e:  # noqa: BLE001 - thiếu mạng / GPU thì vẫn chấm, chỉ thiếu perplexity
            print(f"Không tính được perplexity ({type(e).__name__}: {e}); chấm tiếp không có số đo này")
    syl_path = args.syllables or bakeoff / "syllables.json"
    vocab = load_syllables(syl_path) if syl_path.exists() else None
    if vocab is None:
        print(f"Không có {syl_path}: bỏ số đo âm tiết lạ của bộ B (chạy build_syllables.py)")
    gates = Gates(args.special_recall_min, args.two_col_gap_max, args.bad_rate_max, args.trunc_rate_max,
                  args.gpu_hour_budget)
    s = score_bakeoff(bakeoff, engines, baseline=args.baseline, vocab=vocab, num_gpus=args.num_gpus, gates=gates,
                      corpus_pages=args.corpus_pages)
    print(json.dumps(s["decision"], ensure_ascii=False, indent=2))
    print(f"Báo cáo: {bakeoff / 'report/report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
