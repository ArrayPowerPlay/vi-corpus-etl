"""
Bước 3 của đợt so sánh engine OCR (F-11): chấm điểm và áp quy tắc chọn đã chốt trước khi chạy.

Đọc <bakeoff-dir>/runs/*/outputs.jsonl, ghi <bakeoff-dir>/report/{summary.json, per_page.csv, report.html}. Chấm lại
bao nhiêu lần cũng được (không gọi engine). --ppl tính thêm perplexity Qwen3-0.6B cho bộ B (cần GPU, lưu vào
runs/<engine>/ppl.jsonl, lần sau không tính lại). Giờ chạy cả kho có hai mức (O-11): --hour-soft (mặc định 30, mức mềm
của chủ dự án, vượt chỉ cảnh báo) và --hour-budget (trần, vượt thì engine trượt cổng tốc độ; Claude đề xuất 60; không
đặt thì kết luận chỉ là tạm). Cổng bad_rate tính tương đối so với baseline (--bad-rate-excess-max). Số giờ chạy (đã chạy đợt so sánh, ước tính cho cả kho) luôn được ghi vào
report/hours.csv, summary.json ("hours") và report.html.
Ví dụ:
    uv run python scripts/ocr_bakeoff/score.py --data-root /duong/dan/data
    uv run python scripts/ocr_bakeoff/score.py --data-root /duong/dan/data --ppl --hour-budget 60
    uv run python scripts/ocr_bakeoff/score.py --data-root /duong/dan/data --cascade qwen3vl_8b_mt4k,qwen3vl_8b_rp110,baseline
(--cascade, O-20: thêm engine ảo "cascade" ghép đầu ra theo bộ kiểm từng trang; cổng cắt cụt, số trang còn bị báo và số giờ chạy
cả kho của nó tính SAU bước kiểm.)
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
    p.add_argument("--hour-budget", "--gpu-hour-budget", dest="hour_budget", type=float, default=None,
                   help="Trần số giờ server được chạy để OCR cả kho (mọi GPU chạy cùng lúc), vượt thì engine trượt cổng "
                        "tốc độ; không đặt = chưa kiểm. Đề xuất 60 (O-11).")
    p.add_argument("--hour-soft", type=float, default=Gates.hour_soft,
                   help="Mức mềm số giờ chạy cả kho (mặc định 30, N-02): vượt chỉ cảnh báo, không loại.")
    p.add_argument("--corpus-pages", type=int, default=None,
                   help="Tổng số trang cần OCR (mặc định lấy từ selection.json: stbook + giáo trình cần OCR).")
    p.add_argument("--cascade", default=None,
                   help="Chuỗi engine (O-20), vd qwen3vl_8b_mt4k,qwen3vl_8b_rp110,baseline: thêm engine ảo 'cascade' ghép đầu ra "
                        "theo bộ kiểm; cổng cắt cụt và giờ chạy cả kho tính trên chuỗi, không trên đầu ra thô của từng engine.")
    p.add_argument("--special-recall-min", type=float, default=Gates.special_recall_min)
    p.add_argument("--two-col-gap-max", type=float, default=Gates.two_col_gap_max)
    p.add_argument("--bad-rate-excess-max", type=float, default=Gates.bad_rate_excess_max,
                   help="Tỷ lệ trang hỏng được cao hơn baseline tối đa bao nhiêu (0,01 = 1 điểm %%).")
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
    gates = Gates(args.special_recall_min, args.two_col_gap_max, args.bad_rate_excess_max, args.trunc_rate_max,
                  args.hour_budget, args.hour_soft)
    s = score_bakeoff(bakeoff, engines, baseline=args.baseline, vocab=vocab, num_gpus=args.num_gpus, gates=gates,
                      corpus_pages=args.corpus_pages,
                      cascade=[t.strip() for t in args.cascade.split(",") if t.strip()] if args.cascade else None)
    print(json.dumps(s["decision"], ensure_ascii=False, indent=2))
    for name, h in s["hours"].items():
        est = f"{h['corpus_hours']:.1f} giờ cho cả kho" if h["corpus_hours"] is not None else "chưa đo được tốc độ"
        print(f"{name}: đã chạy {h['bakeoff_run_hours']:.2f} giờ cho {h['bakeoff_pages']} trang; {est}")
    print(f"Báo cáo: {bakeoff / 'report/report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
