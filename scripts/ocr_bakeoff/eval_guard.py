"""
Đo bộ kiểm từng trang (vi_corpus/ocr_bakeoff/guard.py) trên đầu ra thật của đợt so sánh engine OCR (O-18).

Với mỗi engine VLM trong runs/ và mỗi trang bộ A (có đáp án), "hỏng hẳn" nghĩa là: CER của trang (chặn trần 100%) >= --bad-cer,
hoặc finish_reason = "length", hoặc rỗng. Bộ kiểm báo trang nào thì so với nhãn này:
    bắt được (recall) = trang hỏng hẳn mà bộ kiểm báo / mọi trang hỏng hẳn;
    báo nhầm          = trang tốt (CER < --good-cer, không cắt cụt) mà bộ kiểm báo / mọi trang tốt.
Trang hỏng mà bộ kiểm bỏ sót được liệt kê kèm CER và độ dài (thường là đọc sai lẻ tẻ hoặc đáp án thiếu chữ: bộ kiểm không
được thiết kế để bắt). Bộ B (ảnh quét thật, không đáp án) chỉ đếm số trang bị báo theo lý do để xem bằng mắt: ở đó phần lớn
trang bị báo là báo nhầm. Ghi report/guard_eval.json và in bảng tóm tắt. Chỉ dùng CPU.
Ví dụ: uv run python scripts/ocr_bakeoff/eval_guard.py --data-root $SEA_DATA_ROOT
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.ocr_bakeoff.guard import guard_page  # noqa: E402
from vi_corpus.ocr_bakeoff.metrics import page_scores  # noqa: E402
from vi_corpus.ocr_bakeoff.pages import read_pages  # noqa: E402
from vi_corpus.ocr_bakeoff.perplexity import latest_outputs  # noqa: E402


def evaluate(bakeoff: Path, engines: list[str], bad_cer: float, good_cer: float) -> dict:
    """
    Đo bộ kiểm trên các engine (xem docstring module).

    Returns:
        {engine: {"bad_pages", "caught", "good_pages", "false_alarms", "missed": [...], "false_alarm_pages": [...],
                  "reasons_on_bad", "reasons_on_good", "set_b_flagged", "set_b_reasons"}, "_total": {...}}.
    """
    pages = read_pages(bakeoff)
    set_a = [p for p in pages if p["set"] == "A"]
    set_b = [p for p in pages if p["set"] == "B"]
    refs = {p["page_id"]: (bakeoff / p["gt"]).read_text(encoding="utf-8") for p in set_a}
    out: dict = {}
    tot = defaultdict(int)
    for name in engines:
        run = bakeoff / "runs" / name
        recs = latest_outputs(run)
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8")) if (run / "run_info.json").exists() else {}
        max_tokens = info.get("config", {}).get("max_tokens")
        bad = caught = good = false_alarm = 0
        missed, fa_pages = [], []
        r_bad: Counter = Counter()
        r_good: Counter = Counter()
        for p in set_a:
            rec = recs.get(p["page_id"])
            if rec is None:
                continue
            sc = page_scores(rec.get("text") or "", refs[p["page_id"]])
            cer = min(sc["char_dist"], sc["ref_chars"]) / max(sc["ref_chars"], 1)
            cut = rec.get("finish_reason") == "length"
            reasons = guard_page(rec, max_tokens)
            if cer >= bad_cer or cut or sc["hyp_chars"] == 0:
                bad += 1
                if reasons:
                    caught += 1
                    r_bad.update(reasons)
                else:
                    missed.append({"page_id": p["page_id"], "cer": round(cer, 3), "hyp_chars": sc["hyp_chars"],
                                   "ref_chars": sc["ref_chars"], "len_ratio": round(sc["len_ratio"], 2)})
            elif cer < good_cer:
                good += 1
                if reasons:
                    false_alarm += 1
                    r_good.update(reasons)
                    fa_pages.append({"page_id": p["page_id"], "cer": round(cer, 3), "reasons": reasons})
        b_reasons: Counter = Counter()
        b_flagged = 0
        for p in set_b:
            rec = recs.get(p["page_id"])
            reasons = guard_page(rec, max_tokens) if rec else []
            b_flagged += bool(reasons)
            b_reasons.update(reasons)
        out[name] = {"bad_pages": bad, "caught": caught, "good_pages": good, "false_alarms": false_alarm,
                     "missed": missed, "false_alarm_pages": fa_pages, "reasons_on_bad": dict(r_bad),
                     "reasons_on_good": dict(r_good), "set_b_flagged": b_flagged, "set_b_reasons": dict(b_reasons),
                     "set_b_pages": len(set_b)}
        for k in ("bad_pages", "caught", "good_pages", "false_alarms", "set_b_flagged", "set_b_pages"):
            tot[k] += out[name][k]
    out["_total"] = dict(tot)
    return out


def main() -> int:
    """Đọc tham số, đo, ghi report/guard_eval.json, in bảng tóm tắt."""
    p = argparse.ArgumentParser(description="Đo bộ kiểm từng trang OCR trên đầu ra thật (O-18).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    p.add_argument("--bakeoff-dir", type=Path, default=None, help="Mặc định <data-root>/processed/ocr_bakeoff.")
    p.add_argument("--engine", action="append", default=None,
                   help="Engine cần đo (mặc định mọi runs/qwen3vl_8b*).")
    p.add_argument("--bad-cer", type=float, default=0.5, help="CER trang (chặn trần 1) từ mức này là hỏng hẳn.")
    p.add_argument("--good-cer", type=float, default=0.3, help="CER trang dưới mức này là trang tốt.")
    args = p.parse_args()
    bakeoff = args.bakeoff_dir or args.data_root / "processed/ocr_bakeoff"
    if not (bakeoff / "pages.jsonl").exists():
        p.error(f"Chưa có {bakeoff}/pages.jsonl")
    engines = args.engine or sorted(d.name for d in (bakeoff / "runs").glob("qwen3vl_8b*") if d.is_dir())
    res = evaluate(bakeoff, engines, args.bad_cer, args.good_cer)
    rep = bakeoff / "report"
    rep.mkdir(exist_ok=True)
    (rep / "guard_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'engine':24s} {'hỏng':>5s} {'bắt':>5s} {'bỏ sót':>7s} {'tốt':>5s} {'báo nhầm':>9s} {'bộ B báo':>9s}")
    for name in engines:
        r = res[name]
        print(f"{name:24s} {r['bad_pages']:5d} {r['caught']:5d} {r['bad_pages'] - r['caught']:7d} {r['good_pages']:5d} "
              f"{r['false_alarms']:9d} {r['set_b_flagged']:5d}/{r['set_b_pages']}")
    t = res["_total"]
    print(f"TỔNG: bắt {t.get('caught', 0)}/{t.get('bad_pages', 0)} trang hỏng; báo nhầm {t.get('false_alarms', 0)}/"
          f"{t.get('good_pages', 0)} trang tốt; bộ B bị báo {t.get('set_b_flagged', 0)}/{t.get('set_b_pages', 0)}")
    for name in engines:
        for key, title in (("missed", "BỎ SÓT"), ("false_alarm_pages", "BÁO NHẦM")):
            for x in res[name][key]:
                print(f"  [{name}] {title}: {json.dumps(x, ensure_ascii=False)}")
    print(f"Chi tiết: {rep / 'guard_eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
