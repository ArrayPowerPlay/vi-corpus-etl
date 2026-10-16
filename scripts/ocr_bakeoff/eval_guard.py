"""
Đo bộ kiểm từng trang (vi_corpus/ocr_bakeoff/guard.py) trên đầu ra thật của đợt so sánh engine OCR (O-18).

Với mỗi engine VLM trong runs/ và mỗi trang bộ A (có đáp án), "hỏng hẳn" nghĩa là: CER của trang (chặn trần 100%) >= --bad-cer,
hoặc finish_reason = "length", hoặc rỗng. Bộ kiểm báo trang nào thì so với nhãn này:
    bắt được (recall) = trang hỏng hẳn mà bộ kiểm báo / mọi trang hỏng hẳn;
    báo nhầm          = trang tốt (CER < --good-cer, không cắt cụt) mà bộ kiểm báo / mọi trang tốt.
Trang hỏng mà bộ kiểm bỏ sót được liệt kê kèm CER và độ dài (thường là đọc sai lẻ tẻ hoặc đáp án thiếu chữ: bộ kiểm không
được thiết kế để bắt). Bộ B (ảnh quét thật, không đáp án) chỉ đếm số trang bị báo theo lý do để xem bằng mắt: ở đó phần lớn
trang bị báo là báo nhầm. Ghi report/guard_eval.json và in bảng tóm tắt. Chỉ dùng CPU.
Thêm hai chức năng (O-19), cùng chạy bằng CPU:
    --cascade a,b,c : mô phỏng chạy lần lượt các engine a -> b -> c, mỗi trang lấy đầu ra của engine đầu tiên không bị bộ kiểm
                      báo (engine cuối được nhận bất kể); so CER (chặn trần, theo trang dọc / mọi trang), số trang cắt cụt và
                      số trang còn bị báo sau cùng với engine đầu chạy một mình. Dùng lại đầu ra có sẵn nên là cận gần đúng.
    --show-missed N : ghi report/guard_missed.html, N trang hỏng mà mọi engine đang đo đều bị bộ kiểm bỏ sót, đặt cạnh nhau
                      (ảnh, đáp án, đầu ra) để xem đó là lỗi đáp án / trang khó hay lỗi engine.
Ví dụ: uv run python scripts/ocr_bakeoff/eval_guard.py --data-root $SEA_DATA_ROOT
       uv run python scripts/ocr_bakeoff/eval_guard.py --data-root $SEA_DATA_ROOT --cascade qwen3vl_8b_mt4k,qwen3vl_8b_rp110,baseline --show-missed 15
"""

import argparse
import html
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.ocr_bakeoff.guard import guard_page  # noqa: E402
from vi_corpus.ocr_bakeoff.metrics import micro_rate, page_scores  # noqa: E402
from vi_corpus.ocr_bakeoff.pages import is_landscape, read_pages  # noqa: E402
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


def simulate_cascade(bakeoff: Path, tiers: list[str]) -> dict:
    """
    Mô phỏng chuỗi engine `tiers` trên bộ A: mỗi trang lấy đầu ra của engine đầu tiên không bị bộ kiểm báo (engine cuối
    được nhận bất kể), rồi tính CER và số trang lỗi còn lại, kèm số liệu của engine đầu chạy một mình để so.

    Returns:
        {"tiers", "pages_by_tier" (số trang lấy từ mỗi engine), "cascade" và "first_only":
         {"cer_portrait", "cer_all" (CER gộp, lỗi mỗi trang chặn trần), "truncated" (finish_reason = length),
          "flagged" (còn bị bộ kiểm báo), "missing" (thiếu đầu ra)}}.
    """
    pages = [p for p in read_pages(bakeoff) if p["set"] == "A"]
    refs = {p["page_id"]: (bakeoff / p["gt"]).read_text(encoding="utf-8") for p in pages}
    runs, mt = {}, {}
    for name in tiers:
        run = bakeoff / "runs" / name
        runs[name] = latest_outputs(run)
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8")) if (run / "run_info.json").exists() else {}
        mt[name] = info.get("config", {}).get("max_tokens")
    used: Counter = Counter()

    def score(pick) -> dict:
        """Số đo của cách chọn đầu ra `pick(page) -> (engine, rec)` trên mọi trang bộ A."""
        num_p, den_p, num_a, den_a = [], [], [], []
        trunc = flagged = missing = 0
        for p in pages:
            name, rec = pick(p)
            if rec is None:
                missing += 1
                continue
            sc = page_scores(rec.get("text") or "", refs[p["page_id"]])
            err = min(sc["char_dist"], sc["ref_chars"])
            num_a.append(err), den_a.append(sc["ref_chars"])
            if not is_landscape(p.get("size")):
                num_p.append(err), den_p.append(sc["ref_chars"])
            trunc += rec.get("finish_reason") == "length"
            flagged += bool(guard_page(rec, mt[name]))
        return {"cer_portrait": micro_rate(num_p, den_p), "cer_all": micro_rate(num_a, den_a), "truncated": trunc,
                "flagged": flagged, "missing": missing}

    def first_only(p):
        """Chỉ engine đầu."""
        return tiers[0], runs[tiers[0]].get(p["page_id"])

    def cascade(p):
        """Engine đầu tiên không bị báo; engine cuối nhận bất kể."""
        for name in tiers:
            rec = runs[name].get(p["page_id"])
            if rec is not None and (name == tiers[-1] or not guard_page(rec, mt[name])):
                used[name] += 1
                return name, rec
        return tiers[-1], None

    out = {"tiers": tiers, "first_only": score(first_only)}
    out["cascade"] = score(cascade)
    out["pages_by_tier"] = {n: used[n] for n in tiers}
    return out


def write_missed_html(bakeoff: Path, engines: list[str], res: dict, show: int) -> Path | None:
    """
    Ghi report/guard_missed.html: tối đa `show` trang hỏng mà MỌI engine trong `engines` đều bị bộ kiểm bỏ sót, theo thứ tự
    CER giảm dần (trung bình các engine), đặt cạnh nhau ảnh, đáp án và đầu ra của mọi engine. Trả về đường dẫn, hoặc None.
    """
    miss = [{m["page_id"]: m for m in res[n]["missed"]} for n in engines]
    common = set(miss[0]).intersection(*miss[1:]) if miss else set()
    if not common:
        return None
    pages = {p["page_id"]: p for p in read_pages(bakeoff) if p["set"] == "A"}
    order = sorted(common, key=lambda pid: -sum(m[pid]["cer"] for m in miss) / len(miss))[:show]
    recs = {n: latest_outputs(bakeoff / "runs" / n) for n in engines}
    e = html.escape
    parts = ["<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>Trang bị bộ kiểm bỏ sót</title><style>"
             "body{font-family:system-ui,sans-serif;margin:16px}.grid{display:grid;gap:6px}"
             ".grid pre{white-space:pre-wrap;font-size:12px;max-height:500px;overflow:auto;border:1px solid #ddd;padding:4px;margin:0}"
             ".grid img{max-width:100%}</style></head><body>",
             f"<h1>{len(common)} trang hỏng mà cả {len(engines)} engine đều bị bộ kiểm bỏ sót (hiện {len(order)})</h1>"]
    for pid in order:
        p = pages[pid]
        cells = [f"<div><img loading='lazy' src='../{e(p['image'])}'></div>",
                 f"<div><b>đáp án</b><pre>{e((bakeoff / p['gt']).read_text(encoding='utf-8'))}</pre></div>"]
        cells += [f"<div><b>{e(n)}</b> (CER {miss[i][pid]['cer']:.0%})<pre>{e((recs[n].get(pid) or {}).get('text') or '(không có)')}</pre></div>"
                  for i, n in enumerate(engines)]
        src = p["source"]
        parts.append(f"<details open><summary>{e(pid)} — {e(str(src.get('file') or src.get('title')))} trang {src.get('page')}"
                     f"</summary><div class='grid' style='grid-template-columns:repeat({len(cells)},minmax(220px,1fr))'>"
                     f"{''.join(cells)}</div></details>")
    path = bakeoff / "report" / "guard_missed.html"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(parts) + "</body></html>", encoding="utf-8")
    return path


def main() -> int:
    """Đọc tham số, đo, ghi report/guard_eval.json, in bảng tóm tắt."""
    p = argparse.ArgumentParser(description="Đo bộ kiểm từng trang OCR trên đầu ra thật (O-18).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    p.add_argument("--bakeoff-dir", type=Path, default=None, help="Mặc định <data-root>/processed/ocr_bakeoff.")
    p.add_argument("--engine", action="append", default=None,
                   help="Engine cần đo (mặc định mọi runs/qwen3vl_8b*).")
    p.add_argument("--bad-cer", type=float, default=0.5, help="CER trang (chặn trần 1) từ mức này là hỏng hẳn.")
    p.add_argument("--good-cer", type=float, default=0.3, help="CER trang dưới mức này là trang tốt.")
    p.add_argument("--cascade", default=None, help="Mô phỏng chuỗi engine, vd qwen3vl_8b_mt4k,qwen3vl_8b_rp110,baseline.")
    p.add_argument("--show-missed", type=int, default=0, help="Ghi N trang hỏng mà mọi engine đều bị bỏ sót ra guard_missed.html.")
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
    if args.cascade:
        tiers = [t.strip() for t in args.cascade.split(",") if t.strip()]
        sim = simulate_cascade(bakeoff, tiers)
        (rep / "guard_cascade.json").write_text(json.dumps(sim, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nMô phỏng chuỗi {' -> '.join(tiers)}:")
        print(f"  số trang lấy từ mỗi engine: {sim['pages_by_tier']}")
        for key, title in (("first_only", f"{tiers[0]} một mình"), ("cascade", "cả chuỗi")):
            c = sim[key]
            print(f"  {title:34s} CER trang dọc {100 * c['cer_portrait']:.2f}% | CER mọi trang {100 * c['cer_all']:.2f}% | "
                  f"cắt cụt {c['truncated']} | còn bị báo {c['flagged']} | thiếu {c['missing']}")
    if args.show_missed:
        path = write_missed_html(bakeoff, engines, res, args.show_missed)
        print(f"Trang bị mọi engine bỏ sót: {path or 'không có'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
