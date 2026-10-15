"""
Chấm điểm đợt so sánh engine OCR (F-11) và áp quy tắc chọn đã chốt trước khi chạy (docs/DECISION_LOG.md, F-11).

Đầu vào: <bakeoff>/pages.jsonl, gt/, selection.json, runs/<engine>/outputs.jsonl (và ppl.jsonl nếu đã chạy perplexity).
Đầu ra trong <bakeoff>/report/: summary.json (số đo gộp, cổng loại, kết luận, số giờ chạy), per_page.csv (mỗi trang x
engine một dòng), hours.csv (mỗi engine một dòng: số giờ đã chạy đợt so sánh, tốc độ, số giờ ước tính cho cả kho),
report.html (bảng số đo, cổng, số giờ chạy, bộ B và vài trang hai cột bộ A đặt cạnh nhau để xem bằng mắt).

Trang "hỏng" (bỏ sót / bịa) ở bộ A: không có đầu ra (lỗi hoặc chưa chạy), rỗng, độ dài ngoài ±LEN_TOL so với đáp án,
hoặc vòng lặp (một 4-gram lặp >= LOOP_MIN lần và hơn hai lần số lặp trong đáp án); số trang hỏng được tách theo lý do
(no_output / empty / loop / too_short / too_long). Các cổng đặt tương đối so với baseline (O-11): ứng viên chỉ bị loại
nếu tỷ lệ cao hơn baseline quá bad_rate_excess_max, vì tỷ lệ hỏng của baseline đã phản ánh cả lỗi đáp án. Từ O-16 tách
làm ba: bad_rate chỉ tính lỗi cứng của engine (no_output / empty / loop); too_short (engine bỏ sót chữ) là cổng riêng;
too_long chỉ báo, không loại, vì đáp án là lớp chữ của PDF nên thiếu chữ nằm trong hình mà VLM đọc được (stbook là ảnh
quét, đọc hết chữ trong hình là việc cần làm). Cổng thứ tự đọc hai cột chỉ cảnh báo (O-14 ghi nhận i: bộ A chỉ 8 trang).
Recall ký tự đặc biệt tính theo lớp tương đương (metrics.SPECIAL_CLASSES: nháy cong / gạch dài đúng nếu ra ASCII, "…"
đúng nếu ra "…" hoặc dãy từ 3 dấu chấm), có thêm bản chính xác từng ký tự để đối chiếu.
CER (O-14): lỗi mỗi trang bị chặn trần bằng số ký tự đáp án (CER trang tối đa 100%) để vài trang vòng lặp (đầu ra dài gấp
10-50 lần đáp án) không làm méo số gộp; CER thô giữ lại là cer_raw. Bộ A chủ yếu là slide 16:9 nên xếp hạng (cổng
cer_vs_baseline, decide, bootstrap) chỉ tính trên trang dọc, giống trang sách stbook (thiếu trang dọc thì dùng mọi trang);
CER toàn bộ và theo từng hướng vẫn được báo. Báo cáo đặt cạnh nhau (ảnh, đáp án, đầu ra các engine) cho EXAMPLES_PER_ENGINE
trang CER thô cao nhất mỗi engine để xem đó là lỗi engine hay đáp án thiếu chữ trong hình. Mỗi engine có danh sách WORST_PAGES trang CER cao nhất. Cổng giờ chạy cả kho
có hai mức: hour_soft (mức mềm của chủ dự án, vượt chỉ cảnh báo) và hour_budget (trần, vượt thì engine trượt cổng tốc
độ). Trang "cắt cụt": finish_reason = "length". Tốc độ: phiên có nhiều trang nhất của engine, bỏ WARMUP_PAGES trang đầu, trang/giây = số trang còn lại /
khoảng thời gian giữa t_end của chúng (quá ít trang thì dùng mọi trang); cần >= STEADY_MIN trang mới coi là ổn định.
Số giờ cho cả kho: corpus_hours = tổng số trang cần OCR (selection.json) / (trang/giây x số GPU) / 3600 là số giờ
server chạy khi mỗi GPU chạy một bản engine cùng lúc (đây là số so với ngân sách --hour-budget); corpus_gpu_hours =
corpus_hours x số GPU. Mọi ngưỡng là khởi điểm, chỉnh được bằng tham số.
"""

import csv
import html
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from vi_corpus.ocr_bakeoff.metrics import (
    SPECIAL_CHARS,
    SPECIAL_CLASSES,
    max_ngram_repeat,
    micro_rate,
    normalize,
    page_scores,
    paired_bootstrap,
)
from vi_corpus.ocr_bakeoff.pages import is_landscape, read_pages
from vi_corpus.ocr_bakeoff.perplexity import latest_outputs, latest_outputs_ppl
from vi_corpus.ocr_bakeoff.syllables import unknown_rate

HARD_BAD = ("no_output", "empty", "loop")  # lỗi cứng của engine; too_short / too_long là lệch độ dài so với đáp án (O-16)
LEN_TOL = 0.15
LOOP_MIN = 5
WARMUP_PAGES = 20
WORST_PAGES = 10
EXAMPLES_PER_ENGINE = 3
STEADY_MIN = 200
_CSS = (
    "body{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}"
    "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 6px;font-size:13px}"
    ".pass{background:#d9f2d9}.fail{background:#f8d0d0}.warn{background:#fbecc4}.na{background:#eee}"
    ".grid{display:grid;gap:6px}.grid pre{white-space:pre-wrap;font-size:12px;max-height:600px;overflow:auto;"
    "border:1px solid #ddd;padding:4px;margin:0}.grid img{max-width:100%}"
    "@media (prefers-color-scheme: dark){body{background:#111;color:#eee}td,th{border-color:#444}"
    ".pass{background:#1f3d1f}.fail{background:#4a1f1f}.warn{background:#4a3f1a}.na{background:#333}}"
)


@dataclass
class Gates:
    """Ngưỡng của quy tắc loại (F-11)."""

    special_recall_min: float = 0.95
    two_col_gap_max: float = 0.02  # chỉ cảnh báo, không loại (O-14 i, O-16): bộ A chỉ có vài trang hai cột
    bad_rate_excess_max: float = 0.01  # tỷ lệ trang hỏng được cao hơn baseline tối đa bao nhiêu (0,01 = 1 điểm %)
    trunc_rate_max: float = 0.005
    hour_budget: float | None = None  # trần giờ chạy server cho cả kho; None = chưa đặt, cổng tốc độ "chưa kiểm"
    hour_soft: float | None = 30.0  # mức mềm (N-02, O-11): vượt chỉ cảnh báo, không loại


def page_bad_reason(rec: dict | None, sc: dict | None) -> str | None:
    """Lý do trang bộ A bị tính là bỏ sót / bịa, hoặc None."""
    if rec is None or rec.get("error"):
        return "no_output"
    if sc["hyp_chars"] == 0:
        return "empty"
    if sc["max4"] >= LOOP_MIN and sc["max4"] > 2 * sc["ref_max4"]:
        return "loop"
    if abs(sc["len_ratio"] - 1) > LEN_TOL:
        return "too_long" if sc["len_ratio"] > 1 else "too_short"
    return None


def throughput(run_dir: Path, warmup: int = WARMUP_PAGES) -> dict:
    """
    Tốc độ ổn định của engine từ outputs.jsonl (xem docstring module).

    Returns:
        {"session", "pages", "pages_per_s", "steady"} (pages_per_s None nếu quá ít trang).
    """
    by_session: dict[str, list[float]] = defaultdict(list)
    path = run_dir / "outputs.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("error") and rec.get("t_end"):
                    by_session[rec.get("session") or ""].append(rec["t_end"])
    if not by_session:
        return {"session": None, "pages": 0, "pages_per_s": None, "steady": False}
    session, ends = max(by_session.items(), key=lambda kv: len(kv[1]))
    ends = sorted(ends)
    if len(ends) - warmup >= 2:
        ends = ends[warmup:]
    pps = (len(ends) - 1) / (ends[-1] - ends[0]) if len(ends) >= 2 and ends[-1] > ends[0] else None
    return {"session": session, "pages": len(ends), "pages_per_s": pps, "steady": len(ends) >= STEADY_MIN}


def run_hours(run_dir: Path, speed: dict, corpus: dict[str, int], num_gpus: int) -> dict:
    """
    Số giờ của một engine: đã chạy thật trong đợt so sánh, và ước tính cho cả kho.

    Args:
        speed:  Kết quả throughput() của engine.
        corpus: Số trang cần OCR theo nguồn ({"stbook": ..., "giao_trinh_ocr": ...}).

    Returns:
        {"bakeoff_pages", "bakeoff_run_hours" (tổng các phiên, từ trang đầu bắt đầu tới trang cuối xong),
         "bakeoff_startup_hours" (chờ server + nạp mô hình, theo run_info.json), "pages_per_s", "steady",
         "corpus_pages", "corpus_hours", "corpus_gpu_hours", "corpus_hours_by_source"}.
    """
    spans: dict[str, list[float]] = {}
    pages = 0
    path = run_dir / "outputs.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("t_start") is None or rec.get("t_end") is None:
                    continue
                pages += 1
                lo_hi = spans.setdefault(rec.get("session") or "", [rec["t_start"], rec["t_end"]])
                lo_hi[0], lo_hi[1] = min(lo_hi[0], rec["t_start"]), max(lo_hi[1], rec["t_end"])
    startup = 0.0
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        for sess in json.loads(info_path.read_text(encoding="utf-8")).get("sessions", []):
            startup += sess.get("server_wait_s") or 0.0
            if sess["session"] in spans and sess.get("started"):
                startup += max(0.0, spans[sess["session"]][0] - sess["started"])  # nạp mô hình trong tiến trình
    pps = speed.get("pages_per_s")
    total = sum(corpus.values())

    def hours(n: int) -> float | None:
        """Số giờ server cho n trang, mỗi GPU một bản engine."""
        return n / (pps * num_gpus) / 3600 if pps else None

    return {"bakeoff_pages": pages, "bakeoff_run_hours": sum(hi - lo for lo, hi in spans.values()) / 3600,
            "bakeoff_startup_hours": startup / 3600, "pages_per_s": pps, "steady": speed.get("steady", False),
            "corpus_pages": total, "num_gpus": num_gpus, "corpus_hours": hours(total),
            "corpus_gpu_hours": hours(total) * num_gpus if pps else None,
            "corpus_hours_by_source": {k: hours(v) for k, v in corpus.items()}}


def _mean(xs: list[float]) -> float | None:
    """Trung bình, None nếu rỗng."""
    return sum(xs) / len(xs) if xs else None


def score_set_a(pages: list[dict], bakeoff_dir: Path, outputs: dict[str, dict[str, dict]]) -> tuple[dict, list[dict]]:
    """
    Số đo bộ A của từng engine.

    Returns:
        ({engine: số đo gộp + mảng theo trang "_per_page" để bootstrap}, các dòng per_page).
    """
    refs = {p["page_id"]: (bakeoff_dir / p["gt"]).read_text(encoding="utf-8") for p in pages}
    agg, rows = {}, []
    for name, outs in outputs.items():
        per = []
        for p in pages:
            rec = outs.get(p["page_id"])
            sc = page_scores((rec or {}).get("text") or "", refs[p["page_id"]])
            bad = page_bad_reason(rec, sc)
            landscape = is_landscape(p.get("size"))
            row = {"engine": name, "page_id": p["page_id"], "set": "A", "flags": p.get("flags", {}), "sc": sc,
                   "bad": bad, "truncated": bool(rec and rec.get("finish_reason") == "length"), "landscape": landscape}
            per.append(row)
            rows.append({
                "engine": name, "page_id": p["page_id"], "set": "A",
                "flags": ",".join(k for k, v in p.get("flags", {}).items() if v),
                "ref_chars": sc["ref_chars"], "hyp_chars": sc["hyp_chars"],
                "cer": round(sc["char_dist"] / max(sc["ref_chars"], 1), 4),  # CER thô, không chặn trần
                "cer_capped": round(capped_dist(sc) / max(sc["ref_chars"], 1), 4), "landscape": landscape,
                "cer_nodiac": round(sc["char_dist_nodiac"] / max(sc["ref_chars"], 1), 4),
                "wer": round(sc["word_dist"] / max(sc["ref_words"], 1), 4), "bow_f1": round(sc["bow_f1"], 4),
                "len_ratio": round(sc["len_ratio"], 3), "max4": sc["max4"],
                "special_ref": sum(sc["special_ref"].values()), "special_hit": sum(sc["special_hit"].values()),
                "bad": bad or "", "finish_reason": (rec or {}).get("finish_reason") or "",
                "error": (rec or {}).get("error") or ("missing" if rec is None else ""),
            })
        agg[name] = aggregate_a(per)
    return agg, rows


def capped_dist(sc: dict, key: str = "char_dist") -> int:
    """Lỗi ký tự của một trang bị chặn trần bằng số ký tự đáp án (O-14): CER trang tối đa 100%."""
    return min(sc[key], sc["ref_chars"])


def aggregate_a(per: list[dict]) -> dict:
    """
    Gộp số đo bộ A của một engine từ các dòng theo trang của score_set_a.

    "cer" và "cer_nodiac" là CER gộp có chặn trần mỗi trang; "cer_raw" là CER thô; "cer_rank" là CER chặn trần trên trang
    dọc (thiếu trang dọc thì trên mọi trang), số dùng cho cổng cer_vs_baseline, decide và bootstrap ("_per_page").
    """

    def cer(sel: list[dict], key: str = "char_dist", cap: bool = True) -> float | None:
        """CER gộp trên một nhóm trang; cap=False là CER thô."""
        if not sel:
            return None
        num = [capped_dist(r["sc"], key) if cap else r["sc"][key] for r in sel]
        return micro_rate(num, [r["sc"]["ref_chars"] for r in sel])

    def gap(sel: list[dict]) -> float | None:
        """F1 túi từ trung bình trừ (1 - WER) trung bình: dương lớn = sai thứ tự đọc."""
        if not sel:
            return None
        wer = [min(1.0, r["sc"]["word_dist"] / max(r["sc"]["ref_words"], 1)) for r in sel]
        return _mean([r["sc"]["bow_f1"] for r in sel]) - (1 - _mean(wer))

    strata = {k: [r for r in per if r["flags"].get(k)] for k in ("two_column", "table", "special")}
    strata["plain"] = [r for r in per if not any(r["flags"].values())]
    special_ref: dict[str, int] = defaultdict(int)
    special_hit: dict[str, int] = defaultdict(int)
    strict_ref: dict[str, int] = defaultdict(int)
    strict_hit: dict[str, int] = defaultdict(int)
    for r in per:
        for c, n in r["sc"]["special_ref"].items():
            special_ref[c] += n
            special_hit[c] += r["sc"]["special_hit"].get(c, 0)
        for c, n in r["sc"]["special_ref_strict"].items():
            strict_ref[c] += n
            strict_hit[c] += r["sc"]["special_hit_strict"].get(c, 0)
    portrait = [r for r in per if not r["landscape"]]
    landscape = [r for r in per if r["landscape"]]
    rank = portrait or per
    n = len(per)
    bad_counts: dict[str, int] = defaultdict(int)
    for r in per:
        if r["bad"]:
            bad_counts[r["bad"]] += 1
    tot_ref, tot_strict = sum(special_ref.values()), sum(strict_ref.values())
    return {
        "pages": n, "pages_with_output": sum(r["bad"] != "no_output" for r in per),
        "cer": cer(per), "cer_nodiac": cer(per, "char_dist_nodiac"), "cer_raw": cer(per, cap=False),
        "cer_rank": cer(rank), "rank_pages": len(rank), "rank_scope": "portrait" if portrait else "all",
        "cer_portrait": cer(portrait), "cer_landscape": cer(landscape), "cer_raw_portrait": cer(portrait, cap=False),
        "cer_raw_landscape": cer(landscape, cap=False),
        "pages_portrait": len(portrait), "pages_landscape": len(landscape),
        "pages_two_column_portrait": sum(bool(r["flags"].get("two_column")) for r in portrait),
        "pages_cer_raw_over_100": sum(r["sc"]["char_dist"] > r["sc"]["ref_chars"] for r in per),
        "pages_loop": sum(r["bad"] == "loop" for r in per), "pages_truncated": sum(r["truncated"] for r in per),
        "diacritic_error": (cer(per) or 0) - (cer(per, "char_dist_nodiac") or 0),
        "wer": micro_rate([r["sc"]["word_dist"] for r in per], [r["sc"]["ref_words"] for r in per]),
        "bow_f1": _mean([r["sc"]["bow_f1"] for r in per]),
        "order_gap_all": gap(per), "order_gap_two_column": gap(strata["two_column"]),
        "cer_by_stratum": {k: cer(v) for k, v in strata.items()},
        "pages_by_stratum": {k: len(v) for k, v in strata.items()},
        "special_recall": sum(special_hit.values()) / tot_ref if tot_ref else None,
        "special_by_class": {c: [special_hit[c], special_ref[c]] for c in SPECIAL_CLASSES if special_ref[c]},
        "special_recall_strict": sum(strict_hit.values()) / tot_strict if tot_strict else None,
        "special_by_char_strict": {c: [strict_hit[c], strict_ref[c]] for c in SPECIAL_CHARS if strict_ref[c]},
        "bad_rate": sum(bad_counts.get(k, 0) for k in HARD_BAD) / n if n else None,
        "bad_rate_all": sum(bad_counts.values()) / n if n else None,
        "short_rate": bad_counts.get("too_short", 0) / n if n else None,
        "long_rate": bad_counts.get("too_long", 0) / n if n else None,
        "bad_by_reason": dict(bad_counts),
        "bad_rate_by_reason": {k: v / n for k, v in bad_counts.items()},
        "trunc_rate": sum(r["truncated"] for r in per) / n if n else None,
        "_per_page": {"char_dist": [capped_dist(r["sc"]) for r in rank], "ref_chars": [r["sc"]["ref_chars"] for r in rank]},
    }


def worst_pages(rows: list[dict], pages: list[dict], top: int = WORST_PAGES) -> dict[str, list[dict]]:
    """
    Các trang bộ A có CER cao nhất của từng engine (O-11): để xem lỗi do engine hay do đáp án.

    Args:
        rows:  Dòng per_page của score_set_a.
        pages: Bản ghi trang (lấy nguồn file / số trang).

    Returns:
        {engine: [{"page_id", "source", "flags", "cer", "cer_nodiac", "len_ratio", "bad"}]} giảm dần theo CER.
    """
    info = {p["page_id"]: p for p in pages}
    out: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: -r["cer"]):
        if len(out[r["engine"]]) < top:
            p = info.get(r["page_id"], {})
            src = p.get("source", {})
            out[r["engine"]].append({"page_id": r["page_id"], "source": f"{src.get('file') or src.get('title')} tr.{src.get('page')}",
                                     "flags": p.get("flags", {}), "cer": r["cer"], "cer_nodiac": r["cer_nodiac"],
                                     "len_ratio": r["len_ratio"], "bad": r["bad"]})
    return dict(out)


def raw_cer_examples(rows: list[dict], top: int = EXAMPLES_PER_ENGINE) -> list[dict]:
    """
    Các trang để đặt cạnh nhau trong report.html (O-14): `top` trang CER thô cao nhất của mỗi engine, bỏ trùng giữa engine.

    Args:
        rows: Dòng per_page của score_set_a (cột "cer" là CER thô).

    Returns:
        [{"page_id", "engines" (engine có trang này trong top), "cer_raw" ({engine: CER thô của mọi engine})}], theo thứ
        tự gặp đầu tiên (engine theo thứ tự trong rows, trang giảm dần theo CER thô).
    """
    by_engine: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_engine[r["engine"]].append(r)
    cer_of = {(r["engine"], r["page_id"]): r["cer"] for r in rows}
    found: dict[str, list[str]] = {}
    for name, rs in by_engine.items():
        for r in sorted(rs, key=lambda r: -r["cer"])[:top]:
            found.setdefault(r["page_id"], []).append(name)
    return [{"page_id": pid, "engines": names,
             "cer_raw": {n: cer_of[(n, pid)] for n in by_engine if (n, pid) in cer_of}} for pid, names in found.items()]


def score_set_b(pages: list[dict], outputs: dict[str, dict[str, dict]], vocab: frozenset[str] | None,
                ppl: dict[str, dict[str, float | None]]) -> tuple[dict, list[dict]]:
    """
    Số đo bộ B (không đáp án): âm tiết lạ, perplexity, độ dài so với trung vị các engine, rỗng / vòng lặp / cắt cụt,
    mật độ "?" (VietOCR thay ký tự không có trong bộ chữ bằng "?").

    Returns:
        ({engine: số đo gộp}, các dòng per_page).
    """
    texts = {name: {p["page_id"]: normalize((outs.get(p["page_id"]) or {}).get("text") or "") for p in pages}
             for name, outs in outputs.items()}
    # trung vị độ dài theo trang, chỉ tính các engine có đầu ra không lỗi cho trang đó
    med = {p["page_id"]: statistics.median([len(texts[n][p["page_id"]]) for n, outs in outputs.items()
                                            if p["page_id"] in outs] or [0]) for p in pages}
    agg, rows = {}, []
    for name, outs in outputs.items():
        unk = tot = q = chars = empty = loops = trunc = missing = off_len = 0
        ppls = []
        for p in pages:
            pid = p["page_id"]
            rec, text = outs.get(pid), texts[name][pid]
            u, t = unknown_rate(text, vocab) if vocab is not None else (0, 0)
            unk, tot = unk + u, tot + t
            q, chars = q + text.count("?"), chars + len(text)
            m4 = max_ngram_repeat(text)
            ratio = len(text) / med[pid] if med[pid] else None
            missing += rec is None or bool(rec.get("error"))
            empty += bool(rec and not rec.get("error") and not text)
            loops += m4 >= LOOP_MIN
            trunc += bool(rec and rec.get("finish_reason") == "length")
            off_len += ratio is not None and abs(ratio - 1) > LEN_TOL
            pv = ppl.get(name, {}).get(pid)
            if pv is not None:
                ppls.append(pv)
            rows.append({"engine": name, "page_id": pid, "set": "B", "hyp_chars": len(text),
                         "unknown_rate": round(u / t, 4) if t else "", "ppl": round(pv, 2) if pv else "",
                         "len_vs_median": round(ratio, 3) if ratio is not None else "", "max4": m4,
                         "finish_reason": (rec or {}).get("finish_reason") or "",
                         "error": (rec or {}).get("error") or ("missing" if rec is None else "")})
        n = len(pages)
        agg[name] = {"pages": n, "missing": missing, "empty": empty, "loops": loops, "truncated": trunc,
                     "unknown_syllable_rate": unk / tot if tot else None,
                     "ppl_median": statistics.median(ppls) if ppls else None, "ppl_pages": len(ppls),
                     "len_off_median_rate": off_len / n if n else None,
                     "question_marks_per_1k": 1000 * q / chars if chars else None,
                     "special_chars": sum(sum(text.count(c) for c in SPECIAL_CHARS) for text in texts[name].values())}
    return agg, rows


def apply_gates(a: dict, hours: dict, gates: Gates, baseline: str) -> tuple[dict, dict]:
    """
    Cổng loại cho từng ứng viên (mọi engine trừ baseline) và kết luận (xem F-11).

    Returns:
        ({engine: {"checks": {cổng: {"value", "limit", "status"}}, "corpus_hours", "passed_quality", "passed_speed"}},
         kết luận {"status", "winner", "tied_with", "reason", "comparisons"}).
    """
    base = a.get(baseline)
    out = {}
    for name, m in a.items():
        h = hours.get(name, {})
        corpus_hours = h.get("corpus_hours")

        def check(value, limit, ok, warn: bool = False) -> dict:
            """Một cổng: None = chưa kiểm được; ok mà warn = qua nhưng đáng chú ý (không loại)."""
            status = "na" if ok is None else ("fail" if not ok else ("warn" if warn else "pass"))
            return {"value": value, "limit": limit, "status": status}

        base_bad = base["bad_rate"] if base and base["bad_rate"] is not None else 0.0
        bad_limit = base_bad + gates.bad_rate_excess_max
        base_short = base["short_rate"] if base and base["short_rate"] is not None else 0.0
        short_limit = base_short + gates.bad_rate_excess_max
        soft = gates.hour_soft
        hours_ok = (None if corpus_hours is None or (gates.hour_budget is None and soft is None)
                    else gates.hour_budget is None or corpus_hours <= gates.hour_budget)
        hours_warn = corpus_hours is not None and soft is not None and corpus_hours > soft

        two = m["order_gap_two_column"]
        checks = {
            "cer_vs_baseline": check(m["cer_rank"], base["cer_rank"] if base else None,
                                     None if not base or name == baseline else m["cer_rank"] <= base["cer_rank"]),
            "special_recall": check(m["special_recall"], gates.special_recall_min,
                                    None if m["special_recall"] is None else m["special_recall"] >= gates.special_recall_min),
            "order_gap_two_column": check(two, gates.two_col_gap_max, None if two is None else True,
                                          warn=two is not None and two > gates.two_col_gap_max),
            "bad_rate": check(m["bad_rate"], bad_limit, m["bad_rate"] <= bad_limit),
            "too_short_rate": check(m["short_rate"], short_limit, m["short_rate"] <= short_limit),
            "too_long_rate": check(m["long_rate"], None, True,
                                   warn=base is not None and m["long_rate"] > base["long_rate"] + gates.bad_rate_excess_max),
            "trunc_rate": check(m["trunc_rate"], gates.trunc_rate_max, m["trunc_rate"] <= gates.trunc_rate_max),
            "corpus_hours": check(corpus_hours, gates.hour_budget if gates.hour_budget is not None else soft,
                                  hours_ok, hours_warn),
        }
        quality = all(c["status"] != "fail" for k, c in checks.items() if k != "corpus_hours")
        out[name] = {"checks": checks, "corpus_hours": corpus_hours, "pages_per_s": h.get("pages_per_s"),
                     "steady": h.get("steady", False),
                     "passed_quality": quality, "passed_speed": checks["corpus_hours"]["status"] != "fail"}
    return out, decide(a, out, hours, baseline, gates)


def decide(a: dict, gated: dict, hours: dict, baseline: str, gates: Gates) -> dict:
    """
    Chọn engine: trong các ứng viên qua mọi cổng, CER xếp hạng (cer_rank: chặn trần mỗi trang, trang dọc) thấp nhất; ứng
    viên khác chỉ thua khi khoảng tin cậy bootstrap 95% của hiệu CER (khác - tốt nhất) nằm hẳn trên 0; nhóm hoà chọn engine nhanh nhất. Không ai qua cổng tốc độ mà có
    người qua cổng chất lượng: phương án C.
    """
    cands = [n for n in a if n != baseline]
    comparisons = {}
    if baseline in a:
        b = a[baseline]["_per_page"]
        for n in cands:
            d, lo, hi = paired_bootstrap(a[n]["_per_page"]["char_dist"], b["char_dist"], b["ref_chars"])
            comparisons[f"{n} - {baseline}"] = {"cer_diff": d, "ci95": [lo, hi]}
    quality = [n for n in cands if gated[n]["passed_quality"]]
    passed = [n for n in quality if gated[n]["passed_speed"]]
    notes = []
    if gates.hour_budget is None:
        notes.append("chưa đặt giới hạn số giờ chạy cho cả kho (--hour-budget): cổng tốc độ chưa kiểm, kết luận là tạm")
    slow = [f"{n} ({gated[n]['corpus_hours']:.1f} giờ)" for n in passed if gated[n]["checks"]["corpus_hours"]["status"] == "warn"]
    if slow:
        notes.append(f"vượt mức mềm {gates.hour_soft:g} giờ cho cả kho (chỉ cảnh báo, không loại): {', '.join(slow)}")
    if not all(hours.get(n, {}).get("steady") for n in passed):
        notes.append(f"có engine chưa đủ {STEADY_MIN} trang ổn định để đo tốc độ")
    if not passed:
        status = "option_c" if quality else "no_candidate"
        reason = ("có ứng viên qua cổng chất lượng nhưng vượt giới hạn số giờ chạy: chỉ chạy VLM cho trang bị nghi"
                  if quality else "không ứng viên nào qua cổng chất lượng: giữ baseline, xem lại ứng viên / ngưỡng")
        return {"status": status, "winner": None, "quality_passed": quality, "tied_with": [], "reason": reason,
                "notes": notes, "comparisons": comparisons}
    passed.sort(key=lambda n: a[n]["cer_rank"])
    best = passed[0]
    tied = []
    for n in passed[1:]:
        d, lo, hi = paired_bootstrap(a[n]["_per_page"]["char_dist"], a[best]["_per_page"]["char_dist"],
                                     a[best]["_per_page"]["ref_chars"])
        comparisons[f"{n} - {best}"] = {"cer_diff": d, "ci95": [lo, hi]}
        if lo <= 0:
            tied.append(n)
    group = [best, *tied]
    winner = max(group, key=lambda n: gated[n]["pages_per_s"] or 0)
    reason = (f"CER thấp nhất trong các ứng viên qua cổng ({best})" if winner == best and not tied
              else f"hoà CER (khoảng tin cậy chứa 0) giữa {', '.join(group)}; chọn engine nhanh nhất")
    return {"status": "winner", "winner": winner, "quality_passed": quality, "tied_with": tied, "reason": reason,
            "notes": notes, "comparisons": comparisons}


def _round(v: float | None) -> float | str:
    """Làm tròn 4 chữ số cho CSV; None thành ô trống."""
    return "" if v is None else round(v, 4)


def _fmt(v, pct: bool = False) -> str:
    """Định dạng số cho bảng HTML."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{100 * v:.2f}%" if pct else f"{v:.3g}"
    return html.escape(str(v))


def render_html(summary: dict, pages: list[dict], outputs: dict[str, dict[str, dict]], bakeoff_dir: Path,
                max_a_examples: int = 10) -> str:
    """Báo cáo HTML một file: bảng số đo, cổng, kết luận, bộ B và vài trang hai cột bộ A cạnh nhau."""
    engines = list(dict.fromkeys([*summary["set_a"], *summary["set_b"], *summary["hours"]]))
    e = html.escape
    parts = ["<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>So sánh OCR</title><style>",
             _CSS, "</style></head><body>",
             "<h1>So sánh engine OCR (F-11)</h1>"]
    d = summary["decision"]
    parts.append(f"<h2>Kết luận: {e(d['status'])}</h2><p>Engine chọn: <b>{e(str(d['winner']))}</b>. {e(d['reason'])}</p>")
    parts += [f"<p>⚠ {e(n)}</p>" for n in d["notes"]]
    gates = summary["gates"]
    gate_names = list(next(iter(gates.values()))["checks"]) if gates else []
    parts.append("<h2>Cổng loại</h2><table><tr><th>engine</th>" + "".join(f"<th>{e(g)}</th>" for g in gate_names)
                 + "</tr>")
    for name in engines:
        g = gates.get(name)
        if not g:
            continue
        cells = "".join(f"<td class='{c['status']}'>{_fmt(c['value'], k != 'corpus_hours')}"
                        f" / {_fmt(c['limit'], k != 'corpus_hours')}</td>" for k, c in g["checks"].items())
        parts.append(f"<tr><td>{e(name)}</td>{cells}</tr>")
    parts.append("</table>")
    hrs = summary["hours"]
    parts.append(f"<h2>Số giờ chạy</h2><p>Kho cần OCR {_fmt(summary['corpus_pages'])} trang; số giờ cho cả kho tính khi "
                 f"{summary['num_gpus']} GPU cùng chạy, mỗi GPU một bản engine.</p><table><tr><th>engine</th>"
                 "<th>số trang đã chạy</th><th>giờ đã chạy (so sánh)</th><th>giờ chờ server / nạp mô hình</th>"
                 "<th>trang/s (1 GPU)</th><th>giờ cho cả kho</th><th>giờ GPU cho cả kho</th>"
                 + "".join(f"<th>giờ {e(k)}</th>" for k in next(iter(hrs.values()), {}).get("corpus_hours_by_source", {}))
                 + "</tr>")
    for name in engines:
        h = hrs.get(name)
        if not h:
            continue
        parts.append(f"<tr><td>{e(name)}</td><td>{h['bakeoff_pages']}</td><td>{_fmt(h['bakeoff_run_hours'])}</td>"
                     f"<td>{_fmt(h['bakeoff_startup_hours'])}</td><td>{_fmt(h['pages_per_s'])}"
                     f"{'' if h['steady'] else ' (chưa ổn định)'}</td><td>{_fmt(h['corpus_hours'])}</td>"
                     f"<td>{_fmt(h['corpus_gpu_hours'])}</td>"
                     + "".join(f"<td>{_fmt(v)}</td>" for v in h["corpus_hours_by_source"].values()) + "</tr>")
    parts.append("</table>")
    cols = [("cer_rank", True), ("cer", True), ("cer_raw", True), ("cer_nodiac", True), ("diacritic_error", True), ("wer", True), ("bow_f1", True),
            ("order_gap_all", True), ("order_gap_two_column", True), ("special_recall", True),
            ("special_recall_strict", True), ("bad_rate", True), ("short_rate", True), ("long_rate", True), ("trunc_rate", True)]
    parts.append("<h2>Bộ A (có đáp án)</h2><table><tr><th>engine</th>" + "".join(f"<th>{c}</th>" for c, _ in cols)
                 + "".join(f"<th>CER {k}</th>" for k in ("two_column", "table", "special", "plain")) + "</tr>")
    for name in engines:
        m = summary["set_a"].get(name)
        if not m:
            continue
        parts.append(f"<tr><td>{e(name)}</td>" + "".join(f"<td>{_fmt(m[c], p)}</td>" for c, p in cols)
                     + "".join(f"<td>{_fmt(m['cer_by_stratum'][k], True)}</td>"
                               for k in ("two_column", "table", "special", "plain")) + "</tr>")
    parts.append("</table>")
    first = next(iter(summary["set_a"].values()), None)
    if first:
        parts.append(f"<p>Xếp hạng (cer_rank, cổng cer_vs_baseline, bootstrap) tính trên <b>{first['rank_pages']} trang "
                     f"{'dọc' if first['rank_scope'] == 'portrait' else '(không có trang dọc nên lấy mọi trang)'}</b> "
                     f"(bộ A có {first['pages_portrait']} trang dọc, trong đó {first['pages_two_column_portrait']} trang hai cột; "
                     f"{first['pages_landscape']} trang ngang, phần lớn là slide). Lỗi mỗi trang bị chặn trần 100%; "
                     "cer_raw là CER thô.</p>")
    hcols = [("cer_portrait", "CER trang dọc"), ("cer_landscape", "CER trang ngang"), ("cer_raw_portrait", "CER thô trang dọc"),
             ("cer_raw_landscape", "CER thô trang ngang")]
    parts.append("<h3>CER theo hướng trang và trang bất thường</h3><table><tr><th>engine</th>"
                 + "".join(f"<th>{t}</th>" for _, t in hcols)
                 + "<th>trang CER thô &gt; 100%</th><th>trang vòng lặp</th><th>trang cắt cụt</th></tr>")
    for name in engines:
        m = summary["set_a"].get(name)
        if m:
            parts.append(f"<tr><td>{e(name)}</td>" + "".join(f"<td>{_fmt(m[k], True)}</td>" for k, _ in hcols)
                         + f"<td>{m['pages_cer_raw_over_100']}</td><td>{m['pages_loop']}</td><td>{m['pages_truncated']}</td></tr>")
    parts.append("</table>")
    reasons = ("no_output", "empty", "loop", "too_short", "too_long")
    parts.append("<h3>Trang hỏng theo lý do (số trang)</h3><table><tr><th>engine</th>"
                 + "".join(f"<th>{r}</th>" for r in reasons) + "</tr>")
    for name in engines:
        m = summary["set_a"].get(name)
        if m:
            parts.append(f"<tr><td>{e(name)}</td>" + "".join(f"<td>{m['bad_by_reason'].get(r, 0)}</td>" for r in reasons)
                         + "</tr>")
    parts.append("</table>")
    parts.append("<h3>Recall ký tự đặc biệt theo lớp (đúng / có trong đáp án)</h3><table><tr><th>engine</th>"
                 + "".join(f"<th>{e(c)}</th>" for c in SPECIAL_CLASSES) + "</tr>")
    for name in engines:
        m = summary["set_a"].get(name)
        if m:
            cells = "".join(f"<td>{m['special_by_class'][c][0]} / {m['special_by_class'][c][1]}</td>"
                            if c in m["special_by_class"] else "<td>—</td>" for c in SPECIAL_CLASSES)
            parts.append(f"<tr><td>{e(name)}</td>{cells}</tr>")
    parts.append("</table>")
    if summary.get("worst_pages"):
        parts.append(f"<h3>{WORST_PAGES} trang CER cao nhất mỗi engine</h3>")
        for name in engines:
            rows = summary["worst_pages"].get(name)
            if not rows:
                continue
            parts.append(f"<table><caption>{e(name)}</caption><tr><th>trang</th><th>nguồn</th><th>cờ</th><th>CER</th>"
                         "<th>CER bỏ dấu</th><th>độ dài / đáp án</th><th>hỏng</th></tr>")
            for w in rows:
                flags = ",".join(k for k, v in w["flags"].items() if v)
                parts.append(f"<tr><td>{e(w['page_id'])}</td><td>{e(w['source'])}</td><td>{e(flags)}</td>"
                             f"<td>{_fmt(w['cer'], True)}</td><td>{_fmt(w['cer_nodiac'], True)}</td>"
                             f"<td>{_fmt(w['len_ratio'])}</td><td>{e(w['bad'])}</td></tr>")
            parts.append("</table>")
    if summary["decision"]["comparisons"]:
        parts.append("<h3>Hiệu CER cặp theo trang (bootstrap 95%)</h3><table><tr><th>cặp</th><th>hiệu</th><th>KTC</th></tr>")
        for k, v in summary["decision"]["comparisons"].items():
            parts.append(f"<tr><td>{e(k)}</td><td>{_fmt(v['cer_diff'], True)}</td>"
                         f"<td>[{_fmt(v['ci95'][0], True)}, {_fmt(v['ci95'][1], True)}]</td></tr>")
        parts.append("</table>")
    bcols = ["missing", "empty", "loops", "truncated", "unknown_syllable_rate", "ppl_median", "len_off_median_rate",
             "question_marks_per_1k", "special_chars"]
    parts.append("<h2>Bộ B (ảnh quét thật, không đáp án)</h2><table><tr><th>engine</th>"
                 + "".join(f"<th>{c}</th>" for c in bcols) + "</tr>")
    for name in engines:
        m = summary["set_b"].get(name)
        if m:
            parts.append(f"<tr><td>{e(name)}</td>" + "".join(
                f"<td>{_fmt(m[c], c in ('unknown_syllable_rate', 'len_off_median_rate'))}</td>" for c in bcols) + "</tr>")
    parts.append("</table>")

    def side_by_side(p: dict, ref: str | None, note: str = "") -> str:
        """Một trang: ảnh, đáp án (nếu có), đầu ra từng engine; note nối vào tiêu đề."""
        n_cols = 1 + (ref is not None) + len(engines)
        cells = [f"<div><img loading='lazy' src='../{e(p['image'])}'></div>"]
        if ref is not None:
            cells.append(f"<div><b>đáp án</b><pre>{e(ref)}</pre></div>")
        for name in engines:
            rec = outputs.get(name, {}).get(p["page_id"]) or {}
            cells.append(f"<div><b>{e(name)}</b><pre>{e(rec.get('text') or rec.get('error') or '(không có)')}</pre></div>")
        src = p["source"]
        title = e(f"{p['page_id']} — {src.get('title') or src.get('file')} trang {src.get('page')}" + (f" ({note})" if note else ""))
        return (f"<details><summary>{title}</summary><div class='grid' "
                f"style='grid-template-columns:repeat({n_cols},minmax(220px,1fr))'>{''.join(cells)}</div></details>")

    parts.append("<h3>Từng trang bộ B</h3>")
    parts += [side_by_side(p, None) for p in pages if p["set"] == "B"]
    by_id = {p["page_id"]: p for p in pages}
    raw = [x for x in summary.get("raw_cer_examples", []) if x["page_id"] in by_id]
    if raw:
        parts.append(f"<h3>{EXAMPLES_PER_ENGINE} trang CER thô cao nhất mỗi engine (bộ A)</h3>"
                     "<p>Xem trang vòng lặp / đầu ra quá dài là lỗi engine hay đáp án thiếu chữ trong hình.</p>")
        for x in raw:
            p = by_id[x["page_id"]]
            tag = ", ".join(f"{n} {100 * x['cer_raw'][n]:.0f}%" for n in x["cer_raw"])
            parts.append(side_by_side(p, (bakeoff_dir / p["gt"]).read_text(encoding="utf-8"), f"top của {', '.join(x['engines'])} — CER thô: {tag}"))
    two = [p for p in pages if p["set"] == "A" and p.get("flags", {}).get("two_column")][:max_a_examples]
    if two:
        parts.append("<h3>Vài trang hai cột bộ A</h3>")
        parts += [side_by_side(p, (bakeoff_dir / p["gt"]).read_text(encoding="utf-8")) for p in two]
    parts.append("</body></html>")
    return "\n".join(parts)


def score_bakeoff(bakeoff_dir: Path, engines: list[str] | None = None, baseline: str = "baseline",
                  vocab: frozenset[str] | None = None, num_gpus: int = 4, gates: Gates | None = None,
                  corpus_pages: int | None = None) -> dict:
    """
    Chấm toàn bộ và ghi report/ (xem docstring module).

    Args:
        engines:      Engine cần chấm; None = mọi thư mục trong runs/.
        vocab:        Tập âm tiết cho bộ B (None = bỏ số đo âm tiết lạ).
        corpus_pages: Tổng số trang cần OCR; None = lấy từ selection.json (stbook + giáo trình cần OCR).

    Returns:
        Nội dung summary.json.
    """
    gates = gates or Gates()
    pages = read_pages(bakeoff_dir)
    runs = bakeoff_dir / "runs"
    if not engines:
        engines = sorted(p.name for p in runs.iterdir() if p.is_dir()) if runs.exists() else []
    outputs = {n: latest_outputs(runs / n) for n in engines}
    ppl = {n: latest_outputs_ppl(runs / n / "ppl.jsonl") for n in engines}
    sel_path = bakeoff_dir / "selection.json"
    selection = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else {}
    cp = selection.get("corpus_pages", {})
    corpus = ({"stbook": cp.get("stbook", 0), "giao_trinh_ocr": cp.get("giao_trinh_ocr", 0)}
              if corpus_pages is None else {"total": corpus_pages})
    corpus_pages = sum(corpus.values())
    set_a = [p for p in pages if p["set"] == "A"]
    set_b = [p for p in pages if p["set"] == "B"]
    a, rows_a = score_set_a(set_a, bakeoff_dir, outputs) if set_a else ({}, [])
    b, rows_b = score_set_b(set_b, outputs, vocab, ppl) if set_b else ({}, [])
    speed = {n: throughput(runs / n) for n in engines}
    hours = {n: run_hours(runs / n, speed[n], corpus, num_gpus) for n in engines}
    gated, decision = apply_gates(a, hours, gates, baseline) if a else ({}, {
        "status": "no_set_a", "winner": None, "quality_passed": [], "tied_with": [], "notes": [],
        "reason": "không có trang bộ A", "comparisons": {}})
    # engine không có trang nào (server không lên, thiếu thư viện) bị tính là hỏng mọi trang: không để kết luận
    # "no_candidate / giữ baseline" trông như kết quả thật (gặp ở server 2026-10-12: mọi engine lỗi khi khởi động)
    empty = [n for n in engines if not outputs[n]]
    if empty and (baseline in empty or all(n in empty for n in engines if n != baseline)):
        decision = {"status": "no_outputs", "winner": None, "quality_passed": [], "tied_with": [], "notes": [],
                    "comparisons": {},
                    "reason": f"engine không có trang kết quả nào: {', '.join(empty)}; chưa so sánh được, xem "
                              "logs/<engine>.log và runs/<engine>/server.log"}
    elif empty:
        decision["notes"].append(f"engine không có trang kết quả nào, coi như trượt: {', '.join(empty)}")
    info = {}
    for n in engines:
        p = runs / n / "run_info.json"
        if p.exists():
            ri = json.loads(p.read_text(encoding="utf-8"))
            info[n] = {"gpu": ri.get("gpu"), "license": ri.get("config", {}).get("license"),
                       "note": ri.get("config", {}).get("note")}
    summary = {"engines": engines, "baseline": baseline, "corpus_pages": corpus_pages, "num_gpus": num_gpus,
               "thresholds": {**asdict(gates), "len_tol": LEN_TOL, "loop_min": LOOP_MIN, "warmup_pages": WARMUP_PAGES,
                              "steady_min": STEADY_MIN},
               "selection": {k: selection.get(k) for k in ("set_a", "set_b", "strata", "shortfall", "books_b")},
               "set_a": {n: {k: v for k, v in m.items() if not k.startswith("_")} for n, m in a.items()},
               "worst_pages": worst_pages(rows_a, set_a), "raw_cer_examples": raw_cer_examples(rows_a),
               "set_b": b, "speed": speed, "hours": hours, "gates": gated, "decision": decision, "run_info": info}
    rep = bakeoff_dir / "report"
    rep.mkdir(exist_ok=True)
    (rep / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({k for r in rows_a + rows_b for k in r}, key=lambda k: (k not in ("engine", "page_id", "set"), k))
    with open(rep / "per_page.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_a + rows_b)
    with open(rep / "hours.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        by_src = list(corpus)
        w.writerow(["engine", "bakeoff_pages", "bakeoff_run_hours", "bakeoff_startup_hours", "pages_per_s", "steady",
                    "corpus_pages", "num_gpus", "corpus_hours", "corpus_gpu_hours", *[f"corpus_hours_{k}" for k in by_src]])
        for n, h in hours.items():
            w.writerow([n, h["bakeoff_pages"], _round(h["bakeoff_run_hours"]), _round(h["bakeoff_startup_hours"]),
                        _round(h["pages_per_s"]), h["steady"], h["corpus_pages"], num_gpus, _round(h["corpus_hours"]),
                        _round(h["corpus_gpu_hours"]), *[_round(h["corpus_hours_by_source"][k]) for k in by_src]])
    (rep / "report.html").write_text(render_html(summary, pages, outputs, bakeoff_dir), encoding="utf-8")
    return summary
