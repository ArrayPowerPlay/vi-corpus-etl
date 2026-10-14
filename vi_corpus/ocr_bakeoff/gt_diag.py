"""
Chẩn đoán đáp án bộ A của đợt so sánh engine OCR (O-10, O-11): CER 15-27% là do engine hay do đáp án?

Đáp án bộ A là lớp chữ của PDF giáo trình nên có thể khác thứ mà một engine đọc từ ảnh. Ba nguồn lệch đã biết hoặc nghi:
(1) trang đôi (hai trang in trên một tờ, vd A0266 Guyton có số trang "962" giữa trang): đáp án có hai dãy nội dung và hai
số trang; (2) số trang đứng riêng một dòng giữa trang; (3) tiêu đề / chân trang có trong đáp án mà đa số engine bỏ.
Không có sự thật tuyệt đối, nên dùng đồng thuận giữa các engine: dòng ở đầu / cuối đáp án mà quá nửa số engine có đầu
ra không đọc ra (dưới LINE_PRESENT_MIN số từ của dòng xuất hiện trong đầu ra) là "dòng nghi thừa" của đáp án.

Với mỗi engine tính CER ba kiểu: nguyên đáp án (như score.py), bỏ trang ngang (nghi trang đôi), và đáp án đã bỏ dòng nghi
thừa + số trang nghi lẫn trên các trang không ngang. Ngoài ra liệt kê trang mà mọi engine đều có CER cao (đáp án sai,
hoặc trang khó như bảng). Chỉ dùng CPU, đọc runs/*/outputs.jsonl, ghi report/gt_diag.json và report/gt_diag.csv.
Đây là chẩn đoán để quyết định có sửa cách lập đáp án hay không; không đổi cách chấm của score.py.
"""

import csv
import json
import re
from pathlib import Path

from vi_corpus.ocr_bakeoff.metrics import levenshtein, micro_rate, normalize
from vi_corpus.ocr_bakeoff.pages import read_pages
from vi_corpus.ocr_bakeoff.perplexity import latest_outputs

LANDSCAPE_RATIO = 1.15  # rộng / cao lớn hơn mức này: nghi trang đôi (trang sách đứng có tỷ lệ ~0,7)
EDGE_LINES = 3  # số dòng đầu và cuối đáp án coi là tiêu đề / chân trang ứng viên
EDGE_MAX_CHARS = 80  # dòng dài hơn mức này là nội dung, không phải tiêu đề / chân trang
LINE_PRESENT_MIN = 0.6  # tỷ lệ từ của dòng phải có trong đầu ra thì coi engine đã đọc dòng đó
HIGH_CER = 0.3  # trang mà engine nào cũng CER cao hơn mức này (sau khi bỏ dòng nghi thừa) là "khó hoặc đáp án sai"
_NUM_LINE = re.compile(r"^\d{1,4}$")


def gt_lines(gt: str) -> list[str]:
    """Các dòng không rỗng của đáp án, đã bỏ khoảng trắng hai đầu."""
    return [ln.strip() for ln in gt.splitlines() if ln.strip()]


def number_line_suspects(lines: list[str], is_table: bool) -> tuple[list[int], list[str]]:
    """
    Dòng chỉ gồm số (1-4 chữ số) nghi là số trang lẫn vào đáp án, và lý do.

    Trang thường có tối đa một số trang ở đầu hoặc cuối; từ hai dòng số trở lên (trang đôi) hoặc một dòng số nằm giữa
    trang là đáng ngờ. Trang bảng bỏ qua vì cột số là nội dung thật.

    Returns:
        (chỉ số dòng nghi, danh sách lý do trong {"two_number_lines", "number_mid_page"}).
    """
    if is_table:
        return [], []
    nums = [i for i, ln in enumerate(lines) if _NUM_LINE.match(ln)]
    reasons = []
    if len(nums) >= 2:
        reasons.append("two_number_lines")
    if any(EDGE_LINES <= i < len(lines) - EDGE_LINES for i in nums):
        reasons.append("number_mid_page")
    return (nums if reasons else []), reasons


def line_present(line: str, out_words: set[str]) -> bool:
    """Engine có đọc ra dòng này không: tỷ lệ từ của dòng nằm trong tập từ của đầu ra >= LINE_PRESENT_MIN."""
    ws = normalize(line).lower().split()
    if not ws:
        return True
    return sum(w in out_words for w in ws) / len(ws) >= LINE_PRESENT_MIN


def consensus_missing_edges(lines: list[str], out_words_by_engine: dict[str, set[str]]) -> list[int]:
    """
    Chỉ số các dòng tiêu đề / chân trang ứng viên của đáp án mà quá nửa số engine không đọc ra.

    Cần ít nhất hai engine có đầu ra; ít hơn thì không đủ để nói "đồng thuận" và trả danh sách rỗng.
    """
    if len(out_words_by_engine) < 2:
        return []
    n = len(lines)
    edge = sorted({*range(min(EDGE_LINES, n)), *range(max(0, n - EDGE_LINES), n)})
    missing = []
    for i in edge:
        if len(lines[i]) > EDGE_MAX_CHARS:
            continue
        absent = sum(not line_present(lines[i], words) for words in out_words_by_engine.values())
        if absent * 2 > len(out_words_by_engine):
            missing.append(i)
    return missing


def diagnose_gt(bakeoff_dir: Path, engines: list[str] | None = None, high_cer: float = HIGH_CER) -> dict:
    """
    Chẩn đoán đáp án bộ A và ghi report/gt_diag.json, report/gt_diag.csv (mỗi trang một dòng).

    Args:
        engines:  Engine dùng để đối chiếu; None = mọi thư mục trong runs/ có đầu ra.
        high_cer: Ngưỡng CER của mục "trang mọi engine đều cao".

    Returns:
        {"engines", "pages", "reasons" (số trang theo lý do nghi), "trimmed_lines" (số dòng đáp án bị coi là thừa),
         "cer" ({engine: {"original", "without_landscape", "trimmed"}}), "hard_pages" (trang mọi engine CER cao),
         "ground_truth_share" (CER của engine tốt nhất giảm bao nhiêu phần khi sửa đáp án, trên cùng tập trang không ngang)}.
    """
    pages = [p for p in read_pages(bakeoff_dir) if p["set"] == "A"]
    runs = bakeoff_dir / "runs"
    if not engines:
        engines = sorted(d.name for d in runs.iterdir() if d.is_dir()) if runs.exists() else []
    outputs = {n: latest_outputs(runs / n) for n in engines}
    engines = [n for n in engines if outputs[n]]
    reason_counts: dict[str, int] = {}
    trimmed_total = 0
    rows = []
    num = {n: {"original": [], "without_landscape": [], "trimmed": []} for n in engines}
    den = {"original": [], "without_landscape": [], "trimmed": []}
    for p in pages:
        gt = (bakeoff_dir / p["gt"]).read_text(encoding="utf-8")
        lines = gt_lines(gt)
        size = p.get("size") or [0, 0]
        reasons = []
        if size[1] and size[0] > LANDSCAPE_RATIO * size[1]:
            reasons.append("landscape")
        num_idx, num_reasons = number_line_suspects(lines, bool(p.get("flags", {}).get("table")))
        reasons += num_reasons
        words = {n: set(normalize(outputs[n][p["page_id"]].get("text") or "").lower().split())
                 for n in engines if p["page_id"] in outputs[n] and (outputs[n][p["page_id"]].get("text") or "").strip()}
        edge_idx = consensus_missing_edges(lines, words)
        if edge_idx:
            reasons.append("edge_lines_dropped_by_engines")
        drop = set(num_idx) | set(edge_idx)
        trimmed_gt = normalize("\n".join(ln for i, ln in enumerate(lines) if i not in drop))
        orig_gt = normalize(gt)
        trimmed_total += len(drop)
        for r in reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1
        landscape = "landscape" in reasons
        den["original"].append(len(orig_gt))
        den["without_landscape"].append(0 if landscape else len(orig_gt))
        den["trimmed"].append(0 if landscape else len(trimmed_gt))
        row = {"page_id": p["page_id"], "reasons": ",".join(reasons), "dropped_lines": len(drop),
               "flags": ",".join(k for k, v in p.get("flags", {}).items() if v),
               "dropped_text": " | ".join(lines[i] for i in sorted(drop))[:200]}
        trimmed_cers = []
        for n in engines:
            hyp = normalize((outputs[n].get(p["page_id"]) or {}).get("text") or "")
            d_orig = levenshtein(hyp, orig_gt)
            d_trim = levenshtein(hyp, trimmed_gt)
            num[n]["original"].append(d_orig)
            num[n]["without_landscape"].append(0 if landscape else d_orig)
            num[n]["trimmed"].append(0 if landscape else d_trim)
            row[f"cer_{n}"] = round(d_orig / max(len(orig_gt), 1), 4)
            row[f"cer_trimmed_{n}"] = round(d_trim / max(len(trimmed_gt), 1), 4)
            if p["page_id"] in outputs[n]:
                trimmed_cers.append(d_trim / max(len(trimmed_gt), 1))
        row["min_cer_trimmed"] = round(min(trimmed_cers), 4) if trimmed_cers else ""
        rows.append(row)
    cer = {n: {k: micro_rate(num[n][k], den[k]) for k in den} for n in engines}
    hard = [{"page_id": r["page_id"], "min_cer_trimmed": r["min_cer_trimmed"], "reasons": r["reasons"], "flags": r["flags"]}
            for r in rows if r["min_cer_trimmed"] != "" and r["min_cer_trimmed"] > high_cer]
    hard.sort(key=lambda r: -r["min_cer_trimmed"])
    best = min(engines, key=lambda n: cer[n]["original"], default=None)
    share = None
    if best and cer[best]["without_landscape"]:  # cùng tập trang, chỉ khác đáp án
        share = (cer[best]["without_landscape"] - cer[best]["trimmed"]) / cer[best]["without_landscape"]
    result = {"engines": engines, "pages": len(pages), "reasons": reason_counts, "trimmed_lines": trimmed_total,
              "cer": cer, "hard_pages": hard, "high_cer": high_cer, "best_engine": best,
              "ground_truth_share": share}
    rep = bakeoff_dir / "report"
    rep.mkdir(exist_ok=True)
    (rep / "gt_diag.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("page_id", "reasons"), k))
    with open(rep / "gt_diag.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return result
