"""
Công cụ đo cho pipeline VJOL (J-25 Bước 0, chạy CPU, đọc kết quả của một lần chạy pipeline; không sửa dữ liệu):

- audit_run: kiểm MỌI lần chạy pipeline có VJOL: tỷ lệ từng loại lỗi còn lọt trong các đoạn được giữ (email, số điện thoại, ư/ơ thành "-", tiêu đề trang, nhãn tham khảo ...), các nhóm
  tạp chí (và tạp chí x năm) bị loại bất thường so với trung vị, số đoạn bị loại chỉ vì MỘT luật duy nhất, kèm mẫu để đọc. Dấu hiệu `econst` bị loại 26% hay `ncca` 48% ở vjol_10k_v3
  là loại tín hiệu mà không công cụ nào báo trước đây.
- Bộ đáp án (make_labeling_sample, evaluate_labels): lấy ~400 đoạn để đọc và gán nhãn (LABELS), chia hai nửa THEO BÀI (tune: chỉnh ngưỡng; holdout: chỉ để kiểm), rồi đo độ chính xác / độ bao phủ
  của từng bộ dò theo nhãn. Hạn chế đã biết: nhãn do người đọc gán nên chỉ đo được lỗi mà người đọc thấy.
- J-33: journal_balance (tỷ lệ bị loại theo từng tạp chí và trung bình theo tạp chí, vì một tạp chí có thể chiếm hơn 1/4 mẫu), year_checks (đoạn thiếu năm, năm mặc định 1970, năm nào chỉ có ở bài không có
  metadata) và font_misread (chữ có dấu bị đọc thành chữ có dấu khác: "NGHIấN", "nghiờn", "nhõn", "Khỏnh"; cần tập âm tiết) để quyết định có cần sửa họ lỗi này không (J-33 H).
- Tập chữ đúng (make_clean_set, check_clean_set): đoạn kinh tế / luật / y học có dấu đầy đủ; bộ dò nào gắn cờ lên tập này là lỗi (tiêu chí nghiệm thu "0 cờ").
"""

import csv
import hashlib
import json
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from vi_corpus.common.garbled import _HYPHEN_LOST, DEFAULT_THRESHOLDS, hyphen_lost, latin_tokens, load_vocab, page_signals
from vi_corpus.common.state import write_json_atomic
from vi_corpus.pipeline import quality as Q
from vi_corpus.pipeline.config import PROFILES
from vi_corpus.pipeline.io import read_rows
from vi_corpus.vjol.clean import _BOILER, BOILER_MAX_WORDS, EMAIL, find_phones, heading_key, is_ref_heading

OUT_SUBDIR = "audit"
SUSPECT_RULES = (("odd_word", 0.08), ("unaccented", 0.01), ("reversed_gain", 0.15), ("pua_chars", 1), ("junk_script_chars", 1), ("hyphen_lost", 1))
KEPT_BANDS = ("A", "B", "C")
LABELS = ("clean", "lost_diacritics", "hyphen_font", "font_misread", "old_ocr", "reference", "running_header", "other")
_JOURNAL_HEADER = re.compile(r"Tạp chí[^\n]{3,70}?\bsố\s*\d+", re.IGNORECASE)
CLEAN_KEYWORDS = ("kinh tế", "doanh nghiệp", "ngân hàng", "tài chính", "pháp luật", "điều luật", "luật", "bệnh nhân", "điều trị", "lâm sàng", "thuế")
HARD_CODES = Q.HARD_REASONS | {"garbled_font"}
# Họ lỗi đọc font (J-25 (3), J-33 H): chữ có dấu bị thay bằng chữ có dấu khác. Khoá = ký tự sai, giá trị = ký tự đúng ("NGHIấN" -> "NGHIÊN", "nghiờn" -> "nghiên", "nhõn" -> "nhân", "Khỏnh" -> "Khánh").
MISREAD_MAP = {"ấ": "ê", "ờ": "ê", "õ": "â", "ỏ": "á"}
MISREAD_MIN_HITS = 2  # đoạn có ít nhất ngần ấy từ sửa được bằng MISREAD_MAP mới tính là đoạn lỗi (một tên riêng lẻ tẻ không tính)
MISREAD_REPORT_RATE = 0.005  # J-33 H: tỷ lệ đoạn giữ lại vượt mức này thì nên thêm tín hiệu / bản đồ sửa


def metrics_of(row: dict) -> dict:
    """Số đo chất lượng của một bản ghi (cột quality_metrics, {} nếu thiếu)."""
    try:
        return json.loads(row.get("quality_metrics") or "{}")
    except json.JSONDecodeError:
        return {}


def meta_of(row: dict) -> dict:
    """Cột meta của một bản ghi thành dict ({} nếu thiếu / hỏng)."""
    try:
        meta = json.loads(row.get("meta") or "{}")
    except json.JSONDecodeError:
        return {}
    return meta if isinstance(meta, dict) else {}


def suspicious(row: dict) -> list[str]:
    """Tên các số đo của đoạn vượt ngưỡng nghi ngờ (SUSPECT_RULES), thêm 'pii' nếu còn email / số điện thoại thật trong text."""
    m = metrics_of(row)
    why = [k for k, t in SUSPECT_RULES if m.get(k, 0) >= t]
    if EMAIL.search(row["text"]) or find_phones(row["text"]):
        why.append("pii")
    return why


def misread_hits(text: str, vocab: frozenset[str]) -> int:
    """
    Số từ trong `text` KHÔNG thuộc tập âm tiết `vocab` nhưng thành âm tiết thật khi đổi MỘT ký tự theo MISREAD_MAP (họ lỗi đọc font ê -> ấ / ờ, â -> õ, á -> ỏ; J-33 H).
    """
    hits = 0
    for tok in latin_tokens(text):
        low = tok.lower()
        if low in vocab:
            continue
        hits += any(c in MISREAD_MAP and low[:i] + MISREAD_MAP[c] + low[i + 1:] in vocab for i, c in enumerate(low))
    return hits


def _lines(text: str) -> list[str]:
    """Các dòng không rỗng của văn bản."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


ERROR_CHECKS: dict[str, Callable[[dict, dict], bool]] = {
    "email": lambda r, m: bool(EMAIL.search(r["text"])),
    "phone": lambda r, m: bool(find_phones(r["text"])),
    "hyphen_lost": lambda r, m: m.get("hyphen_lost", hyphen_lost(r["text"])[2]) >= 1,
    "unaccented": lambda r, m: m.get("unaccented", 0.0) >= 0.01,
    "odd_word_high": lambda r, m: m.get("odd_word", 0.0) >= 0.2,
    "private_use": lambda r, m: m.get("pua_chars", 0) >= 1,
    "junk_script": lambda r, m: m.get("junk_script_chars", 0) >= 1,
    "reference_label": lambda r, m: "reference_list" in (r.get("reason_codes") or []),
    "reference_heading": lambda r, m: any(is_ref_heading(heading_key(ln)) for ln in _lines(r["text"])),
    "journal_header": lambda r, m: bool(_JOURNAL_HEADER.search(r["text"])),
    "received_line": lambda r, m: any(len(ln.split()) <= BOILER_MAX_WORDS and _BOILER.match(ln) for ln in _lines(r["text"])),
    "high_perplexity": lambda r, m: m.get("perplexity", 0.0) >= Q.PPL_HIGH,
}


def _locate_phone(text: str) -> tuple[int, int] | None:
    """Vị trí số điện thoại thật đầu tiên (find_phones) trong văn bản."""
    found = find_phones(text)
    i = text.find(found[0]) if found else -1
    return (i, i + len(found[0])) if i >= 0 else None


# Hàm định vị chỗ lỗi cho mẫu của audit (J-28): trả về (đầu, cuối) hoặc None. Lỗi không có ở đây thì mẫu lấy đầu đoạn.
HIT_LOCATORS: dict[str, Callable[[str], tuple[int, int] | None]] = {
    "journal_header": lambda t: (m.start(), m.end()) if (m := _JOURNAL_HEADER.search(t)) else None,
    "email": lambda t: (m.start(), m.end()) if (m := EMAIL.search(t)) else None,
    "phone": _locate_phone,
    "hyphen_lost": lambda t: (m.start(), m.end()) if (m := _HYPHEN_LOST.search(unicodedata.normalize("NFC", t))) else None,
    "lost_letters": lambda t: (m.start(), m.end()) if (m := _HYPHEN_LOST.search(unicodedata.normalize("NFC", t))) else None,
}


def excerpt_of(text: str, name: str = "", width: int = 100) -> str:
    """Trích đoạn để đọc: quanh chỗ lỗi `name` (HIT_LOCATORS) ±`width` ký tự, không có thì 240 ký tự đầu."""
    loc = HIT_LOCATORS.get(name, lambda t: None)(text)
    if loc is None:
        return text[:240]
    return "…" + text[max(0, loc[0] - width):loc[1] + width].replace("\n", " | ") + "…"


def error_flags(row: dict) -> list[str]:
    """Các loại lỗi (ERROR_CHECKS) mà một đoạn còn dính."""
    m = metrics_of(row)
    return [name for name, check in ERROR_CHECKS.items() if check(row, m)]


def _hard_reasons(row: dict) -> list[str]:
    """Các mã lỗi cứng của bản ghi."""
    return [c for c in row.get("reason_codes") or [] if c in HARD_CODES]


def journal_balance(rows: list[dict], journals: dict[str, dict], min_group: int = 20) -> dict:
    """
    Độ lệch của mẫu theo tạp chí và tỷ lệ bị loại tính theo từng tạp chí (J-33 J): một tạp chí chiếm hơn 1/4 mẫu (vjol_10k_v5: `tcdantochocvatongiaohoc` 26,6%) làm tỷ lệ bị loại chung
    thiên về tạp chí đó, nên báo thêm trung bình theo tạp chí (macro), không chỉ trung bình theo đoạn (micro).

    Args:
        rows:      Bản ghi vjol đã qua stage quality.
        journals:  Kết quả group_stats theo tạp chí ({slug: {"chunks", "dropped_rate", ...}}).
        min_group: Tạp chí ít đoạn hơn ngần này không đưa vào trung bình theo tạp chí (số nhỏ không đáng tin).

    Returns:
        {"journals", "top": [[slug, tỷ lệ đoạn]] (5 tạp chí lớn nhất), "max_share", "dropped_rate_micro", "dropped_rate_macro" (trung bình theo tạp chí đủ min_group đoạn), "macro_journals"}.
    """
    total = len(rows)
    ranked = sorted(journals.items(), key=lambda kv: -kv[1]["chunks"])
    big = [v["dropped_rate"] for _, v in ranked if v["chunks"] >= min_group]
    dropped = sum(r["quality_band"] not in KEPT_BANDS for r in rows)
    return {"journals": len(journals), "top": [[k, round(v["chunks"] / total, 4)] for k, v in ranked[:5]] if total else [],
            "max_share": round(ranked[0][1]["chunks"] / total, 4) if ranked and total else 0.0,
            "dropped_rate_micro": round(dropped / total, 4) if total else 0.0,
            "dropped_rate_macro": round(statistics.mean(big), 4) if big else None, "macro_journals": len(big)}


def year_checks(rows: list[dict], top: int = 5) -> dict:
    """
    Kiểm cột năm của meta (J-33 D): số đoạn thiếu năm, số đoạn mang năm mặc định 1970 (phải bằng 0 sau khi sửa), các năm chiếm nhiều đoạn nhất và năm nào có nhiều bài không tìm thấy metadata.

    Returns:
        {"missing": số đoạn năm None, "default_1970": số đoạn có year hoặc year_dir 1970 (year_dir 1970 là bình thường: năm lấy từ thư mục chỉ mang tính tên thư mục), "year_1970": số đoạn có year 1970,
         "top_years": [[năm, tỷ lệ]], "no_metadata_years": {năm: tỷ lệ đoạn không có metadata} (chỉ năm có >= 20 đoạn và tỷ lệ >= 0,5)}.
    """
    metas = [meta_of(r) for r in rows]
    years = Counter(m.get("year") for m in metas)
    per_year_total: Counter = Counter()
    per_year_nometa: Counter = Counter()
    for m in metas:
        y = m.get("year_dir") if m.get("year_dir") is not None else m.get("year")
        per_year_total[y] += 1
        per_year_nometa[y] += not m.get("metadata_found", True)
    n = len(rows)
    return {"missing": years.get(None, 0), "year_1970": years.get(1970, 0), "default_1970": sum(m.get("year") == 1970 or m.get("year_dir") == 1970 for m in metas),
            "top_years": [[y, round(c / n, 4)] for y, c in years.most_common(top)] if n else [],
            "no_metadata_years": {str(y): round(per_year_nometa[y] / t, 3) for y, t in sorted(per_year_total.items(), key=lambda kv: str(kv[0]))
                                  if t >= 20 and per_year_nometa[y] / t >= 0.5}}


def audit_rows(rows: list[dict], min_group: int = 20, factor: float = 3.0, min_rate: float = 0.15, samples: int = 5, seed: int = 0,
               vocab: frozenset[str] | None = None) -> dict:
    """
    Kiểm một danh sách bản ghi VJOL đã qua stage quality (xem audit_run).

    Args:
        rows:      Bản ghi nguồn vjol có quality_band, reason_codes, quality_metrics, meta.
        min_group: Nhóm tạp chí (x năm) có ít hơn ngần ấy đoạn thì bỏ qua (số nhỏ không đáng tin).
        factor:    Tỷ lệ bị loại phải lớn hơn factor x trung vị các nhóm...
        min_rate:  ... và ít nhất mức này thì nhóm bị báo bất thường.
        samples:   Số mẫu mỗi loại lỗi / luật.
        vocab:     Tập âm tiết thật; có thì đo thêm họ lỗi đọc font "font_misread" (misread_hits, J-33 H).

    Returns:
        {"chunks", "kept", "dropped", "error_rates": {lỗi: tỷ lệ trên đoạn giữ}, "error_counts", "single_rule_drops": {mã: số đoạn chỉ bị mã đó}, "groups": {"journal"|"journal_year": {...}},
         "anomalies": [{"level", "key", "chunks", "dropped_rate", "median", "top_reasons"}], "samples": {tên: [{doc_id, journal, year, excerpt}]},
         "journal_balance": xem journal_balance, "year_checks": xem year_checks}.
    """
    rng = random.Random(seed)
    kept = [r for r in rows if r["quality_band"] in KEPT_BANDS]
    dropped = [r for r in rows if r["quality_band"] not in KEPT_BANDS]
    counts: Counter = Counter()
    found: dict[str, list[dict]] = defaultdict(list)
    for r in kept:
        flags = error_flags(r)
        if vocab is not None and misread_hits(r["text"], vocab) >= MISREAD_MIN_HITS:
            flags.append("font_misread")
        for name in flags:
            counts[name] += 1
            found[name].append(r)
    single: Counter = Counter()
    single_rows: dict[str, list[dict]] = defaultdict(list)
    for r in dropped:
        hard = _hard_reasons(r)
        if len(hard) == 1:
            single[hard[0]] += 1
            single_rows[hard[0]].append(r)

    def group_stats(level: str, keyf: Callable[[dict], str | None]) -> dict:
        """Tỷ lệ bị loại và mã lỗi hàng đầu theo nhóm."""
        acc: dict[str, dict] = defaultdict(lambda: {"chunks": 0, "dropped": 0, "reasons": Counter()})
        for r in rows:
            key = keyf(r)
            if not key:
                continue
            g = acc[key]
            g["chunks"] += 1
            if r["quality_band"] not in KEPT_BANDS:
                g["dropped"] += 1
                g["reasons"].update(_hard_reasons(r))
        return {k: {"chunks": g["chunks"], "dropped_rate": round(g["dropped"] / g["chunks"], 4), "top_reasons": g["reasons"].most_common(3)} for k, g in acc.items()}

    groups = {"journal": group_stats("journal", lambda r: meta_of(r).get("journal_slug")),
              "journal_year": group_stats("journal_year", lambda r: f"{meta_of(r).get('journal_slug')}|{meta_of(r).get('year')}" if meta_of(r).get("journal_slug") else None)}
    anomalies = []
    for level, stats in groups.items():
        big = {k: v for k, v in stats.items() if v["chunks"] >= min_group}
        if len(big) < 3:
            continue
        med = statistics.median(v["dropped_rate"] for v in big.values())
        for k, v in sorted(big.items(), key=lambda kv: -kv[1]["dropped_rate"]):
            if v["dropped_rate"] > factor * med and v["dropped_rate"] >= min_rate:
                anomalies.append({"level": level, "key": k, "chunks": v["chunks"], "dropped_rate": v["dropped_rate"], "median": round(med, 4), "top_reasons": v["top_reasons"]})

    def pick(pool: list[dict], name: str = "") -> list[dict]:
        """Mẫu ngẫu nhiên để đọc; với lỗi có định vị được (HIT_LOCATORS) thì trích quanh chỗ lỗi thay vì đầu đoạn."""
        return [{"doc_id": r["doc_id"], "journal": meta_of(r).get("journal_slug"), "year": meta_of(r).get("year"), "excerpt": excerpt_of(r["text"], name)}
                for r in rng.sample(pool, min(samples, len(pool)))]

    return {"chunks": len(rows), "kept": len(kept), "dropped": len(dropped),
            "error_counts": dict(counts), "error_rates": {k: round(v / len(kept), 4) for k, v in counts.items()} if kept else {},
            "single_rule_drops": dict(single.most_common()), "groups": groups, "anomalies": anomalies,
            "journal_balance": journal_balance(rows, groups["journal"], min_group), "year_checks": year_checks(rows),
            "samples": {**{f"kept:{k}": pick(v, k) for k, v in found.items()}, **{f"dropped_only_by:{k}": pick(v, k) for k, v in single_rows.items()}}}


def audit_run(run_dir: Path, out_dir: Path | None = None, syllables: Path | None = None, **kwargs) -> Path:
    """
    Kiểm một lần chạy pipeline: đọc 04_quality.parquet (nguồn vjol), ghi audit.json, audit_journals.csv và audit.md (đọc nhanh) vào <run_dir>/audit/.

    Args:
        syllables: File tập âm tiết cho phép đo họ lỗi đọc font (J-33 H); mặc định <data-root>/processed/ocr_bakeoff/syllables.json với data-root suy ra từ run_dir
                   (<data-root>/processed/pipeline_runs/<tên>), thiếu file thì bỏ phép đo đó.

    Returns:
        Thư mục kết quả.
    """
    rows = [r for r in read_rows(run_dir / "04_quality.parquet") if r["source_key"] == "vjol"]
    out = out_dir or run_dir / OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    default = run_dir.resolve().parents[2] / "processed" / "ocr_bakeoff" / "syllables.json" if len(run_dir.resolve().parents) > 2 else None
    vocab = load_vocab(syllables or default) if (syllables or default) else None
    res = audit_rows(rows, vocab=vocab, **kwargs)
    write_json_atomic(out / "audit.json", res)
    with open(out / "audit_journals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "key", "chunks", "dropped_rate", "top_reasons"])
        for level, stats in res["groups"].items():
            for k, v in sorted(stats.items(), key=lambda kv: -kv[1]["dropped_rate"]):
                w.writerow([level, k, v["chunks"], v["dropped_rate"], "; ".join(f"{c}:{n}" for c, n in v["top_reasons"])])
    lines = [f"# Kiểm {run_dir.name}", "", f"- {res['chunks']} đoạn vjol: giữ {res['kept']}, loại {res['dropped']}", "", "## Lỗi còn lọt trong đoạn được giữ (tỷ lệ)"]
    lines += [f"- {k}: {res['error_counts'][k]} ({v:.2%})" for k, v in sorted(res["error_rates"].items(), key=lambda kv: -kv[1])]
    jb, yc = res["journal_balance"], res["year_checks"]
    lines += ["", "## Độ lệch theo tạp chí (J-33 J)",
              f"- {jb['journals']} tạp chí; lớn nhất {jb['top']}; bị loại: theo đoạn {jb['dropped_rate_micro']:.2%}, trung bình theo {jb['macro_journals']} tạp chí (>= {kwargs.get('min_group', 20)} đoạn) "
              f"{'-' if jb['dropped_rate_macro'] is None else format(jb['dropped_rate_macro'], '.2%')}"]
    lines += ["", "## Năm (J-33 D)", f"- thiếu năm {yc['missing']} đoạn; year=1970 {yc['year_1970']} đoạn (phải 0); year hoặc year_dir 1970 {yc['default_1970']}; năm nhiều đoạn nhất {yc['top_years']}; "
              f"năm có hơn nửa đoạn không có metadata {yc['no_metadata_years']}"]
    if vocab is None:
        lines += ["", "(Không có tập âm tiết: bỏ phép đo font_misread, J-33 H)"]
    elif res["error_rates"].get("font_misread", 0.0) >= MISREAD_REPORT_RATE:
        lines += ["", f"## CHÚ Ý (J-33 H): font_misread {res['error_rates']['font_misread']:.2%} >= {MISREAD_REPORT_RATE:.1%} đoạn giữ lại: nên thêm tín hiệu hoặc bản đồ sửa cho họ lỗi ê -> ấ / ờ"]
    lines += ["", "## Đoạn bị loại chỉ vì một luật duy nhất"] + [f"- {k}: {v}" for k, v in res["single_rule_drops"].items()]
    lines += ["", "## Nhóm bị loại bất thường"] + [f"- [{a['level']}] {a['key']}: {a['dropped_rate']:.0%} ({a['chunks']} đoạn, trung vị {a['median']:.0%}), {a['top_reasons']}" for a in res["anomalies"]]
    lines += ["", "## Mẫu"]
    for name, items in res["samples"].items():
        lines.append(f"### {name}")
        lines += [f"- `{s['doc_id']}` ({s['journal']}, {s['year']}): {s['excerpt']!r}" for s in items]
    (out / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def compare_audits(old: dict, new: dict, old_name: str = "cũ", new_name: str = "mới") -> str:
    """
    So hai kết quả audit_rows (audit.json) thành bảng văn bản: số đoạn giữ / loại, từng lỗi còn lọt, đoạn bị loại chỉ vì một luật, năm và độ lệch tạp chí (J-33 "cách đo xong").

    Returns:
        Bảng nhiều dòng "chỉ số | cũ | mới | đổi"; chỉ số thiếu ở một bên ghi "-".
    """
    def fmt(v) -> str:
        """Số hiển thị: tỷ lệ nhỏ hơn 1 dạng %, số nguyên giữ nguyên, thiếu là "-"."""
        if v is None:
            return "-"
        return f"{v:.2%}" if isinstance(v, float) and abs(v) < 1 else str(v)

    def delta(a, b) -> str:
        """Hiệu mới - cũ (cùng đơn vị với fmt) hoặc "-"."""
        if a is None or b is None:
            return "-"
        d = b - a
        return f"{d:+.2%}" if isinstance(d, float) and abs(a) < 1 and abs(b) < 1 else f"{d:+}"

    rows: list[tuple[str, object, object]] = [("đoạn", old["chunks"], new["chunks"]), ("giữ", old["kept"], new["kept"]), ("loại", old["dropped"], new["dropped"])]
    rows += [(f"lỗi lọt: {k}", old["error_rates"].get(k), new["error_rates"].get(k)) for k in sorted(set(old["error_rates"]) | set(new["error_rates"]))]
    rows += [(f"chỉ bị loại bởi: {k}", old["single_rule_drops"].get(k), new["single_rule_drops"].get(k)) for k in sorted(set(old["single_rule_drops"]) | set(new["single_rule_drops"]))]
    for key, label in (("missing", "năm: thiếu"), ("year_1970", "năm: year=1970")):
        rows.append((label, old.get("year_checks", {}).get(key), new.get("year_checks", {}).get(key)))
    for key, label in (("max_share", "tạp chí: tỷ lệ lớn nhất"), ("dropped_rate_micro", "tạp chí: loại theo đoạn"), ("dropped_rate_macro", "tạp chí: loại theo tạp chí")):
        rows.append((label, old.get("journal_balance", {}).get(key), new.get("journal_balance", {}).get(key)))
    rows.append(("nhóm tạp chí bất thường", len(old["anomalies"]), len(new["anomalies"])))
    width = max(len(r[0]) for r in rows)
    lines = [f"{'chỉ số':<{width}} | {old_name:>10} | {new_name:>10} | đổi"]
    lines += [f"{name:<{width}} | {fmt(a):>10} | {fmt(b):>10} | {delta(a, b)}" for name, a, b in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------- bộ đáp án


def split_of(article_id: object, seed: int = 0) -> str:
    """Nửa của bộ đáp án theo BÀI ("tune" | "holdout"): băm (seed, article_id), nên mọi đoạn của một bài ở cùng một nửa và cùng seed cho cùng kết quả."""
    return "tune" if int(hashlib.sha1(f"{seed}:{article_id}".encode()).hexdigest(), 16) % 2 == 0 else "holdout"


def make_labeling_sample(rows: list[dict], n: int = 400, seed: int = 0, suspect_share: float = 0.5) -> list[dict]:
    """
    Chọn n đoạn đã giữ (band A/B/C) để đọc và gán nhãn: `suspect_share` là đoạn nghi ngờ (suspicious), phần còn lại ngẫu nhiên; mỗi bài tối đa 3 đoạn để nửa chỉnh / nửa kiểm đa dạng.

    Returns:
        Danh sách {doc_id, article_id, journal, year, split, group, band, reason_codes, suspicious, quality_metrics, text, label, note}; cột label, note để trống.
    """
    rng = random.Random(seed)
    pool = [r for r in rows if r["quality_band"] in KEPT_BANDS]
    rng.shuffle(pool)
    per_article: Counter = Counter()

    def take(cands: list[dict], k: int) -> list[dict]:
        """Lấy tối đa k đoạn, mỗi bài không quá 3."""
        got = []
        for r in cands:
            aid = meta_of(r).get("article_id") or r.get("parent_doc_id")
            if len(got) < k and per_article[aid] < 3:
                per_article[aid] += 1
                got.append(r)
        return got

    sus = take([r for r in pool if suspicious(r)], int(n * suspect_share))
    taken = {r["doc_id"] for r in sus}
    rest = take([r for r in pool if r["doc_id"] not in taken], n - len(sus))
    out = []
    for group, rs in (("nghi_ngờ", sus), ("ngẫu_nhiên", rest)):
        for r in rs:
            meta = meta_of(r)
            out.append({"doc_id": r["doc_id"], "article_id": meta.get("article_id"), "journal": meta.get("journal_slug"), "year": meta.get("year"),
                        "split": split_of(meta.get("article_id") or r.get("parent_doc_id"), seed), "group": group, "band": r["quality_band"],
                        "reason_codes": "+".join(r.get("reason_codes") or []), "suspicious": "+".join(suspicious(r)),
                        "quality_metrics": r.get("quality_metrics") or "{}", "text": r["text"], "label": "", "note": ""})
    return out


def write_labeling_sample(items: list[dict], out_dir: Path) -> None:
    """Ghi labeling_sample.csv (điền cột label bằng một trong LABELS) và labeling_sample.md (đọc nhanh) vào out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["doc_id", "article_id", "journal", "year", "split", "group", "band", "reason_codes", "suspicious", "quality_metrics", "text", "label", "note"]
    with open(out_dir / "labeling_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(items)
    with open(out_dir / "labeling_sample.md", "w", encoding="utf-8") as f:
        f.write(f"Nhãn hợp lệ: {', '.join(LABELS)}\n\n")
        for i, it in enumerate(items, 1):
            f.write(f"## {i}. {it['doc_id']} [{it['split']}/{it['group']}] {it['journal']} {it['year']} band {it['band']} {it['reason_codes']}\n\n{it['text'][:1500]}\n\n")


DETECTORS: dict[str, Callable[[dict], bool]] = {
    "lost_diacritics": lambda m: m.get("unaccented", 0.0) >= DEFAULT_THRESHOLDS.nodiac,
    "hyphen_font": lambda m: m.get("hyphen_lost", 0) >= Q.HYPHEN_HARD_COUNT,
    "font_misread": lambda m: m.get("odd_word", 0.0) >= Q.ODD_SOFT,
    "old_ocr": lambda m: m.get("odd_word", 0.0) >= Q.ODD_SOFT,
    "reference": lambda m: m.get("ref_line", 0.0) >= Q.REF_LINE_TAG,
}


def evaluate_labels(items: list[dict]) -> dict:
    """
    Độ chính xác / độ bao phủ của từng bộ dò (DETECTORS) theo nhãn, tách hai nửa "tune" và "holdout".

    Chỉ tính dòng đã có nhãn hợp lệ. Với nhãn "clean" ghi số dòng bị bộ dò nào gắn cờ nhầm (dương tính giả).

    Returns:
        {"labeled": {nửa: số dòng}, "by_label": {nhãn: số dòng}, "detectors": {tên: {nửa: {tp, fp, fn, precision, recall}}}, "clean_false_flags": {nửa: {tên: số}}}.
    """
    rows = [it for it in items if it.get("label") in LABELS]
    res: dict = {"labeled": dict(Counter(it["split"] for it in rows)), "by_label": dict(Counter(it["label"] for it in rows)), "detectors": {}, "clean_false_flags": {}}
    for name, det in DETECTORS.items():
        per_split = {}
        for split in ("tune", "holdout"):
            sub = [it for it in rows if it["split"] == split]
            tp = sum(det(json.loads(it["quality_metrics"])) and it["label"] == name for it in sub)
            fp = sum(det(json.loads(it["quality_metrics"])) and it["label"] != name for it in sub)
            fn = sum((not det(json.loads(it["quality_metrics"]))) and it["label"] == name for it in sub)
            per_split[split] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None}
        res["detectors"][name] = per_split
    for split in ("tune", "holdout"):
        clean = [it for it in rows if it["split"] == split and it["label"] == "clean"]
        res["clean_false_flags"][split] = {name: sum(det(json.loads(it["quality_metrics"])) for it in clean) for name, det in DETECTORS.items()}
    return res


# ---------------------------------------------------------------------------------------------------------------- tập chữ đúng


def make_clean_set(rows: list[dict], n: int = 300, seed: int = 0) -> list[dict]:
    """
    Tập "chữ đúng không được đụng vào": đoạn đã giữ có dấu đầy đủ (vi_diacritic >= 0,2), không dính bất kỳ lỗi nào của ERROR_CHECKS, ưu tiên đoạn kinh tế / luật / y học (CLEAN_KEYWORDS).

    Returns:
        Tối đa n bản ghi {doc_id, journal, year, text}; đoạn có từ khoá lĩnh vực xếp trước.
    """
    rng = random.Random(seed)
    ok = [r for r in rows if r["quality_band"] in KEPT_BANDS and metrics_of(r).get("vi_diacritic", 0.0) >= 0.2 and not error_flags(r) and not suspicious(r)]
    rng.shuffle(ok)
    ok.sort(key=lambda r: not any(k in r["text"].lower() for k in CLEAN_KEYWORDS))
    return [{"doc_id": r["doc_id"], "journal": meta_of(r).get("journal_slug"), "year": meta_of(r).get("year"), "text": r["text"]} for r in ok[:n]]


def check_clean_set(items: list[dict], vocab: frozenset[str] | None = None, unaccented: frozenset[str] | None = None) -> list[dict]:
    """
    Chạy các bộ dò lên tập chữ đúng, trả về những đoạn bị gắn cờ nhầm (danh sách rỗng = đạt tiêu chí "0 cờ").

    Bộ dò: page_signals (sáu tín hiệu trang) và mã cứng của score_metrics với hồ sơ vjol.

    Returns:
        [{"doc_id", "signals", "hard_reasons"}] cho mỗi đoạn bị cờ.
    """
    profile = PROFILES["vjol"]
    bad = []
    for it in items:
        signals = page_signals(it["text"], DEFAULT_THRESHOLDS, vocab, unaccented)
        m = Q.compute_metrics(it["text"], vocab, unaccented)
        _, codes = Q.score_metrics(m, "vi", profile, 1.0, "{}")
        hard = [c for c in codes if c in HARD_CODES]
        if signals or hard:
            bad.append({"doc_id": it.get("doc_id"), "signals": signals, "hard_reasons": hard})
    return bad
