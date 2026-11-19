"""
Đo các bản ghi mẫu của một nguồn ứng viên và tổng hợp thành số liệu so sánh giữa các nguồn.

Dùng lại đúng số đo của pipeline để kết quả khớp với cách nguồn sẽ bị lọc khi chạy thật:
- ngôn ngữ: vi_corpus.pipeline.language.annotate_language (fastText lid.176 theo đoạn);
- chất lượng: vi_corpus.pipeline.quality.annotate_quality (điểm 0-100, band A-D, reason code), với hồ sơ đo riêng cho lần thử
  (không giới hạn độ dài trên vì pipeline cắt đoạn văn bản dài trước khi chấm; web thì bật kiểm tra boilerplate; hỏi-đáp thì
  hạ số từ tối thiểu);
- số token: tokenizer Qwen3 (vi_corpus.pipeline.tokens), hoặc đếm từ nếu không tải được tokenizer.
Thêm số đo riêng:
- công thức LaTeX bị hỏng (thường do Google Translate chèn khoảng trắng sau dấu "\\" hoặc dịch luôn tên lệnh: "\\ lần");
- dữ liệu dịch máy: tỷ lệ độ dài bản Việt / bản Anh, tỷ lệ bản "dịch" mà fastText vẫn nhận là tiếng Anh (chưa dịch).
"""

import json
import re
import statistics
from collections import Counter

from vi_corpus.pipeline.config import RunConfig, SourceProfile
from vi_corpus.pipeline.language import annotate_language
from vi_corpus.pipeline.quality import annotate_quality

PROBE_KEY = "source_probe"
_LATEX_CMD = re.compile(r"\\[A-Za-z]{2,}")  # lệnh LaTeX lành: \frac, \times, \sqrt...
_LATEX_BROKEN = re.compile(r"\\ +[A-Za-zÀ-ỹđĐ]")  # "\ lần", "\ frac": khoảng trắng chen vào sau "\"
_DOLLAR = re.compile(r"(?<!\\)\$")


def latex_stats(text: str) -> dict:
    """
    Số đo công thức LaTeX của một văn bản.

    Returns:
        has_latex: có lệnh LaTeX hoặc cặp "$"; broken: số chỗ "\\" bị chen khoảng trắng; odd_dollar: số dấu "$" lẻ (công thức
        mở mà không đóng, hay gặp khi dịch máy làm rơi dấu).
    """
    cmds = len(_LATEX_CMD.findall(text))
    broken = len(_LATEX_BROKEN.findall(text))
    dollars = len(_DOLLAR.findall(text))
    return {"has_latex": cmds > 0 or broken > 0 or dollars >= 2, "broken": broken, "odd_dollar": dollars % 2 == 1}


def probe_profile(is_web: bool, is_sft: bool) -> SourceProfile:
    """Hồ sơ đo cho lần thử: không trần độ dài, web bật boilerplate, hỏi-đáp hạ số từ tối thiểu."""
    return SourceProfile(web_checks=is_web, max_words=10**9, min_words=5 if is_sft else 50,
                         min_alpha=0.4 if is_sft else 0.55)


def _counter(spec: str):
    """Bộ đếm token Qwen3; không tải được (mất mạng) thì đếm từ."""
    from vi_corpus.pipeline.tokens import get_counter
    try:
        return get_counter(spec)
    except Exception:  # noqa: BLE001 - mất mạng / thiếu transformers: quay về đếm từ, báo cáo ghi rõ tokenizer đã dùng
        return get_counter("words")


def measure_records(records: list[dict], is_web: bool = False, is_sft: bool = False,
                    lang_model: str = "fasttext:lid.176", tokenizer: str = "hf:Qwen/Qwen3-0.6B") -> list[dict]:
    """
    Đo từng bản ghi mẫu; trả về list dict (giữ id, extra) có thêm language, lang_score, quality_*, reason_codes, tokens,
    latex, và với dữ liệu dịch máy: en_language, len_ratio.
    """
    rows = [{"source_key": PROBE_KEY, "doc_id": r["id"], "text": r["text"] or "", "extra": r.get("extra", {}),
             "en": r.get("en", "")} for r in records]
    annotate_language(rows, lang_model)
    cfg = RunConfig(mix={}, profiles={PROBE_KEY: probe_profile(is_web, is_sft)})
    annotate_quality(rows, cfg)
    counter = _counter(tokenizer)
    for row, n_tok in zip(rows, counter.count_batch([r["text"] for r in rows])):
        row["tokens"] = n_tok
        row["tokenizer"] = counter.spec
        row["latex"] = latex_stats(row["text"])
        if row["en"]:
            row["len_ratio"] = round(len(row["text"]) / max(1, len(row["en"])), 3)
    with_en = [r for r in rows if r["en"]]
    if with_en:
        # bản dịch mà vẫn là tiếng Anh: so ngôn ngữ của bản "vi" (đã đo ở trên); thêm ngôn ngữ của bản gốc để kiểm tra chéo
        en_rows = annotate_language([{"text": r["en"]} for r in with_en], lang_model)
        for r, e in zip(with_en, en_rows):
            r["en_language"] = e["language"]
    return rows


def _pct(values: list[float], q: float) -> float:
    """Phân vị q (0-1) của một danh sách số, 0 nếu rỗng."""
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def summarize(rows: list[dict]) -> dict:
    """Tổng hợp số đo của một nguồn: ngôn ngữ, band, reason code, token, LaTeX, dịch máy, cột số đo sẵn có."""
    n = len(rows)
    if not n:
        return {"n": 0}
    tokens = [r["tokens"] for r in rows]
    bands = Counter(r["quality_band"] for r in rows)
    reasons = Counter(c for r in rows for c in r["reason_codes"])
    latex_rows = [r for r in rows if r["latex"]["has_latex"]]
    out = {
        "n": n,
        "tokenizer": rows[0].get("tokenizer"),
        "tokens_median": statistics.median(tokens),
        "tokens_p10": _pct(tokens, 0.1),
        "tokens_p90": _pct(tokens, 0.9),
        "vi_share": round(sum(r["language"] == "vi" for r in rows) / n, 3),
        "lang_top": Counter(r["language"] for r in rows).most_common(4),
        "score_median": statistics.median(r["quality_score"] for r in rows),
        "band_share": {b: round(bands.get(b, 0) / n, 3) for b in "ABCD"},
        "reasons_top": [(c, round(k / n, 3)) for c, k in reasons.most_common(8)],
        "latex_share": round(len(latex_rows) / n, 3),
        "latex_broken_share": round(sum(r["latex"]["broken"] > 0 for r in latex_rows) / len(latex_rows), 3) if latex_rows else None,
        "odd_dollar_share": round(sum(r["latex"]["odd_dollar"] for r in latex_rows) / len(latex_rows), 3) if latex_rows else None,
    }
    mt = [r for r in rows if r.get("en")]
    if mt:
        out["mt_len_ratio_median"] = statistics.median(r["len_ratio"] for r in mt)
        out["mt_untranslated_share"] = round(sum(r["language"] == "en" for r in mt) / len(mt), 3)
        out["mt_empty_share"] = round(sum(not r["text"].strip() for r in mt) / len(mt), 3)
    out["extra"] = summarize_extra([r["extra"] for r in rows])
    return out


def _scalar(value):
    """Rút một số đại diện từ giá trị cột: số giữ nguyên, list số lấy trung bình, còn lại None."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
        return sum(value) / len(value)
    return None


def summarize_extra(extras: list[dict]) -> dict:
    """Tổng hợp cột sẵn có: cột số (hoặc list số) lấy trung vị / p10 / p90, cột chuỗi ngắn đếm 5 giá trị nhiều nhất."""
    out: dict = {}
    keys = sorted({k for e in extras for k in e})
    for k in keys:
        vals = [e.get(k) for e in extras if e.get(k) is not None]
        nums = [s for s in (_scalar(v) for v in vals) if s is not None]
        if nums and len(nums) >= len(vals) / 2:
            out[k] = {"median": round(statistics.median(nums), 3), "p10": round(_pct(nums, 0.1), 3),
                      "p90": round(_pct(nums, 0.9), 3)}
        elif k != "url":
            out[k] = Counter(str(v)[:40] for v in vals).most_common(5)
        else:
            hosts = Counter(re.sub(r"^https?://(www\.)?", "", str(v)).split("/")[0] for v in vals)
            out["url_hosts_top"] = hosts.most_common(10)
    return out


def dump_rows(rows: list[dict], path) -> None:
    """Ghi bản ghi đã đo ra jsonl (bỏ trường en để file gọn; giữ 2.000 ký tự đầu văn bản để đọc lại)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            keep = {k: v for k, v in r.items() if k not in ("en", "text", "source_key")}
            keep["text_head"] = r["text"][:2000]
            f.write(json.dumps(keep, ensure_ascii=False) + "\n")
