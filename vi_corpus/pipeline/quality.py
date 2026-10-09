"""
Stage quality: đo các chỉ số chất lượng văn bản, chấm điểm 0-100, xếp band A/B/C/D kèm reason code.

Chất lượng kỹ thuật và quyền sử dụng là hai trục tách biệt (docs/PIPELINE.md): stage này chỉ quyết định chất lượng.
- Lỗi cứng (hard): bản ghi tự động band D (too_short, too_long, empty, lang_not_allowed, low_alpha, replacement_chars,
  repeated_ngrams, duplicate_lines).
- Lỗi mềm (soft): trừ điểm, band theo điểm: A >= 85, B >= 70, C >= 55, còn lại D.
- Ngôn ngữ (từ stage language, D-02): low_lang_score (lang_score dưới SourceProfile.min_lang_score, trừ nhẹ),
  mixed_language (ngôn ngữ thứ hai >= 20%, chỉ gắn nhãn, không trừ điểm: đoạn song ngữ được giữ theo R-33),
  low_diacritic (văn bản nhận là tiếng Việt nhưng gần như không có dấu, số đo vi_diacritic).
Mọi số đo thô được lưu ở cột quality_metrics (JSON) để vẽ phân phối và chỉnh ngưỡng trong config.py.
Ngưỡng là điểm khởi đầu, cần chỉnh sau khi xem báo cáo (report.html) trên dữ liệu thật.
"""

import json
import re

from vi_corpus.pipeline.config import RunConfig, SourceProfile
from vi_corpus.pipeline.language import is_mixed, vi_diacritic_ratio

QUALITY_VERSION = "2"  # 2 = thêm low_lang_score, mixed_language, low_diacritic
MIN_VI_DIACRITIC = 0.05  # tiếng Việt có dấu bình thường ~0,2-0,3 chữ có dấu / chữ cái

_URL = re.compile(r"https?://\S+|www\.\S+")
_BULLET = re.compile(r"^\s*([-•*–·►▪]|\d+[.)])\s")
_BOILERPLATE = ("đăng nhập", "đăng ký", "liên hệ", "bản quyền", "copyright", "xem thêm", "đọc thêm", "chia sẻ",
                "bình luận", "tin liên quan", "cookie", "hotline", "quảng cáo", "all rights reserved", "click here",
                "privacy policy", "điều khoản", "theo dõi chúng tôi")
_ENDINGS = (".", "!", "?", "…", ":", ";", '"', "”", ")")
BAND_CUTS = (("A", 85), ("B", 70), ("C", 55))


def compute_metrics(text: str) -> dict:
    """
    Tính các số đo thô của một văn bản (không phụ thuộc nguồn).

    Các tỷ lệ tính trên số ký tự không phải khoảng trắng (alpha / digit / symbol / upper / replacement), số dòng
    (dup_line / short_line / bullet / end_punct / boilerplate_line) hoặc số trigram từ (rep_trigram).
    """
    words = text.split()
    nonspace = [c for c in text if not c.isspace()]
    n_chars = max(len(nonspace), 1)
    alpha = sum(c.isalpha() for c in nonspace)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    n_lines = max(len(lines), 1)
    trigrams = list(zip(words, words[1:], words[2:]))
    lowered = [ln.lower() for ln in lines]
    return {
        "words": len(words),
        "alpha": alpha / n_chars,
        "digit": sum(c.isdigit() for c in nonspace) / n_chars,
        "symbol": sum(not c.isalnum() for c in nonspace) / n_chars,
        "upper": sum(c.isupper() for c in nonspace) / max(alpha, 1),
        "replacement": text.count("�") / n_chars,
        "mean_word_len": sum(len(w) for w in words) / max(len(words), 1),
        "url": len(_URL.findall(text)) / max(len(words), 1),
        "dup_line": 1 - len(set(lines)) / n_lines if len(lines) >= 5 else 0.0,
        "short_line": sum(len(ln.split()) < 5 for ln in lines) / n_lines,
        "bullet": sum(bool(_BULLET.match(ln)) for ln in lines) / n_lines,
        "end_punct": sum(ln.endswith(_ENDINGS) for ln in lines) / n_lines,
        "boilerplate_line": sum(len(ln.split()) < 15 and any(b in ln for b in _BOILERPLATE) for ln in lowered) / n_lines,
        "rep_trigram": 1 - len(set(trigrams)) / len(trigrams) if len(trigrams) >= 30 else 0.0,
        "vi_diacritic": vi_diacritic_ratio(text),
        "lines": len(lines),
    }


def score_metrics(m: dict, lang: str | None, profile: SourceProfile, lang_score: float | None = None,
                  lang_mix: str | None = None) -> tuple[float, list[str]]:
    """
    Chấm điểm 0-100 và trả về reason code từ số đo thô, theo hồ sơ của nguồn.

    lang_score / lang_mix (từ stage language) là tuỳ chọn: thiếu thì bỏ qua các mã ngôn ngữ low_lang_score, mixed_language.

    Returns:
        (điểm, danh sách reason code). Có ít nhất một mã trong HARD_REASONS thì bản ghi bị band D bất kể điểm.
    """
    reasons: list[str] = []
    score = 100.0
    if m["words"] == 0:
        return 0.0, ["empty"]
    if m["words"] < profile.min_words:
        reasons.append("too_short")
    if m["words"] > profile.max_words:
        reasons.append("too_long")
    if lang not in profile.allowed_langs:
        reasons.append("lang_not_allowed")
    if m["alpha"] < profile.min_alpha:
        reasons.append("low_alpha")
    if m["replacement"] > 0.005:
        reasons.append("replacement_chars")
    if m["rep_trigram"] > 0.5:
        reasons.append("repeated_ngrams")
    if m["dup_line"] > 0.5:
        reasons.append("duplicate_lines")

    soft = [  # (điều kiện, mã, điểm trừ)
        (m["rep_trigram"] > 0.3, "repetitive", 20),
        (m["dup_line"] > 0.25, "dup_lines_some", 15),
        (m["digit"] > 0.3, "digit_heavy", 15),
        (m["symbol"] > 0.35, "symbol_heavy", 15),
        (m["upper"] > 0.5, "mostly_upper", 15),
        (not 2.5 <= m["mean_word_len"] <= 10, "odd_word_length", 15),
        (m["replacement"] > 0.0005, "some_replacement_chars", 10),
        (lang_score is not None and lang_score < profile.min_lang_score, "low_lang_score", 10),
        (lang_mix is not None and is_mixed(lang_mix), "mixed_language", 0),
        (lang == "vi" and m["words"] >= 20 and m.get("vi_diacritic", 1.0) < MIN_VI_DIACRITIC, "low_diacritic", 15),
    ]
    if profile.web_checks:
        soft += [
            (m["url"] > 0.1, "url_heavy", 20),
            (m["boilerplate_line"] > 0.15, "boilerplate", 20),
            (m["lines"] >= 5 and m["short_line"] > 0.7, "list_like", 15),
            (m["lines"] >= 5 and m["bullet"] > 0.5, "bullet_heavy", 10),
            (m["lines"] >= 5 and m["end_punct"] < 0.15, "few_sentences", 10),
        ]
    for cond, code, penalty in soft:
        if cond:
            reasons.append(code)
            score -= penalty
    if any(r in HARD_REASONS for r in reasons):
        score = min(score, 40.0)
    return max(score, 0.0), reasons


HARD_REASONS = frozenset({"empty", "too_short", "too_long", "lang_not_allowed", "low_alpha", "replacement_chars",
                          "repeated_ngrams", "duplicate_lines"})


def band_of(score: float) -> str:
    """Band chất lượng từ điểm: A >= 85, B >= 70, C >= 55, còn lại D."""
    for band, cut in BAND_CUTS:
        if score >= cut:
            return band
    return "D"


def annotate_quality(rows: list[dict], cfg: RunConfig) -> list[dict]:
    """
    Điền quality_metrics, quality_score, quality_band, reason_codes cho mọi bản ghi (sửa tại chỗ, trả lại chính list đó).

    Hàm thuần theo từng bản ghi nên chạy song song được (xem vi_corpus.pipeline.curator).
    """
    for row in rows:
        m = compute_metrics(row["text"] or "")
        score, codes = score_metrics(m, row["language"], cfg.profile(row["source_key"]), row.get("lang_score"),
                                     row.get("lang_mix"))
        row["quality_metrics"] = json.dumps({k: round(v, 4) for k, v in m.items()})
        row["quality_score"] = round(score, 1)
        row["quality_band"] = band_of(score)
        row["reason_codes"] = codes
    return rows


def quality_stats(rows: list[dict]) -> dict:
    """Thống kê của stage quality: số bản ghi theo band, số lần xuất hiện của từng reason code."""
    bands: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for row in rows:
        bands[row["quality_band"]] = bands.get(row["quality_band"], 0) + 1
        for c in row["reason_codes"]:
            reasons[c] = reasons.get(c, 0) + 1
    return {"bands": bands, "reasons": reasons}
