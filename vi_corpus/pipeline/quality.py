"""
Stage quality: đo các chỉ số chất lượng văn bản, chấm điểm 0-100, xếp band A/B/C/D kèm reason code.

Chất lượng kỹ thuật và quyền sử dụng là hai trục tách biệt (docs/PIPELINE.md): stage này chỉ quyết định chất lượng.
- Lỗi cứng (hard): bản ghi tự động band D (too_short, too_long, empty, lang_not_allowed, low_alpha, replacement_chars,
  repeated_ngrams, duplicate_lines, garbled_font).
- Lỗi mềm (soft): trừ điểm, band theo điểm: A >= 85, B >= 70, C >= 55, còn lại D.
- Ngôn ngữ (từ stage language, D-02): low_lang_score (lang_score dưới SourceProfile.min_lang_score, trừ nhẹ),
  mixed_language (ngôn ngữ thứ hai >= 20%, chỉ gắn nhãn, không trừ điểm: đoạn song ngữ được giữ theo R-33),
  low_diacritic (văn bản nhận là tiếng Việt nhưng gần như không có dấu, số đo vi_diacritic).
  lang_unknown: không còn chữ cái để nhận diện (language "und", F-07), chỉ gắn nhãn, không trừ điểm.
  garbled_font (J-10): chữ hỏng do font PDF (từ có >= 2 dấu thanh chiếm tỷ lệ cao, vi_corpus.common.garbled); lưới an toàn khi
  trang lỗi font lọt qua bước trích text, vì chữ hỏng vẫn toàn chữ Latinh có dấu nên fastText nhận là tiếng Việt.
  Bốn mã cứng của J-22/J-23 (chữ hỏng mà fastText không thấy): private_use_chars (còn >= PUA_HARD_CHARS ký tự vùng riêng sau khi
  normalize đã đổi font Symbol), math_font_junk (>= JUNK_HARD_CHARS chữ Ấn Độ / Ả Rập / Thái, rác của font toán; chữ Hán không tính),
  lost_diacritics (tiếng Việt mất dấu, vi_corpus.common.garbled.unaccented_ratio), reversed_text (chữ đảo ngược, garbled.reversed_gain,
  cần tập âm tiết). odd_words (chỉ nguồn bật SourceProfile.odd_words) là "từ lạ": từ Latinh không phải âm tiết Việt hợp lệ và xuất hiện ở
  < ODD_MIN_DOCS bài của lần chạy, tính ở annotate_odd_words sau khi cả lần chạy đã có số đo. reference_list chỉ gắn nhãn (không trừ
  điểm): đoạn phần lớn là dòng danh mục tài liệu tham khảo còn sót, nhận theo mẫu đầu dòng (vi_corpus.common.references, J-25 E).
  lost_letters (J-25 B, cứng): font mất ư / ơ thành dấu gạch nối ("đ-ợc", "ng-ời"). high_perplexity (J-25 D3, mềm): perplexity cao theo mô hình ngôn ngữ nhỏ; chỉ khi kèm
  tỷ lệ từ lạ vừa phải (ODD_SOFT) mới thành lỗi cứng odd_words, vì bảng số / công thức / tên riêng cũng có perplexity cao.
Công thức / code (F-03): các số đo hình dạng (rep_trigram, symbol, digit, upper, mean_word_len, alpha, vi_diacritic)
tính trên phần văn xuôi còn lại sau vi_corpus.pipeline.spans.strip_math_code; words, lines, dup_line, end_punct... giữ
trên văn bản gốc; math_share = tỷ lệ ký tự bị bỏ. Văn bản gần như toàn công thức / code (văn xuôi < MATH_PROSE_WORDS từ
và math_share >= MATH_SHARE_MIN) được miễn MATH_EXEMPT; chống vòng lặp trong công thức bằng rep_trigram_raw (trên văn bản
gốc) >= RAW_REPEAT_HARD -> repeated_ngrams.
Mọi số đo thô được lưu ở cột quality_metrics (JSON) để vẽ phân phối và chỉnh ngưỡng trong config.py.
Ngưỡng là điểm khởi đầu, cần chỉnh sau khi xem báo cáo (report.html) trên dữ liệu thật.
"""

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from vi_corpus.common.garbled import (DEFAULT_THRESHOLDS, GARBLED_HARD_RATIO, GARBLED_MIN_WORDS, garbled_ratio, hyphen_lost,
                                      latin_tokens, load_unaccented, load_vocab, reversed_gain, unaccented_ratio)
from vi_corpus.common.references import reference_share
from vi_corpus.pipeline.config import RunConfig, SourceProfile
from vi_corpus.pipeline.language import UNDETERMINED, is_mixed, vi_diacritic_ratio
from vi_corpus.pipeline.spans import math_share, strip_math_code

QUALITY_VERSION = "7"  # 2 = thêm low_lang_score, mixed_language, low_diacritic; 3 = dup_line bỏ dòng không có chữ
# cái (F-01), số đo hình dạng trên văn xuôi, miễn cho văn bản toàn công thức / code (F-03), lang_unknown (F-07);
# 4 = thêm số đo garbled và mã cứng garbled_font (J-10); 5 = private_use_chars, math_font_junk, lost_diacritics, reversed_text,
# odd_words, reference_list (J-22, J-23); 6 = J-25: lost_letters (font mất ư / ơ), reference_list nhận theo mẫu đầu dòng, từ lạ đếm theo tạp chí,
# perplexity (high_perplexity), danh sách từ không dấu sinh từ dữ liệu; 7 = J-28: lost_diacritics bỏ đường dẫn / văn bản tiếng Anh và tính mỗi từ tối đa 3 lần,
# lost_letters không khớp "C- section"
MIN_VI_DIACRITIC = 0.05  # tiếng Việt có dấu bình thường ~0,2-0,3 chữ có dấu / chữ cái
# F-03, ngưỡng khởi điểm chưa đo: văn xuôi < MATH_PROSE_WORDS từ và math_share >= MATH_SHARE_MIN thì miễn MATH_EXEMPT
MATH_PROSE_WORDS = 20
MATH_SHARE_MIN = 0.3
RAW_REPEAT_HARD = 0.8  # rep_trigram_raw (văn bản gốc, kể cả công thức) từ mức này là vòng lặp: repeated_ngrams
MATH_EXEMPT = frozenset({"repetitive", "symbol_heavy", "digit_heavy", "odd_word_length", "repeated_ngrams",
                         "low_alpha"})

# J-23, các ngưỡng dưới đây là điểm khởi đầu từ số đo J-22 (chưa hiệu chỉnh trên đọc tay 300 + 100 đoạn của J-22 (10)); số đo thô nằm trong quality_metrics để chỉnh.
PUA_HARD_CHARS = 5  # từ ngần này ký tự vùng riêng còn sót thì band D
JUNK_HARD_CHARS = 5  # từ ngần này chữ Ấn Độ / Ả Rập / Thái thì band D (J-22 (5)); không áp cho chữ Hán (bài trích Hán-Nôm hợp lệ)
UNACCENTED_MIN_WORDS = 50  # số từ Latinh tối thiểu để đo tỷ lệ mất dấu
REVERSED_MIN_WORDS = 30  # số từ (>= 3 chữ cái) tối thiểu để đo chữ đảo ngược
REVERSED_HARD = 0.25  # độ lợi tỷ lệ âm tiết hợp lệ khi viết ngược từ mức này là chữ đảo ngược
REVERSED_MIN_REV = 0.5  # và bản viết ngược phải có ít nhất tỷ lệ này từ hợp lệ
REF_LINE_TAG = 0.5  # đoạn có từ tỷ lệ này số dòng là mục tài liệu tham khảo thì gắn reference_list
REF_MIN_LINES = 5
ODD_MIN_WORDS = 50  # số từ Latinh tối thiểu để tính tỷ lệ từ lạ
ODD_MIN_DOCS = 3  # từ lạ xuất hiện ở ít hơn ngần ấy bài của lần chạy mới tính là "lạ" (từ lặp lại ở nhiều bài là thuật ngữ thật)
ODD_ABBREV_MAX = 6  # từ viết hoa toàn bộ không dài hơn ngần này chữ cái là viết tắt, miễn
ODD_HARD = 0.15  # tỷ lệ từ lạ từ mức này thì band D (J-22 (3) C)
ODD_MIN_JOURNALS = 3  # J-25 D1: từ lạ xuất hiện ở ít hơn ngần ấy TẠP CHÍ mới tính là "lạ" (lỗi font lặp lại trong cả một tạp chí không còn bị coi là từ thật)
ODD_MIN_JOURNALS_RUN = 10  # lần chạy có ít tạp chí hơn ngần ấy thì quay về đếm theo bài (mẫu nhỏ, tạp chí không đại diện)
ODD_SOFT = 0.08  # tỷ lệ từ lạ vừa phải; kèm perplexity cao (PPL_HIGH) thì thành odd_words (J-25 D3)
PPL_HIGH = 120.0  # perplexity (Qwen3-0.6B, 512 token đầu) từ mức này là cao; điểm khởi đầu chưa đo, chỉnh bằng nửa chỉnh của bộ đáp án (J-25 Bước 0)
HYPHEN_HARD_COUNT = 5  # từ ngần này chỗ font mất ư / ơ thành "-" trong một đoạn thì band D (lost_letters, J-25 B)

_PUA = re.compile("[\ue000-\uf8ff]")
_JUNK_SCRIPT = re.compile("[\u0600-\u06ff\u0900-\u0eff]")
_URL = re.compile(r"https?://\S+|www\.\S+")
_BULLET = re.compile(r"^\s*([-•*–·►▪]|\d+[.)])\s")
_BOILERPLATE = ("đăng nhập", "đăng ký", "liên hệ", "bản quyền", "copyright", "xem thêm", "đọc thêm", "chia sẻ",
                "bình luận", "tin liên quan", "cookie", "hotline", "quảng cáo", "all rights reserved", "click here",
                "privacy policy", "điều khoản", "theo dõi chúng tôi")
_ENDINGS = (".", "!", "?", "…", ":", ";", '"', "”", ")")
BAND_CUTS = (("A", 85), ("B", 70), ("C", 55))


def _rep_trigram(words: list[str]) -> float:
    """Tỷ lệ trigram từ bị lặp (0 nếu < 30 trigram)."""
    trigrams = list(zip(words, words[1:], words[2:]))
    return 1 - len(set(trigrams)) / len(trigrams) if len(trigrams) >= 30 else 0.0


def compute_metrics(text: str, vocab: frozenset[str] | None = None, unaccented: frozenset[str] | None = None) -> dict:
    """
    Tính các số đo thô của một văn bản (không phụ thuộc nguồn).

    Các tỷ lệ tính trên số ký tự không phải khoảng trắng (alpha / digit / symbol / upper / replacement), số dòng
    (dup_line / short_line / bullet / end_punct / boilerplate_line) hoặc số trigram từ (rep_trigram); garbled là tỷ lệ từ có
    >= 2 dấu thanh trên các từ >= 2 chữ cái, tính trên văn bản gốc, 0 nếu dưới GARBLED_MIN_WORDS từ. alpha, digit,
    symbol, upper, mean_word_len, rep_trigram, vi_diacritic đo trên văn xuôi (đã bỏ công thức / code, F-03); dup_line chỉ
    xét dòng có chữ cái (dòng "$$", "}", "---" lặp là cú pháp, F-01).

    Số đo của J-23, đều trên văn bản gốc: pua_chars (số ký tự vùng riêng), junk_script_chars (số chữ Ấn Độ / Ả Rập / Thái), unaccented (tỷ lệ
    từ không dấu thông dụng, 0 nếu dưới UNACCENTED_MIN_WORDS từ Latinh), reversed_gain (độ lợi âm tiết hợp lệ khi viết ngược từ, 0 nếu
    không có `vocab` hoặc dưới REVERSED_MIN_WORDS từ hoặc bản viết ngược dưới REVERSED_MIN_REV), ref_line (tỷ lệ dòng là mục tài liệu tham khảo,
    0 nếu dưới REF_MIN_LINES dòng), odd_word (luôn 0 ở đây; annotate_odd_words điền sau), perplexity (luôn 0 ở đây; annotate_perplexity điền sau), hyphen_lost (số chỗ font mất
    ư / ơ thành dấu gạch nối, J-25 B). `unaccented` là danh sách từ không dấu của lost_diacritics (None = danh sách tối thiểu).
    """
    words = text.split()
    prose = strip_math_code(text)
    pwords = prose.split()
    nonspace = [c for c in prose if not c.isspace()]
    n_chars = max(len(nonspace), 1)
    alpha = sum(c.isalpha() for c in nonspace)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    n_lines = max(len(lines), 1)
    garbled_share, garbled_words = garbled_ratio(text)
    garbled = garbled_share if garbled_words >= GARBLED_MIN_WORDS else 0.0
    worded = [ln for ln in lines if any(c.isalpha() for c in ln)]
    lowered = [ln.lower() for ln in lines]
    unaccented_rate, n_latin = unaccented_ratio(text, unaccented)
    _, _, n_hyphen = hyphen_lost(text)
    gain, rev_known, n_rev = reversed_gain(text, vocab) if vocab is not None else (0.0, 0.0, 0)
    return {
        "words": len(words),
        "prose_words": len(pwords),
        "math_share": math_share(text, prose),
        "alpha": alpha / n_chars,
        "digit": sum(c.isdigit() for c in nonspace) / n_chars,
        "symbol": sum(not c.isalnum() for c in nonspace) / n_chars,
        "upper": sum(c.isupper() for c in nonspace) / max(alpha, 1),
        "replacement": text.count("\ufffd") / max(sum(not c.isspace() for c in text), 1),
        "mean_word_len": sum(len(w) for w in pwords) / max(len(pwords), 1),
        "url": len(_URL.findall(text)) / max(len(words), 1),
        "dup_line": 1 - len(set(worded)) / len(worded) if len(worded) >= 5 else 0.0,
        "short_line": sum(len(ln.split()) < 5 for ln in lines) / n_lines,
        "bullet": sum(bool(_BULLET.match(ln)) for ln in lines) / n_lines,
        "end_punct": sum(ln.endswith(_ENDINGS) for ln in lines) / n_lines,
        "boilerplate_line": sum(len(ln.split()) < 15 and any(b in ln for b in _BOILERPLATE) for ln in lowered) / n_lines,
        "rep_trigram": _rep_trigram(pwords),
        "rep_trigram_raw": _rep_trigram(words),
        "vi_diacritic": vi_diacritic_ratio(prose),
        "garbled": garbled,
        "lines": len(lines),
        "pua_chars": len(_PUA.findall(text)),
        "junk_script_chars": len(_JUNK_SCRIPT.findall(text)),
        "unaccented": unaccented_rate if n_latin >= UNACCENTED_MIN_WORDS else 0.0,
        "hyphen_lost": n_hyphen,
        "reversed_gain": gain if n_rev >= REVERSED_MIN_WORDS and rev_known >= REVERSED_MIN_REV else 0.0,
        "ref_line": reference_share(lines) if len(lines) >= REF_MIN_LINES else 0.0,
        "odd_word": 0.0,
        "perplexity": 0.0,
    }


def is_math_only(m: dict) -> bool:
    """Văn bản gần như toàn công thức / code (F-03): văn xuôi < MATH_PROSE_WORDS từ và math_share >= MATH_SHARE_MIN."""
    return m.get("prose_words", m["words"]) < MATH_PROSE_WORDS and m.get("math_share", 0.0) >= MATH_SHARE_MIN


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
    if lang != UNDETERMINED and lang not in profile.allowed_langs:
        reasons.append("lang_not_allowed")
    if m["alpha"] < profile.min_alpha:
        reasons.append("low_alpha")
    if m["replacement"] > 0.005:
        reasons.append("replacement_chars")
    if m["rep_trigram"] > 0.5 or m.get("rep_trigram_raw", 0.0) >= RAW_REPEAT_HARD:
        reasons.append("repeated_ngrams")
    if m["dup_line"] > 0.5:
        reasons.append("duplicate_lines")
    if m.get("garbled", 0.0) >= GARBLED_HARD_RATIO:
        reasons.append("garbled_font")
    if m.get("pua_chars", 0) >= PUA_HARD_CHARS:
        reasons.append("private_use_chars")
    if m.get("junk_script_chars", 0) >= JUNK_HARD_CHARS:
        reasons.append("math_font_junk")
    if m.get("unaccented", 0.0) >= DEFAULT_THRESHOLDS.nodiac:
        reasons.append("lost_diacritics")
    if m.get("reversed_gain", 0.0) >= REVERSED_HARD:
        reasons.append("reversed_text")
    if m.get("hyphen_lost", 0) >= HYPHEN_HARD_COUNT:
        reasons.append("lost_letters")
    if profile.odd_words and (m.get("odd_word", 0.0) >= ODD_HARD
                              or (m.get("odd_word", 0.0) >= ODD_SOFT and m.get("perplexity", 0.0) >= PPL_HIGH)):
        reasons.append("odd_words")

    soft = [  # (điều kiện, mã, điểm trừ)
        (m["rep_trigram"] > 0.3, "repetitive", 20),
        (m["dup_line"] > 0.25, "dup_lines_some", 15),
        (m["digit"] > 0.3, "digit_heavy", 15),
        (m["symbol"] > 0.35, "symbol_heavy", 15),
        (m["upper"] > 0.5, "mostly_upper", 15),
        (not 2.5 <= m["mean_word_len"] <= 10, "odd_word_length", 15),
        (m["replacement"] > 0.0005, "some_replacement_chars", 10),
        (lang == UNDETERMINED, "lang_unknown", 0),
        (lang != UNDETERMINED and lang_score is not None and lang_score < profile.min_lang_score, "low_lang_score", 10),
        (lang_mix is not None and is_mixed(lang_mix), "mixed_language", 0),
        (lang == "vi" and m["words"] >= 20 and m.get("vi_diacritic", 1.0) < MIN_VI_DIACRITIC, "low_diacritic", 15),
        (m.get("ref_line", 0.0) >= REF_LINE_TAG, "reference_list", 0),
        (m.get("perplexity", 0.0) >= PPL_HIGH, "high_perplexity", 10),
    ]
    if profile.web_checks:
        soft += [
            (m["url"] > 0.1, "url_heavy", 20),
            (m["boilerplate_line"] > 0.15, "boilerplate", 20),
            (m["lines"] >= 5 and m["short_line"] > 0.7, "list_like", 15),
            (m["lines"] >= 5 and m["bullet"] > 0.5, "bullet_heavy", 10),
            (m["lines"] >= 5 and m["end_punct"] < 0.15, "few_sentences", 10),
        ]
    math_only = is_math_only(m)
    if math_only:  # F-03: bỏ các mã hình dạng; vòng lặp trong công thức vẫn bị rep_trigram_raw bắt
        reasons = [r for r in reasons if r not in MATH_EXEMPT or (r == "repeated_ngrams"
                                                                   and m.get("rep_trigram_raw", 0.0) >= RAW_REPEAT_HARD)]
    for cond, code, penalty in soft:
        if cond and not (math_only and code in MATH_EXEMPT):
            reasons.append(code)
            score -= penalty
    if any(r in HARD_REASONS for r in reasons):
        score = min(score, 40.0)
    return max(score, 0.0), reasons


HARD_REASONS = frozenset({"empty", "too_short", "too_long", "lang_not_allowed", "low_alpha", "replacement_chars",
                          "repeated_ngrams", "duplicate_lines", "garbled_font", "private_use_chars", "math_font_junk",
                          "lost_diacritics", "reversed_text", "odd_words", "lost_letters"})


def band_of(score: float) -> str:
    """Band chất lượng từ điểm: A >= 85, B >= 70, C >= 55, còn lại D."""
    for band, cut in BAND_CUTS:
        if score >= cut:
            return band
    return "D"


@lru_cache(maxsize=4)
def _load_vocab(path: str) -> frozenset[str] | None:
    """Nạp tập âm tiết từ file (nhớ kết quả theo đường dẫn); None nếu file không tồn tại."""
    return load_vocab(Path(path))


@lru_cache(maxsize=4)
def _load_unaccented(path: str) -> frozenset[str] | None:
    """Nạp danh sách từ không dấu từ file (nhớ kết quả theo đường dẫn); None nếu file không tồn tại."""
    return load_unaccented(Path(path))


def unaccented_for(cfg: RunConfig) -> frozenset[str] | None:
    """Danh sách từ không dấu của lần chạy (cfg.unaccented); None nếu không đặt hoặc file không có (khi đó dùng danh sách tối thiểu)."""
    return _load_unaccented(cfg.unaccented) if cfg.unaccented else None


def vocab_for(cfg: RunConfig) -> frozenset[str] | None:
    """Tập âm tiết của lần chạy (cfg.syllables); None nếu không đặt hoặc file không có (khi đó bỏ reversed_text và odd_words)."""
    return _load_vocab(cfg.syllables) if cfg.syllables else None


def vocab_identity(cfg: RunConfig) -> list | None:
    """Dấu nhận dạng tập âm tiết cho vân tay stage quality: [đường dẫn, kích thước]; None nếu không dùng (đổi file thì stage chạy lại)."""
    return _file_identity(cfg.syllables)


def _file_identity(path_str: str | None) -> list | None:
    """[đường dẫn, kích thước] của một file phụ trợ để băm vân tay; None nếu không đặt hoặc file không có."""
    if not path_str:
        return None
    path = Path(path_str)
    return [str(path), path.stat().st_size] if path.is_file() else None


def unaccented_identity(cfg: RunConfig) -> list | None:
    """Dấu nhận dạng danh sách từ không dấu cho vân tay stage quality (đổi file thì stage chạy lại); None nếu không dùng."""
    return _file_identity(cfg.unaccented)


def annotate_quality(rows: list[dict], cfg: RunConfig) -> list[dict]:
    """
    Điền quality_metrics, quality_score, quality_band, reason_codes cho mọi bản ghi (sửa tại chỗ, trả lại chính list đó).

    Hàm thuần theo từng bản ghi nên chạy song song được (xem vi_corpus.pipeline.curator). Tập âm tiết (cfg.syllables) nạp một lần mỗi tiến trình.
    """
    vocab = vocab_for(cfg)
    unaccented = unaccented_for(cfg)
    for row in rows:
        m = compute_metrics(row["text"] or "", vocab, unaccented)
        score, codes = score_metrics(m, row["language"], cfg.profile(row["source_key"]), row.get("lang_score"),
                                     row.get("lang_mix"))
        row["quality_metrics"] = json.dumps({k: round(v, 4) for k, v in m.items()})
        row["quality_score"] = round(score, 1)
        row["quality_band"] = band_of(score)
        row["reason_codes"] = codes
    return rows


def odd_tokens(text: str, vocab: frozenset[str]) -> tuple[list[str], int]:
    """
    Các từ "ứng viên lạ" của một văn bản: từ Latinh (chữ thường) không phải âm tiết trong `vocab`, trừ viết tắt (viết hoa toàn bộ, <= ODD_ABBREV_MAX
    chữ cái) và tên khoa học (hai từ ASCII liền nhau dạng "Escherichia coli": Genus viết hoa rồi tên loài viết thường).

    Returns:
        (danh sách từ ứng viên, tổng số từ Latinh).
    """
    toks = latin_tokens(text)
    out: list[str] = []
    for i, t in enumerate(toks):
        low = t.lower()
        if low in vocab or (t.isupper() and len(t) <= ODD_ABBREV_MAX):
            continue
        prev, nxt = toks[i - 1] if i else "", toks[i + 1] if i + 1 < len(toks) else ""
        genus = t.isascii() and t[:1].isupper() and t[1:].islower() and len(t) >= 4
        species = t.isascii() and t.islower() and len(t) >= 3
        if (genus and nxt.isascii() and nxt.islower() and len(nxt) >= 3) or \
                (species and prev.isascii() and prev[:1].isupper() and prev[1:].islower() and len(prev) >= 4):
            continue
        out.append(low)
    return out, len(toks)


def _journal_of(row: dict) -> str | None:
    """Mã tạp chí (journal_slug trong cột meta) của một bản ghi; None nếu không có."""
    try:
        meta = json.loads(row.get("meta") or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    return meta.get("journal_slug") if isinstance(meta, dict) else None


def annotate_odd_words(rows: list[dict], cfg: RunConfig) -> dict:
    """
    Chỉ số "từ lạ" (J-22 (3) C): với các nguồn bật SourceProfile.odd_words, tính tỷ lệ từ lạ của từng đoạn tiếng Việt rồi chấm điểm lại.

    Từ lạ = ứng viên (odd_tokens) xuất hiện ở ít hơn ODD_MIN_JOURNALS TẠP CHÍ (J-25 D1) của lần chạy: lỗi font / OCR hiếm khi lặp lại ở nhiều tạp chí, còn thuật ngữ và
    tên riêng thì có. Đếm theo bài (ODD_MIN_DOCS) như bản cũ khiến lỗi lặp lại cả trong một tạp chí bị coi là từ thật; nay chỉ còn dùng cách đó cho bản ghi không có tạp chí hoặc khi
    lần chạy có dưới ODD_MIN_JOURNALS_RUN tạp chí (mẫu quá nhỏ). Cần số tạp chí / bài chứa từ trên CẢ lần chạy nên không chạy song song theo lô được; chạy tuần tự sau annotate_quality.
    Ghi odd_word vào quality_metrics rồi gọi lại score_metrics (điểm, band, reason code cập nhật tại chỗ). Không có tập âm tiết thì bỏ qua.

    Returns:
        {"checked": số đoạn được tính, "flagged": số đoạn bị mã odd_words, "skipped": lý do bỏ qua hoặc None, "unit": "journal" | "doc", "journals": số tạp chí}.
    """
    vocab = vocab_for(cfg)
    gated = [r for r in rows if cfg.profile(r["source_key"]).odd_words]
    if not gated:
        return {"checked": 0, "flagged": 0, "skipped": "không nguồn nào bật odd_words"}
    if vocab is None:
        return {"checked": 0, "flagged": 0, "skipped": "thiếu tập âm tiết (RunConfig.syllables)"}
    found = [odd_tokens(r["text"] or "", vocab) for r in gated]
    journals = [_journal_of(r) for r in gated]
    n_journals = len({j for j in journals if j})
    by_journal = n_journals >= ODD_MIN_JOURNALS_RUN
    groups: set[tuple[str, str, str]] = set()  # (nguồn, từ, tạp chí hoặc bài)
    for r, (cands, _), journal in zip(gated, found, journals):
        unit = journal if by_journal and journal else (r.get("parent_doc_id") or r["doc_id"])
        groups.update((r["source_key"], w, unit) for w in set(cands))
    df: Counter = Counter((src, w) for src, w, _ in groups)
    checked = flagged = 0
    for r, (cands, n), journal in zip(gated, found, journals):
        if r.get("language") != "vi" or n < ODD_MIN_WORDS:
            continue
        need = ODD_MIN_JOURNALS if by_journal and journal else ODD_MIN_DOCS
        rate = sum(df[(r["source_key"], w)] < need for w in cands) / n
        m = json.loads(r["quality_metrics"])
        m["odd_word"] = round(rate, 4)
        score, codes = score_metrics(m, r["language"], cfg.profile(r["source_key"]), r.get("lang_score"), r.get("lang_mix"))
        r["quality_metrics"] = json.dumps(m)
        r["quality_score"], r["quality_band"], r["reason_codes"] = round(score, 1), band_of(score), codes
        checked += 1
        flagged += "odd_words" in codes
    return {"checked": checked, "flagged": flagged, "skipped": None, "unit": "journal" if by_journal else "doc", "journals": n_journals}


def annotate_perplexity(rows: list[dict], cfg: RunConfig, scorer=None) -> dict:
    """
    Perplexity của đoạn (J-25 D3) cho các nguồn bật SourceProfile.perplexity, rồi chấm điểm lại: ghi perplexity vào quality_metrics, mã high_perplexity (mềm) khi từ PPL_HIGH,
    và mã cứng odd_words khi kèm tỷ lệ từ lạ từ ODD_SOFT (xem score_metrics). Chạy sau annotate_odd_words (cần odd_word đã điền). Không phải cổng duy nhất: một mình perplexity chỉ trừ điểm.

    Args:
        scorer: Đối tượng có .score(list[str]) -> list[float | None] (mặc định LmScorer của cfg.perplexity_model); truyền giả trong test.

    Returns:
        {"checked": số đoạn được chấm, "high": số đoạn có perplexity >= PPL_HIGH, "flagged": số đoạn bị mã odd_words, "skipped": lý do bỏ qua hoặc None, "median": trung vị}.
    """
    gated = [r for r in rows if cfg.profile(r["source_key"]).perplexity and r.get("language") == "vi"]
    if not cfg.perplexity_model and scorer is None:
        return {"checked": 0, "high": 0, "flagged": 0, "skipped": "perplexity_model chưa đặt"}
    if not gated:
        return {"checked": 0, "high": 0, "flagged": 0, "skipped": "không nguồn nào bật perplexity"}
    if scorer is None:
        from vi_corpus.pipeline.perplexity import LmScorer

        scorer = LmScorer(cfg.perplexity_model, cfg.perplexity_device)
    values = scorer.score([r["text"] or "" for r in gated])
    checked = high = flagged = 0
    seen: list[float] = []
    for r, ppl in zip(gated, values):
        if ppl is None:
            continue
        m = json.loads(r["quality_metrics"])
        m["perplexity"] = round(ppl, 2)
        score, codes = score_metrics(m, r["language"], cfg.profile(r["source_key"]), r.get("lang_score"), r.get("lang_mix"))
        r["quality_metrics"] = json.dumps(m)
        r["quality_score"], r["quality_band"], r["reason_codes"] = round(score, 1), band_of(score), codes
        seen.append(ppl)
        checked += 1
        high += ppl >= PPL_HIGH
        flagged += "odd_words" in codes
    seen.sort()
    return {"checked": checked, "high": high, "flagged": flagged, "skipped": None, "median": round(seen[len(seen) // 2], 2) if seen else None}


def quality_stats(rows: list[dict]) -> dict:
    """Thống kê của stage quality: số bản ghi theo band, số lần xuất hiện của từng reason code."""
    bands: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for row in rows:
        bands[row["quality_band"]] = bands.get(row["quality_band"], 0) + 1
        for c in row["reason_codes"]:
            reasons[c] = reasons.get(c, 0) + 1
    return {"bands": bands, "reasons": reasons}
