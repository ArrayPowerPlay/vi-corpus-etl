"""
Số đo so sánh đầu ra OCR với đáp án (F-11): khoảng cách sửa (ký tự / từ), CER, CER bỏ dấu, WER, F1 túi từ, đếm ký tự đặc
biệt, phát hiện vòng lặp 4-gram, bootstrap cặp theo trang.

Khoảng cách Levenshtein dùng thuật toán song song bit của Myers / Hyyrö (số nguyên Python làm vector bit dài tuỳ ý),
không cần thư viện C: trang ~3.000 ký tự mất vài mili giây. Hàm nhận mọi dãy phần tử băm được (chuỗi ký tự, list từ).
"""

import random
import re
import unicodedata
from collections import Counter
from collections.abc import Hashable, Sequence

SPECIAL_CHARS = "²³“”‘’–—…°"  # ký tự VietOCR không xuất được (F-06 ý 5, F-11)
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def levenshtein(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Khoảng cách sửa (chèn / xoá / thay, mỗi phép 1) giữa hai dãy."""
    if len(a) < len(b):
        a, b = b, a
    m = len(b)
    if m == 0:
        return len(a)
    peq: dict = {}
    for i, c in enumerate(b):
        peq[c] = peq.get(c, 0) | (1 << i)
    mask = (1 << m) - 1
    high = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for c in a:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = ((((eq & pv) + pv) & mask) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return score


def normalize(text: str) -> str:
    """Chuẩn hoá trước khi so: NFC, mọi khoảng trắng (kể cả xuống dòng) thành một dấu cách, bỏ khoảng trắng hai đầu."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text or "")).strip()


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt (NFD rồi bỏ dấu kết hợp; đ -> d) để tách lỗi dấu khỏi lỗi chữ."""
    nfd = unicodedata.normalize("NFD", text)
    out = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", out.replace("đ", "d").replace("Đ", "D"))


def words(text: str) -> list[str]:
    """Các từ (tách theo khoảng trắng) của văn bản đã chuẩn hoá."""
    return normalize(text).split()


def bow_f1(hyp: list[str], ref: list[str]) -> float:
    """F1 túi từ (không xét thứ tự): cao mà WER cũng cao nghĩa là đúng chữ nhưng sai thứ tự đọc."""
    if not hyp and not ref:
        return 1.0
    common = sum((Counter(hyp) & Counter(ref)).values())
    if common == 0:
        return 0.0
    p, r = common / len(hyp), common / len(ref)
    return 2 * p * r / (p + r)


def special_counts(text: str) -> dict[str, int]:
    """Số lần xuất hiện của từng ký tự trong SPECIAL_CHARS."""
    return {c: text.count(c) for c in SPECIAL_CHARS if c in text}


def max_ngram_repeat(text: str, n: int = 4) -> int:
    """Số lần lặp nhiều nhất của một n-gram từ (đo vòng lặp do mô hình sinh)."""
    ws = words(text)
    grams = Counter(zip(*(ws[i:] for i in range(n))))
    return max(grams.values(), default=0)


def letter_words(text: str) -> list[str]:
    """Các âm tiết / từ chỉ gồm chữ cái, chữ thường (đo tỷ lệ âm tiết lạ)."""
    return [w.lower() for w in _WORD.findall(unicodedata.normalize("NFC", text or ""))]


def page_scores(hyp: str, ref: str) -> dict:
    """
    Số đo của một trang có đáp án.

    Returns:
        {"ref_chars", "char_dist", "char_dist_nodiac", "ref_words", "word_dist", "bow_f1", "hyp_chars", "len_ratio",
         "special_ref", "special_hit", "max4"}; special_hit[c] = min(số lần c trong hyp, trong ref).
    """
    h, r = normalize(hyp), normalize(ref)
    hw, rw = h.split(), r.split()
    sref, shyp = special_counts(r), special_counts(h)
    return {
        "ref_chars": len(r), "hyp_chars": len(h),
        "char_dist": levenshtein(h, r),
        "char_dist_nodiac": levenshtein(strip_diacritics(h), strip_diacritics(r)),
        "ref_words": len(rw), "word_dist": levenshtein(hw, rw),
        "bow_f1": bow_f1(hw, rw),
        "len_ratio": len(h) / len(r) if r else (1.0 if not h else float("inf")),
        "special_ref": sref, "special_hit": {c: min(n, shyp.get(c, 0)) for c, n in sref.items()},
        "max4": max_ngram_repeat(h),
        "ref_max4": max_ngram_repeat(r),
    }


def micro_rate(num: Sequence[float], den: Sequence[float]) -> float:
    """Tỷ lệ gộp (tổng tử / tổng mẫu); mẫu bằng 0 trả 0."""
    total = sum(den)
    return sum(num) / total if total else 0.0


def paired_bootstrap(num_a: Sequence[float], num_b: Sequence[float], den: Sequence[float], iters: int = 2000,
                     seed: int = 0) -> tuple[float, float, float]:
    """
    Khoảng tin cậy 95% của hiệu tỷ lệ gộp (A - B) khi lấy mẫu lại các trang có hoàn lại, cùng một mẫu cho hai engine.

    Args:
        num_a, num_b: Tử số theo trang của engine A, B (vd số lỗi ký tự).
        den:          Mẫu số theo trang dùng chung (vd số ký tự đáp án).

    Returns:
        (hiệu điểm, cận dưới 2,5%, cận trên 97,5%).
    """
    n = len(den)
    point = micro_rate(num_a, den) - micro_rate(num_b, den)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(den[i] for i in idx) or 1
        diffs.append((sum(num_a[i] for i in idx) - sum(num_b[i] for i in idx)) / d)
    diffs.sort()
    return point, diffs[int(0.025 * iters)], diffs[min(iters - 1, int(0.975 * iters))]
