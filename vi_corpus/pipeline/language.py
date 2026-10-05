"""
Stage 4 (language id): nhận diện tiếng Việt / tiếng Anh bằng heuristic không cần mô hình ngoài.

Hai tín hiệu: (1) tỷ lệ chữ cái có dấu đặc trưng tiếng Việt (ă â đ ê ô ơ ư và các dấu thanh), (2) tỷ lệ từ chức năng
(stopword) tiếng Việt / tiếng Anh. Nhãn: "vi", "en", hoặc "other" (không đủ tín hiệu: mã, bảng số, ngôn ngữ khác,
tiếng Việt không dấu). Với 10.000 mẫu kiểm tra pipeline thì đủ; khi chạy quy mô lớn nên thay bằng fastText lid.176
(chỉ cần thay hàm detect_language, các stage sau chỉ đọc cột language / lang_score).
"""

import re

_VI_CHARS = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
_VI_STOP = frozenset("và của là có không được những một các trong cho với này để người đã khi từ ra như cũng nhiều "
                     "theo về lại rất còn hay bị đến sẽ họ tôi chúng".split())
_EN_STOP = frozenset("the of and to in is that for it with as was on are by be this from or an at which not have "
                     "has had were been their its but they his her you".split())
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

MAX_CHARS = 20_000  # chỉ xét phần đầu văn bản: đủ để nhận diện và giữ chi phí cố định với văn bản rất dài


def detect_language(text: str) -> tuple[str, float]:
    """
    Nhận diện ngôn ngữ của một văn bản.

    Returns:
        (nhãn "vi" | "en" | "other", điểm tin cậy 0..1). Văn bản không có chữ cái trả ("other", 0.0).
    """
    sample = text[:MAX_CHARS].lower()
    words = _WORD.findall(sample)
    letters = sum(len(w) for w in words)
    if not words or letters == 0:
        return "other", 0.0
    diacritic = sum(ch in _VI_CHARS for w in words for ch in w) / letters
    vi_stop = sum(w in _VI_STOP for w in words) / len(words)
    en_stop = sum(w in _EN_STOP for w in words) / len(words)
    if diacritic >= 0.08 or (vi_stop >= 0.08 and vi_stop > en_stop):
        return "vi", round(min(1.0, 0.5 * min(1.0, diacritic / 0.2) + 0.5 * min(1.0, vi_stop / 0.15)), 3)
    if en_stop >= 0.12 and diacritic < 0.03:
        return "en", round(min(1.0, en_stop / 0.25), 3)
    return "other", round(max(vi_stop, en_stop), 3)


def annotate_language(rows: list[dict]) -> list[dict]:
    """Điền cột language và lang_score cho mọi bản ghi (sửa tại chỗ, trả lại chính list đó). Hàm thuần, chạy song song được."""
    for row in rows:
        row["language"], row["lang_score"] = detect_language(row["text"] or "")
    return rows


def language_stats(rows: list[dict]) -> dict:
    """Thống kê của stage language: số bản ghi theo từng nhãn ngôn ngữ."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["language"]] = counts.get(row["language"], 0) + 1
    return {"counts": counts}


def language_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Chạy annotate_language rồi trả về (rows, language_stats(rows))."""
    return annotate_language(rows), language_stats(rows)
