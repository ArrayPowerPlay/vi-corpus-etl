"""
Tách công thức toán và khối code ra khỏi văn xuôi (F-03, F-07), dùng chung cho stage language và stage quality.

Các số đo hình dạng (rep_trigram, symbol, digit, độ dài từ...) và bộ nhận diện ngôn ngữ được thiết kế cho văn xuôi: LaTeX
lặp "=", "\\frac{", "+" làm bài toán đúng bị chấm "lặp" / "nhiều ký hiệu", còn code / công thức làm fastText đoán "en".
strip_math_code bỏ các vùng sau (theo thứ tự), phần còn lại là văn xuôi:
1. Khối rào ``` ... ``` (code markdown, có thể nhiều dòng).
2. $$ ... $$, \\[ ... \\], \\( ... \\) (không tham lam, có thể nhiều dòng).
3. $ ... $ chỉ khi nằm trên một dòng, dài <= INLINE_MATH_MAX ký tự và chứa một trong \\ ^ _ { } = (tránh nhầm ký hiệu
   tiền "$5 và $10").
Dấu mở không có dấu đóng (không cân) thì giữ nguyên. Vùng bị bỏ được thay bằng một dấu cách (hoặc xuống dòng nếu vùng đó
nhiều dòng) để không dính hai từ hai bên.
"""

import re

SPANS_VERSION = "1"
INLINE_MATH_MAX = 200
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_DISPLAY = re.compile(r"\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)", re.DOTALL)
_INLINE = re.compile(rf"(?<![\\$])\$(?!\$)([^\n$]{{1,{INLINE_MATH_MAX}}}?)(?<!\\)\$(?!\$)")
_INLINE_MATH_CHARS = frozenset("\\^_{}=")


def _blank(m: re.Match) -> str:
    """Chuỗi thay cho một vùng bị bỏ: xuống dòng nếu vùng nhiều dòng, ngược lại một dấu cách."""
    return "\n" if "\n" in m.group(0) else " "


def strip_math_code(text: str) -> str:
    """Văn xuôi còn lại sau khi bỏ khối code, công thức display và công thức inline (xem docstring module)."""
    text = _FENCE.sub(_blank, text)
    text = _DISPLAY.sub(_blank, text)
    return _INLINE.sub(lambda m: " " if any(c in _INLINE_MATH_CHARS for c in m.group(1)) else m.group(0), text)


def nonspace_len(text: str) -> int:
    """Số ký tự không phải khoảng trắng."""
    return sum(not c.isspace() for c in text)


def math_share(text: str, prose: str) -> float:
    """Tỷ lệ ký tự (không tính khoảng trắng) bị strip_math_code bỏ đi; văn bản rỗng trả 0."""
    total = nonspace_len(text)
    return max(0.0, 1 - nonspace_len(prose) / total) if total else 0.0
