"""
Làm sạch ký tự Unicode không mã hoá được ra UTF-8 (surrogate lẻ, U+D800–U+DFFF) trước khi ghi checkpoint.

Lỗi gốc (lần chạy full VJOL 2026-10-29): một số PDF có bảng ToUnicode hỏng nên PyMuPDF trả ra "nửa" cặp surrogate
(vd "\\ud835" của chữ toán học). Python giữ được chuỗi đó trong bộ nhớ, nhưng json.dump(..., ensure_ascii=False) ghi ra file
UTF-8 thì ném UnicodeEncodeError và làm cả lượt trích text dừng. Ký tự lẻ được thay bằng U+FFFD (ký tự thay thế) để vẫn
thấy chỗ đó từng có chữ.
"""

import re
from typing import Any

REPLACEMENT = "�"
_SURROGATE = re.compile("[\ud800-\udfff]")


def scrub_surrogates(text: str) -> tuple[str, int]:
    """
    Thay mọi surrogate lẻ trong `text` bằng U+FFFD.

    Returns:
        (text đã sạch, số ký tự đã thay); không có gì để thay thì trả đúng chuỗi cũ.
    """
    return _SURROGATE.subn(REPLACEMENT, text)


def scrub_obj(obj: Any) -> tuple[Any, int]:
    """
    Làm sạch đệ quy mọi chuỗi (giá trị dict, phần tử list / tuple) của một cấu trúc JSON; không sửa `obj` tại chỗ.

    Returns:
        (cấu trúc đã sạch, tổng số ký tự đã thay).
    """
    if isinstance(obj, str):
        return scrub_surrogates(obj)
    if isinstance(obj, dict):
        total = 0
        out = {}
        for key, value in obj.items():
            out[key], n = scrub_obj(value)
            total += n
        return out, total
    if isinstance(obj, (list, tuple)):
        items = [scrub_obj(v) for v in obj]
        return type(obj)(v for v, _ in items), sum(n for _, n in items)
    return obj, 0
