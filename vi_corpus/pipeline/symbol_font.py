"""
Đổi ký tự vùng riêng (private use) của font Symbol về Unicode (J-22 (4), J-23).

PDF / Word dùng font Symbol hay xuất ký tự ở U+F020-U+F0FE: mã trong font Symbol cộng 0xF000 (`+ - = ( )`, chữ Hy Lạp, ký hiệu toán). Văn bản đã
trích mà không đổi thì còn nguyên các ký tự không đọc được ("" thay cho "α", "±"). Bảng dưới là bảng chuẩn Adobe Symbol -> Unicode
(unicode.org, SYMBOL.TXT) cho các mã dùng trong văn xuôi và công thức; mảnh ghép ngoặc / tích phân nhiều dòng (U+F0E6-F0F0, U+F0F3-F0FE ...) không
có nghĩa khi đứng riêng nên bị bỏ.

Mơ hồ với font Wingdings (cùng dải U+F0xx): các điểm gạch đầu dòng Word hay dùng U+F0A7, U+F0D8, U+F0FC khi là Wingdings là hình vuông / mũi tên / dấu tích,
còn khi là Symbol là ♣ / ¬ / mảnh ngoặc. Văn xuôi gần như luôn là điểm gạch đầu dòng nên ba mã này đổi thành "-" (BULLET_CODES).
Ký tự vùng riêng ngoài U+F020-U+F0FE (font toán khác) không đụng tới ở đây; stage quality đếm chúng và đưa đoạn nhiều ký tự loại này xuống band D (private_use_chars).
"""

import re

SYMBOL_VERSION = "1"
BULLET_CODES = (0xA7, 0xD8, 0xFC)  # mã Symbol/Wingdings dùng làm gạch đầu dòng (xem docstring module)

# mã font Symbol -> ký tự Unicode (bảng Adobe Symbol); "" = mảnh ghép không có nghĩa khi đứng riêng, bỏ đi
_SYMBOL = {
    0x20: " ", 0x21: "!", 0x22: "∀", 0x23: "#", 0x24: "∃", 0x25: "%", 0x26: "&", 0x27: "∋", 0x28: "(", 0x29: ")",
    0x2A: "∗", 0x2B: "+", 0x2C: ",", 0x2D: "−", 0x2E: ".", 0x2F: "/", 0x3A: ":", 0x3B: ";", 0x3C: "<", 0x3D: "=",
    0x3E: ">", 0x3F: "?", 0x40: "≅", 0x5B: "[", 0x5C: "∴", 0x5D: "]", 0x5E: "⊥", 0x5F: "_", 0x60: "‾",
    0x7B: "{", 0x7C: "|", 0x7D: "}", 0x7E: "∼",
    0xA0: "€", 0xA1: "ϒ", 0xA2: "′", 0xA3: "≤", 0xA4: "⁄", 0xA5: "∞", 0xA6: "ƒ", 0xA7: "♣",
    0xA8: "♦", 0xA9: "♥", 0xAA: "♠", 0xAB: "↔", 0xAC: "←", 0xAD: "↑", 0xAE: "→", 0xAF: "↓",
    0xB0: "°", 0xB1: "±", 0xB2: "″", 0xB3: "≥", 0xB4: "×", 0xB5: "∝", 0xB6: "∂", 0xB7: "•",
    0xB8: "÷", 0xB9: "≠", 0xBA: "≡", 0xBB: "≈", 0xBC: "…", 0xBD: "|", 0xBE: "—", 0xBF: "↵",
    0xC0: "ℵ", 0xC1: "ℑ", 0xC2: "ℜ", 0xC3: "℘", 0xC4: "⊗", 0xC5: "⊕", 0xC6: "∅", 0xC7: "∩",
    0xC8: "∪", 0xC9: "⊃", 0xCA: "⊇", 0xCB: "⊄", 0xCC: "⊂", 0xCD: "⊆", 0xCE: "∈", 0xCF: "∉",
    0xD0: "∠", 0xD1: "∇", 0xD2: "®", 0xD3: "©", 0xD4: "™", 0xD5: "∏", 0xD6: "√", 0xD7: "⋅",
    0xD8: "¬", 0xD9: "∧", 0xDA: "∨", 0xDB: "⇔", 0xDC: "⇐", 0xDD: "⇑", 0xDE: "⇒", 0xDF: "⇓",
    0xE0: "◊", 0xE1: "〈", 0xE2: "®", 0xE3: "©", 0xE4: "™", 0xE5: "∑", 0xF1: "〉", 0xF2: "∫",
}
# chữ số
_SYMBOL.update({0x30 + d: str(d) for d in range(10)})
# chữ Hy Lạp hoa A-Z (0x41-0x5A) và thường a-z (0x61-0x7A), theo bảng Symbol
_GREEK_UPPER = "ΑΒΧΔΕΦΓΗΙϑΚΛΜΝΟΠΘΡΣΤΥςΩΞΨΖ"
_GREEK_LOWER = "αβχδεφγηιϕκλμνοπθρστυϖωξψζ"
_SYMBOL.update({0x41 + i: ch for i, ch in enumerate(_GREEK_UPPER)})
_SYMBOL.update({0x61 + i: ch for i, ch in enumerate(_GREEK_LOWER)})
# mảnh ghép ngoặc / tích phân nhiều dòng và ký tự mở rộng: bỏ
_SYMBOL.update({code: "" for code in (*range(0xE6, 0xF1), *range(0xF3, 0xFF))})
_SYMBOL.update({code: "-" for code in BULLET_CODES})

PUA_SYMBOL_BASE = 0xF000
SYMBOL_MAP: dict[int, str] = {PUA_SYMBOL_BASE + code: ch for code, ch in _SYMBOL.items()}
_SYMBOL_CHARS = re.compile("[\uf020-\uf0fe]")


def map_symbol_font(text: str) -> tuple[str, int]:
    """
    Đổi mọi ký tự U+F020-U+F0FE về Unicode theo SYMBOL_MAP; mã trong dải mà bảng không có (vd 0x7F ...) bị bỏ.

    Returns:
        (văn bản mới, số ký tự vùng riêng đã xử lý). Văn bản không có ký tự nào trong dải thì trả về nguyên chuỗi, số 0.
    """
    return _SYMBOL_CHARS.subn(lambda m: SYMBOL_MAP.get(ord(m.group()), ""), text)
