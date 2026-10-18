"""
Phát hiện chữ hỏng do font PDF (J-10): lớp chữ của PDF toàn chữ Latinh có dấu nhưng sai ("Baỷng, hieọu, ủoọ, ngửụứi").

Khác TCVN3 (vi_corpus.common.pdf_text.is_tcvn3: đổi được sang Unicode), kiểu hỏng này không có bảng chuyển mã đã biết; chữ vẫn
trông như tiếng Việt nên fastText nhận là tiếng Việt và bộ chấm điểm không bắt được. Tín hiệu dùng ở đây: tiếng Việt đúng chỉ có
tối đa MỘT dấu thanh (huyền, sắc, ngã, hỏi, nặng) trên mỗi âm tiết, còn chữ hỏng hay có từ mang từ hai dấu thanh trở lên.

- garbled_ratio: tỷ lệ từ (>= 2 chữ cái) có >= 2 dấu thanh; độc lập ngôn ngữ, không dùng fastText.
- is_garbled: tỷ lệ đó >= GARBLED_RATIO và văn bản đủ GARBLED_MIN_WORDS từ để kết luận.
- plan_garbled: từ text lớp chữ của các trang một PDF, quyết định trang nào phải OCR lại và vì lý do gì. Trang bị nghi thì OCR;
  nếu >= GARBLED_DOC_SHARE số trang có chữ của bài bị nghi thì CẢ BÀI đi OCR (lỗi font thường theo font nên theo cả bài, trang
  ngắn như chú thích hình dễ lọt).
- Ba tín hiệu bổ sung (J-15, J-16) cho các họ chữ hỏng ít dấu thanh mà tín hiệu dấu thanh bỏ sót: case_mix_ratio (chữ hoa chen
  giữa từ viết thường: "xuAáT", "TRoNg", "7K{QJWLQYjJLảLSKiSNKRDKọFF" của chữ lệch mã ASCII), foreign_letter_ratio (chữ cái Latin-1 không có
  trong tiếng Việt: ñ ö ï ø å, họ VNI "Thöïc TRAïNg"; ngưỡng 0,3% và phải có ít nhất 5 chữ cái lạ để tên riêng không kích hoạt, J-18), unknown_syllable_rate (tỷ lệ âm tiết không có trong tập âm tiết tham chiếu
  processed/ocr_bakeoff/syllables.json: bắt thêm họ sai dấu kiểu Windows-1258 "Phỏt triển trờn"; chỉ tính từ Latinh nên trang chữ Hán không
  bị phạt; thiếu file thì bỏ tín hiệu này). page_signals trả tên các tín hiệu vượt ngưỡng của một trang; ngưỡng nằm trong Thresholds.
- single_letter_ratio: tín hiệu thứ hai (chữ bị mất kiểu "đ u"), chỉ để đo (scripts/vjol/check_garbled.py), chưa dùng để quyết.
Ngưỡng chỉnh bằng số liệu của scripts/vjol/check_garbled.py (J-13: trang 5%, cấp bài 30%, cứng ở quality 8%).
"""

import re
import unicodedata
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

# Ngưỡng theo trang (J-10 (1), (2)); J-13: đo trên 2.937 bài thấy 5% bắt thêm kiểu chữ hỏng sát ngưỡng ("Trữớng Ôi hồc" ở 8,1%) với giá
# 39 trang OCR thêm; OCR một trang sạch gần như vô hại, bỏ sót trang hỏng thì chữ rác vào kho.
GARBLED_RATIO = 0.05
# Ngưỡng cứng của stage quality (mã garbled_font, band D) trên đoạn văn: giữ 8% vì loại bỏ nặng tay hơn là OCR lại.
GARBLED_HARD_RATIO = 0.08
GARBLED_MIN_WORDS = 10
GARBLED_DOC_SHARE = 0.30
FOREIGN_LETTERS = frozenset("ñöïøåæÑÖÏØÅÆ")  # chữ cái Latin-1 không có trong tiếng Việt; trong chữ VNI chúng thay cho ư, ị, ờ, ả ...
SIGNALS = ("tone", "case", "foreign", "unknown")  # tên các tín hiệu, theo thứ tự page_signals trả về


class Thresholds(NamedTuple):
    """Ngưỡng của các tín hiệu chữ hỏng theo trang (điểm khởi đầu, chỉnh bằng scripts/vjol/check_garbled.py)."""

    tone: float = GARBLED_RATIO  # tỷ lệ từ >= 2 dấu thanh
    case: float = 0.05  # tỷ lệ từ >= 4 chữ cái có chữ hoa đứng sau chữ thường (J-16)
    foreign: float = 0.003  # tỷ lệ chữ cái thuộc FOREIGN_LETTERS (J-18: hạ từ 0,01 để bắt tiêu đề / đầu trang VNI)
    unknown: float = 0.30  # tỷ lệ từ Latinh không có trong tập âm tiết (J-18: hạ từ 0,40; trang tiếng Anh thường 0,20-0,27, trang hỏng đo được >= 0,30)
    min_words: int = GARBLED_MIN_WORDS  # số từ tối thiểu để xét tín hiệu tone và case
    unknown_min_words: int = 30  # số từ Latinh tối thiểu để xét tín hiệu unknown
    foreign_min_letters: int = 100  # số chữ cái tối thiểu để xét tín hiệu foreign
    foreign_min_count: int = 5  # số chữ cái thuộc FOREIGN_LETTERS tối thiểu (J-18: tên riêng có 1-2 chữ ñ / ö không kích hoạt)


DEFAULT_THRESHOLDS = Thresholds()
# Mã lý do ghi trên trang bị đưa đi OCR vì lỗi font.
REASON_PAGE = "garbled_text_layer"  # chính trang này bị nghi
REASON_DOC = "garbled_doc"  # trang bình thường nhưng bài có >= GARBLED_DOC_SHARE số trang bị nghi

# Dấu thanh (NFD): huyền, sắc, ngã, hỏi, nặng. Dấu mũ / trăng / râu không phải dấu thanh.
_TONES = frozenset("̣̀́̃̉")
_WORD = re.compile(r"[^\W\d_]{2,}")
_LETTERS = re.compile(r"[^\W\d_]+")
_LONG_WORD = re.compile(r"[^\W\d_]{4,}")
_LATIN_WORD = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]+")


def _words(text: str) -> list[str]:
    """Các từ gồm >= 2 chữ cái của văn bản (đã đưa về NFC)."""
    return _WORD.findall(unicodedata.normalize("NFC", text))


def _tone_count(word: str) -> int:
    """Số dấu thanh của một từ (0 nếu toàn chữ ASCII)."""
    if word.isascii():
        return 0
    return sum(c in _TONES for c in unicodedata.normalize("NFD", word))


def garbled_ratio(text: str) -> tuple[float, int]:
    """
    Tỷ lệ từ có >= 2 dấu thanh trong các từ >= 2 chữ cái.

    Returns:
        (tỷ lệ, số từ >= 2 chữ cái); tỷ lệ là 0.0 khi không có từ nào.
    """
    words = _words(text)
    if not words:
        return 0.0, 0
    return sum(_tone_count(w) >= 2 for w in words) / len(words), len(words)


def is_garbled(text: str, min_ratio: float = GARBLED_RATIO, min_words: int = GARBLED_MIN_WORDS) -> bool:
    """True nếu text có >= min_words từ và tỷ lệ từ nhiều dấu thanh >= min_ratio (văn bản ngắn không kết luận, trả False)."""
    ratio, n = garbled_ratio(text)
    return n >= min_words and ratio >= min_ratio


def single_letter_ratio(text: str) -> float:
    """
    Tỷ lệ "từ" một chữ cái trên mọi cụm chữ cái (tín hiệu thứ hai, chữ bị mất kiểu "đ u"); tiếng Việt đúng có ít từ một chữ cái.

    Chỉ để đo trên trang tiếng Việt; chưa dùng để quyết định (J-10 (5)).
    """
    runs = _LETTERS.findall(unicodedata.normalize("NFC", text))
    return sum(len(r) == 1 for r in runs) / len(runs) if runs else 0.0


def case_mix_ratio(text: str) -> tuple[float, int]:
    """
    Tỷ lệ từ >= 4 chữ cái có chữ hoa đứng ngay sau chữ thường ("xuAáT", "TRoNg", "iPhone"); chữ hoa đầu từ hay cả từ viết hoa không tính.

    Returns:
        (tỷ lệ, số từ >= 4 chữ cái); tỷ lệ là 0.0 khi không có từ nào.
    """
    words = _LONG_WORD.findall(unicodedata.normalize("NFC", text))
    if not words:
        return 0.0, 0
    return sum(any(a.islower() and b.isupper() for a, b in zip(w, w[1:])) for w in words) / len(words), len(words)


def foreign_letter_ratio(text: str) -> tuple[float, int, int]:
    """
    Tỷ lệ chữ cái thuộc FOREIGN_LETTERS (ñ ö ï ø å ...) trên mọi chữ cái.

    Returns:
        (tỷ lệ, số chữ cái, số chữ cái lạ); tỷ lệ là 0.0 khi không có chữ cái nào.
    """
    letters = [c for c in unicodedata.normalize("NFC", text) if c.isalpha()]
    if not letters:
        return 0.0, 0, 0
    n_foreign = sum(c in FOREIGN_LETTERS for c in letters)
    return n_foreign / len(letters), len(letters), n_foreign


def load_vocab(path: Path) -> frozenset[str] | None:
    """Đọc tập âm tiết tham chiếu (processed/ocr_bakeoff/syllables.json, khoá "syllables"); None nếu file không tồn tại."""
    path = Path(path)
    if not path.is_file():
        return None
    return frozenset(json.loads(path.read_text(encoding="utf-8"))["syllables"])


def unknown_syllable_rate(text: str, vocab: frozenset[str]) -> tuple[float, int]:
    """
    Tỷ lệ từ Latinh (chữ thường, NFC) không có trong `vocab`; từ có chữ Hán, Kirin... bị bỏ nên trang chữ Hán không bị phạt.

    Returns:
        (tỷ lệ, số từ Latinh); tỷ lệ là 0.0 khi không có từ nào.
    """
    words = [w for w in _LETTERS.findall(unicodedata.normalize("NFC", text).lower()) if _LATIN_WORD.fullmatch(w)]
    if not words:
        return 0.0, 0
    return sum(w not in vocab for w in words) / len(words), len(words)


def page_measures(text: str, vocab: frozenset[str] | None = None) -> dict:
    """
    Số đo thô của bốn tín hiệu cho một trang: {"tone" | "case" | "foreign" | "unknown": (tỷ lệ, số mẫu[, số chữ cái lạ])}; "unknown" là None khi không có vocab.

    Số mẫu là số từ (tone), số từ >= 4 chữ cái (case), số chữ cái (foreign, kèm số chữ cái lạ ở phần tử thứ ba), số từ Latinh (unknown). Tách khỏi page_signals để script đo
    tính một lần rồi thử nhiều ngưỡng.
    """
    return {"tone": garbled_ratio(text), "case": case_mix_ratio(text), "foreign": foreign_letter_ratio(text),
            "unknown": unknown_syllable_rate(text, vocab) if vocab is not None else None}


def signals_from_measures(m: dict, th: Thresholds = DEFAULT_THRESHOLDS) -> list[str]:
    """Các tín hiệu (tên trong SIGNALS) vượt ngưỡng `th`, từ số đo của page_measures."""
    out = []
    if m["tone"][1] >= th.min_words and m["tone"][0] >= th.tone:
        out.append("tone")
    if m["case"][1] >= th.min_words and m["case"][0] >= th.case:
        out.append("case")
    if m["foreign"][1] >= th.foreign_min_letters and m["foreign"][2] >= th.foreign_min_count and m["foreign"][0] >= th.foreign:
        out.append("foreign")
    if m["unknown"] is not None and m["unknown"][1] >= th.unknown_min_words and m["unknown"][0] >= th.unknown:
        out.append("unknown")
    return out


def page_signals(text: str, th: Thresholds = DEFAULT_THRESHOLDS, vocab: frozenset[str] | None = None) -> list[str]:
    """
    Các tín hiệu chữ hỏng vượt ngưỡng của một trang (tên trong SIGNALS); rỗng nghĩa là trang không bị nghi.

    Args:
        text:  Lớp chữ của trang.
        th:    Ngưỡng.
        vocab: Tập âm tiết tham chiếu cho tín hiệu "unknown"; None thì bỏ tín hiệu này.
    """
    return signals_from_measures(page_measures(text, vocab), th)


def plan_from_flags(has_text: Sequence[bool], suspect: Sequence[bool], doc_share: float = GARBLED_DOC_SHARE) -> list[str | None]:
    """
    Từ cờ "có lớp chữ" và "bị nghi" của từng trang, quyết định lý do OCR lại (xem plan_garbled): trang nghi -> REASON_PAGE; nếu
    >= doc_share số trang có chữ bị nghi thì mọi trang có chữ còn lại -> REASON_DOC; còn lại None.
    """
    n_text, n_suspect = sum(has_text), sum(suspect)
    whole_doc = n_text > 0 and n_suspect / n_text >= doc_share
    return [REASON_PAGE if s else REASON_DOC if whole_doc and ok else None for s, ok in zip(suspect, has_text)]


def plan_garbled(texts: Sequence[str | None], th: Thresholds = DEFAULT_THRESHOLDS, doc_share: float = GARBLED_DOC_SHARE,
                 vocab: frozenset[str] | None = None) -> list[str | None]:
    """
    Quyết định trang nào của một PDF phải OCR lại vì lớp chữ hỏng.

    Args:
        texts:     Text lớp chữ từng trang; None cho trang không phải trang chữ (trang scan đã đi OCR, trang trắng).
                   Trang chữ có text rỗng sau strip cũng không được tính.
        th:        Ngưỡng các tín hiệu (page_signals); trang bị nghi nếu có ít nhất một tín hiệu vượt ngưỡng.
        doc_share: Tỷ lệ trang bị nghi trên số trang có chữ mà từ đó cả bài bị coi là hỏng.
        vocab:     Tập âm tiết cho tín hiệu "unknown" (None = bỏ tín hiệu này).

    Returns:
        Danh sách cùng độ dài `texts`: REASON_PAGE cho trang bị nghi, REASON_DOC cho trang còn lại của bài bị nâng lên cấp bài, None
        cho trang không cần OCR lại.
    """
    has_text = [t is not None and bool(t.strip()) for t in texts]
    suspect = [ok and bool(page_signals(t, th, vocab)) for t, ok in zip(texts, has_text)]
    return plan_from_flags(has_text, suspect, doc_share)
