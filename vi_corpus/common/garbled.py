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
- single_letter_ratio: tín hiệu thứ hai (chữ bị mất kiểu "đ u"), chỉ để đo (scripts/vjol/check_garbled.py), chưa dùng để quyết.
Ngưỡng là điểm khởi đầu (đo trên 10.000 đoạn VJOL), chỉnh bằng số liệu của scripts/vjol/check_garbled.py.
"""

import re
import unicodedata
from collections.abc import Sequence

# Ngưỡng khởi điểm (J-10 (1), (2)).
GARBLED_RATIO = 0.08
GARBLED_MIN_WORDS = 10
GARBLED_DOC_SHARE = 0.30
# Mã lý do ghi trên trang bị đưa đi OCR vì lỗi font.
REASON_PAGE = "garbled_text_layer"  # chính trang này bị nghi
REASON_DOC = "garbled_doc"  # trang bình thường nhưng bài có >= GARBLED_DOC_SHARE số trang bị nghi

# Dấu thanh (NFD): huyền, sắc, ngã, hỏi, nặng. Dấu mũ / trăng / râu không phải dấu thanh.
_TONES = frozenset("̣̀́̃̉")
_WORD = re.compile(r"[^\W\d_]{2,}")
_LETTERS = re.compile(r"[^\W\d_]+")


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


def plan_garbled(texts: Sequence[str | None], min_ratio: float = GARBLED_RATIO, min_words: int = GARBLED_MIN_WORDS,
                 doc_share: float = GARBLED_DOC_SHARE) -> list[str | None]:
    """
    Quyết định trang nào của một PDF phải OCR lại vì lớp chữ hỏng.

    Args:
        texts:     Text lớp chữ từng trang; None cho trang không phải trang chữ (trang scan đã đi OCR, trang trắng).
                   Trang chữ có text rỗng sau strip cũng không được tính.
        min_ratio: Ngưỡng tỷ lệ từ nhiều dấu thanh của một trang.
        min_words: Số từ tối thiểu để kết luận một trang.
        doc_share: Tỷ lệ trang bị nghi trên số trang có chữ mà từ đó cả bài bị coi là hỏng.

    Returns:
        Danh sách cùng độ dài `texts`: REASON_PAGE cho trang bị nghi, REASON_DOC cho trang còn lại của bài bị nâng lên cấp bài, None
        cho trang không cần OCR lại.
    """
    has_text = [t is not None and bool(t.strip()) for t in texts]
    suspect = [ok and is_garbled(t, min_ratio, min_words) for t, ok in zip(texts, has_text)]
    n_text, n_suspect = sum(has_text), sum(suspect)
    whole_doc = n_text > 0 and n_suspect / n_text >= doc_share
    return [REASON_PAGE if s else REASON_DOC if whole_doc and ok else None for s, ok in zip(suspect, has_text)]
