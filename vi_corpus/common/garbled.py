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
- Tín hiệu thứ năm "nodiac" (J-22 (2), J-23): mất dấu tiếng Việt từng phần (nhiều từ chức năng không dấu như "cua", "nhung", "duoc"
  trong khi chữ có dấu khác vẫn còn); lớp chữ kiểu này là chữ Latinh hợp lệ nên bốn tín hiệu trên không bắt. Tập từ chức năng nằm ở
  UNACCENTED_WORDS; trang bị nghi thì đi OCR VLM như lỗi font. Không khôi phục dấu bằng mô hình (dễ bịa chữ).
- reversed_gain (J-22 (1), J-23): chữ đảo ngược (trang xoay 180 độ): tỷ lệ âm tiết hợp lệ của từng từ viết ngược trừ tỷ lệ của văn bản gốc.
  Chỉ dùng ở stage quality (mã reversed_text), chưa là tín hiệu OCR lại vì nguồn gốc lỗi (lớp chữ nhúng hay OCR VLM) chưa xác định.
- Tín hiệu thứ sáu "hyphen" (J-25 B): font đời cũ mất ư / ơ thành dấu gạch nối ("đ-ợc", "ng-ời", "th- viện"); trang nghi đi OCR VLM
  như lỗi font. Cùng bộ dò dùng ở stage quality (mã lost_letters) làm lưới an toàn.
- Danh sách từ không dấu (nodiac) lấy từ file sinh bằng scripts/build_unaccented.py (J-25 A): từ mà chữ đúng gần như không bao giờ viết
  trần, lập từ kho chữ sạch thay cho danh sách viết tay (vốn có "doanh", "cung" là từ đúng không dấu nên loại oan). Thiếu file thì dùng
  UNACCENTED_WORDS (danh sách tối thiểu đã bỏ hai từ đó).
- single_letter_ratio: tín hiệu thứ hai (chữ bị mất kiểu "đ u"), chỉ để đo (scripts/vjol/check_garbled.py), chưa dùng để quyết.
Ngưỡng chỉnh bằng số liệu của scripts/vjol/check_garbled.py (J-13: trang 5%, cấp bài 30%, cứng ở quality 8%).
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter
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
SIGNALS = ("tone", "case", "foreign", "unknown", "nodiac", "hyphen")  # tên các tín hiệu, theo thứ tự page_signals trả về


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
    nodiac: float = 0.02  # tỷ lệ từ Latinh thuộc UNACCENTED_WORDS (J-23: tiếng Việt có dấu đúng gần như 0, mất dấu hoàn toàn khoảng 5-6%)
    nodiac_min_words: int = 50  # số từ Latinh tối thiểu để xét tín hiệu nodiac
    hyphen: float = 0.004  # tỷ lệ từ Latinh có dạng "đ-ợc" / "th- viện" (J-25 B, điểm khởi đầu chưa đo trên bài thật)
    hyphen_min_count: int = 4  # số chỗ gạch nối thay ư / ơ tối thiểu (một vài "x-ray", "N-acetyl" không kích hoạt)
    hyphen_min_words: int = 50  # số từ Latinh tối thiểu để xét tín hiệu hyphen


DEFAULT_THRESHOLDS = Thresholds()
# Mã lý do ghi trên trang bị đưa đi OCR vì lỗi font.
REASON_PAGE = "garbled_text_layer"  # chính trang này bị nghi
REASON_DOC = "garbled_doc"  # trang bình thường nhưng bài có >= GARBLED_DOC_SHARE số trang bị nghi

# Từ chức năng / thông dụng của tiếng Việt viết KHÔNG dấu ("của" -> "cua") mà tiếng Anh gần như không dùng; chữ có dấu đúng thì hầu như không
# có từ nào trong tập này (J-23, điểm khởi đầu chưa đo trên bài thật). Không đưa tên riêng hay tên đệm (Thanh, Hien, Cong, Quoc, Gia ...) vào
# để danh mục tham khảo "Tran Thanh Hien" không kích hoạt.
# J-25 A: đã bỏ "doanh" ("doanh nghiệp") và "cung" ("cung cấp") vì đó là từ đúng không dấu, làm loại oan 102 đoạn kinh tế của vjol_10k_v3.
UNACCENTED_WORDS = frozenset("cua nhung duoc khong nguoi nhieu mot voi hoac nghien cuu viec hoat thuc trien".split())

# J-28: từ chức năng tiếng Anh mà tiếng Việt viết trần gần như không có ("the", "in", "to", "can", "may", "be", "that" bị loại vì trùng "thế", "ín", "tô", "căn",
# "máy", "bé", "thật"). Văn bản có >= ENGLISH_SHARE từ loại này là tiếng Anh, không đo mất dấu (điểm khởi đầu: tiếng Anh thường 10-15%, chưa đo trên bài thật).
ENGLISH_WORDS = frozenset("of and is are was were with for from which by have has been not or as this these those their also will".split())
ENGLISH_SHARE = 0.04
# J-28: mỗi từ trong danh sách chỉ tính tối đa UNACCENTED_CAP lần mỗi đoạn. Chữ trần thật mất dấu có hàng chục từ khác nhau; còn thuật ngữ đúng như "dở dang"
# (kế toán), "cao su", "axit lactic toan" lặp một từ hàng chục lần thì không thể làm đoạn bị loại.
UNACCENTED_CAP = 3
# Đường dẫn, email, DOI và "slug" nối bằng >= 3 dấu gạch ("Trung-Quoc-dang-tranh-chap" của URL bị ngắt dòng): bỏ khỏi phép đo mất dấu.
_URL_SPAN = re.compile(r"(?:https?://|www\.)\S+|\S+@\S+\.\S+|\bdoi:\s*\S+|\w+(?:-\w+){3,}", re.IGNORECASE)

# Font mất ư / ơ: phụ âm đầu + "-" + nguyên âm có móc (đ-ợc, ng-ời, tr-ờng) hoặc + khoảng trắng (th- viện, h- ớng). Chỉ nhận phụ âm đầu thật của
# tiếng Việt đứng đầu từ để "e-mail", "x-ray", "k-means", "co-operate" không khớp; "N-acetyl" (chữ đơn + nguyên âm thường) không khớp vì nhánh
# nguyên âm chỉ nhận ơ ờ ớ ở ỡ ợ ư ừ ứ ử ữ ự.
# J-28: nhánh "khoảng trắng + chữ thường" chỉ nhận phụ âm đơn viết THƯỜNG ("c- dân", "h- ớng"), vì "C- section", "N- TiO2" (chữ hoa đơn + gạch nối) là tiếng Anh / hoá học đúng.
_HYPHEN_MULTI = r"(?:ngh|ng|nh|ch|kh|th|tr|gh|gi|ph|qu)"
_HYPHEN_SINGLE = r"[bcdđghklmnpqrstvx]"
_HYPHEN_LOST = re.compile(
    rf"(?<![\w-])(?:(?:{_HYPHEN_MULTI}|{_HYPHEN_SINGLE})-(?=[ơờớởỡợưừứửữự])|{_HYPHEN_MULTI}-(?=\s+[a-zà-ỹ])|(?-i:{_HYPHEN_SINGLE})-(?=\s+[a-zà-ỹ]))", re.IGNORECASE)

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


def latin_tokens(text: str) -> list[str]:
    """Các cụm chữ cái Latinh (NFC, giữ nguyên chữ hoa / thường) của văn bản; cụm có chữ Hán, Kirin... bị bỏ."""
    return [w for w in _LETTERS.findall(unicodedata.normalize("NFC", text)) if _LATIN_WORD.fullmatch(w)]


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


def load_unaccented(path: Path) -> frozenset[str] | None:
    """Đọc danh sách từ không dấu do scripts/build_unaccented.py sinh (khoá "words"); None nếu file không tồn tại."""
    path = Path(path)
    if not path.is_file():
        return None
    return frozenset(json.loads(path.read_text(encoding="utf-8"))["words"])


def unaccented_tag(words: frozenset[str] | None) -> str:
    """Dấu nhận dạng ngắn (băm nội dung) của danh sách từ không dấu, ghi vào extract_version để đổi danh sách thì checkpoint cũ bị làm lại; "" nếu None."""
    return hashlib.sha1("\n".join(sorted(words)).encode("utf-8")).hexdigest()[:6] if words else ""


def unaccented_ratio(text: str, words: frozenset[str] | None = None) -> tuple[float, int]:
    """
    Tỷ lệ từ Latinh (chữ thường, NFC) thuộc `words` (mặc định UNACCENTED_WORDS): tiếng Việt bị mất dấu ("cua", "nhung", "duoc" thay cho "của", "những", "được").

    J-28: đường dẫn / email / DOI được bỏ trước khi đếm (tên bài trong URL báo "Trung-Quoc-dang-tranh-chap" không phải chữ mất dấu), và văn bản tiếng Anh
    (tỷ lệ từ chức năng tiếng Anh >= ENGLISH_SHARE) trả về tỷ lệ 0 vì "day", "mat", "hon" ... trùng chữ trần của tiếng Việt nhưng là tiếng Anh đúng. Trên
    vjol_10k_v4a hai nguyên nhân này làm loại oan khoảng hai phần ba trong 476 đoạn bị mã lost_diacritics.

    Returns:
        (tỷ lệ, số từ Latinh); tỷ lệ là 0.0 khi không có từ nào hoặc văn bản là tiếng Anh. Số từ Latinh tính sau khi bỏ đường dẫn.
    """
    lexicon = UNACCENTED_WORDS if words is None else words
    text = _URL_SPAN.sub(" ", unicodedata.normalize("NFC", text))
    toks = [w for w in _LETTERS.findall(text.lower()) if _LATIN_WORD.fullmatch(w)]
    if not toks:
        return 0.0, 0
    if sum(w in ENGLISH_WORDS for w in toks) / len(toks) >= ENGLISH_SHARE:
        return 0.0, len(toks)
    hits = Counter(w for w in toks if w in lexicon)
    return sum(min(c, UNACCENTED_CAP) for c in hits.values()) / len(toks), len(toks)


def hyphen_lost(text: str) -> tuple[float, int, int]:
    """
    Đo chỗ font mất ư / ơ thành dấu gạch nối ("đ-ợc", "ng-ời", "th- viện"; J-25 B).

    Returns:
        (tỷ lệ chỗ lỗi trên số từ Latinh, số từ Latinh, số chỗ lỗi); (0.0, 0, 0) khi không có từ nào.
    """
    nfc = unicodedata.normalize("NFC", text)
    n = len(latin_tokens(nfc))
    if not n:
        return 0.0, 0, 0
    hits = len(_HYPHEN_LOST.findall(nfc))
    return hits / n, n, hits


def reversed_gain(text: str, vocab: frozenset[str], min_len: int = 3) -> tuple[float, float, int]:
    """
    Đo chữ đảo ngược (trang xoay 180 độ): so tỷ lệ từ nằm trong tập âm tiết `vocab` của văn bản với tỷ lệ của từng từ viết NGƯỢC.

    Chỉ xét từ Latinh >= min_len chữ cái (từ ngắn dễ là đối xứng / trùng hợp). Văn bản đúng chiều có tỷ lệ gốc cao và tỷ lệ ngược thấp
    (độ lợi âm); văn bản bị đảo thì ngược lại.

    Returns:
        (độ lợi = tỷ lệ ngược - tỷ lệ gốc, tỷ lệ ngược, số từ xét); (0.0, 0.0, 0) khi không có từ nào.
    """
    words = [w for w in _LETTERS.findall(unicodedata.normalize("NFC", text).lower()) if len(w) >= min_len and _LATIN_WORD.fullmatch(w)]
    if not words:
        return 0.0, 0.0, 0
    fwd = sum(w in vocab for w in words) / len(words)
    rev = sum(w[::-1] in vocab for w in words) / len(words)
    return rev - fwd, rev, len(words)


def page_measures(text: str, vocab: frozenset[str] | None = None, unaccented: frozenset[str] | None = None) -> dict:
    """
    Số đo thô của sáu tín hiệu cho một trang: {"tone" | "case" | "foreign" | "unknown" | "nodiac" | "hyphen": (tỷ lệ, số mẫu[, số chữ cái lạ / số chỗ lỗi])}; "unknown" là None khi không có vocab.

    Số mẫu là số từ (tone), số từ >= 4 chữ cái (case), số chữ cái (foreign, kèm số chữ cái lạ ở phần tử thứ ba), số từ Latinh (unknown, nodiac,
    hyphen; hyphen kèm số chỗ lỗi ở phần tử thứ ba). Tách khỏi page_signals để script đo tính một lần rồi thử nhiều ngưỡng.
    `unaccented`: danh sách từ không dấu (None = UNACCENTED_WORDS).
    """
    return {"tone": garbled_ratio(text), "case": case_mix_ratio(text), "foreign": foreign_letter_ratio(text),
            "unknown": unknown_syllable_rate(text, vocab) if vocab is not None else None, "nodiac": unaccented_ratio(text, unaccented),
            "hyphen": hyphen_lost(text)}


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
    if m["nodiac"][1] >= th.nodiac_min_words and m["nodiac"][0] >= th.nodiac:
        out.append("nodiac")
    if m["hyphen"][1] >= th.hyphen_min_words and m["hyphen"][2] >= th.hyphen_min_count and m["hyphen"][0] >= th.hyphen:
        out.append("hyphen")
    return out


def page_signals(text: str, th: Thresholds = DEFAULT_THRESHOLDS, vocab: frozenset[str] | None = None,
                 unaccented: frozenset[str] | None = None) -> list[str]:
    """
    Các tín hiệu chữ hỏng vượt ngưỡng của một trang (tên trong SIGNALS); rỗng nghĩa là trang không bị nghi.

    Args:
        text:       Lớp chữ của trang.
        th:         Ngưỡng.
        vocab:      Tập âm tiết tham chiếu cho tín hiệu "unknown"; None thì bỏ tín hiệu này.
        unaccented: Danh sách từ không dấu cho tín hiệu "nodiac" (None = UNACCENTED_WORDS).
    """
    return signals_from_measures(page_measures(text, vocab, unaccented), th)


def plan_from_flags(has_text: Sequence[bool], suspect: Sequence[bool], doc_share: float = GARBLED_DOC_SHARE) -> list[str | None]:
    """
    Từ cờ "có lớp chữ" và "bị nghi" của từng trang, quyết định lý do OCR lại (xem plan_garbled): trang nghi -> REASON_PAGE; nếu
    >= doc_share số trang có chữ bị nghi thì mọi trang có chữ còn lại -> REASON_DOC; còn lại None.
    """
    n_text, n_suspect = sum(has_text), sum(suspect)
    whole_doc = n_text > 0 and n_suspect / n_text >= doc_share
    return [REASON_PAGE if s else REASON_DOC if whole_doc and ok else None for s, ok in zip(suspect, has_text)]


def plan_garbled(texts: Sequence[str | None], th: Thresholds = DEFAULT_THRESHOLDS, doc_share: float = GARBLED_DOC_SHARE,
                 vocab: frozenset[str] | None = None, unaccented: frozenset[str] | None = None) -> list[str | None]:
    """
    Quyết định trang nào của một PDF phải OCR lại vì lớp chữ hỏng.

    Args:
        texts:      Text lớp chữ từng trang; None cho trang không phải trang chữ (trang scan đã đi OCR, trang trắng).
                    Trang chữ có text rỗng sau strip cũng không được tính.
        th:         Ngưỡng các tín hiệu (page_signals); trang bị nghi nếu có ít nhất một tín hiệu vượt ngưỡng.
        doc_share:  Tỷ lệ trang bị nghi trên số trang có chữ mà từ đó cả bài bị coi là hỏng.
        vocab:      Tập âm tiết cho tín hiệu "unknown" (None = bỏ tín hiệu này).
        unaccented: Danh sách từ không dấu cho tín hiệu "nodiac" (None = UNACCENTED_WORDS).

    Returns:
        Danh sách cùng độ dài `texts`: REASON_PAGE cho trang bị nghi, REASON_DOC cho trang còn lại của bài bị nâng lên cấp bài, None
        cho trang không cần OCR lại.
    """
    has_text = [t is not None and bool(t.strip()) for t in texts]
    suspect = [ok and bool(page_signals(t, th, vocab, unaccented)) for t, ok in zip(texts, has_text)]
    return plan_from_flags(has_text, suspect, doc_share)
