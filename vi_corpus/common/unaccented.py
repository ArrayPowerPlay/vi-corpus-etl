"""
Sinh danh sách "từ không dấu" từ kho chữ sạch (J-25 A), thay cho danh sách viết tay của vi_corpus.common.garbled.UNACCENTED_WORDS.

Ý tưởng: tiếng Việt bị mất dấu từng phần ("cua", "nhung", "duoc" thay cho "của", "những", "được") để lại nhiều từ chức năng viết trần mà chữ đúng gần như KHÔNG BAO GIỜ
viết trần. Với mỗi dạng bỏ dấu (khoá) đếm hai số trên kho chữ sạch: số lần các dạng CÓ dấu cho ra khoá đó ("của", "cữa", "cưa" -> "cua") và số lần chính khoá đó
xuất hiện trần. Giữ khoá khi dạng có dấu đủ thường gặp và dạng trần chỉ chiếm tỷ lệ rất nhỏ. Danh sách viết tay cũ có "doanh" và "cung" ("doanh nghiệp", "cung cấp"
là từ đúng không dấu) nên loại oan 102 đoạn kinh tế; sinh từ dữ liệu thì hai từ đó tự rơi ra vì dạng trần của chúng chiếm đa số.

Hai lớp bảo vệ để bộ dò không gắn cờ nhầm văn bản không phải tiếng Việt: (1) chỉ đếm văn bản có đủ chữ có dấu (MIN_DOC_DIACRITIC), nên trang web viết hoàn toàn không dấu
không làm bẩn số đếm dạng trần; (2) loại các khoá trùng từ chức năng thông dụng của tiếng Anh / Pháp / Tây Ban Nha / Đức (FOREIGN_STOPLIST: "the", "de", "la", "an" ...).
"""

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

_LETTERS = re.compile(r"[^\W\d_]+")
MIN_DOC_LETTERS = 200  # văn bản ngắn hơn không đủ để biết có phải tiếng Việt có dấu
MIN_DOC_DIACRITIC = 0.10  # tỷ lệ chữ cái có dấu tối thiểu của văn bản được đếm (tiếng Việt bình thường ~0,2-0,3)
MIN_ACCENTED = 500  # khoá phải xuất hiện ít nhất ngần này lần ở dạng có dấu (trên mẫu khoảng 300 nghìn văn bản); đổi theo cỡ mẫu
MAX_BARE_SHARE = 0.01  # tỷ lệ dạng trần / (trần + có dấu) tối đa; từ đúng không dấu như "doanh", "cung" có tỷ lệ này rất lớn
TOP_WORDS = 300  # giữ tối đa ngần này khoá (theo số lần dạng có dấu, giảm dần)

# Từ chức năng thông dụng của tiếng Anh / Pháp / Tây Ban Nha / Đức / Bồ Đào Nha trùng khoá bỏ dấu của tiếng Việt: giữ chúng lại thì bài tiếng nước ngoài bị gắn cờ mất dấu.
FOREIGN_STOPLIST = frozenset("""
the a an to of in on at as by or and no so be do go he me my we us up it is if am are was has had not but for you your his her its our out all any can may one two
new old who why how now off yes too per via pre non sub con com con con men man ten tan tin ton son san sun din don dan den ban bin bon
de la le les des du en el los las un una une et est se si ce ci da di do mi lo al ya yo tu su que por con sin sus mis nos fue era
der die das den dem ein eine und ist von mit auf zu im am es er sie wir ihr ich
na ao uma os um no em da do dos das
""".split())


def strip_marks(word: str) -> str:
    """Bỏ mọi dấu của từ (thanh, mũ, trăng, râu) và đổi đ -> d; trả chữ thường."""
    nfd = unicodedata.normalize("NFD", word.lower().replace("đ", "d"))
    return "".join(c for c in nfd if not unicodedata.combining(c))


def diacritic_share(text: str) -> tuple[float, int]:
    """(tỷ lệ chữ cái có dấu trên mọi chữ cái, số chữ cái) của văn bản."""
    letters = accented = 0
    for w in _LETTERS.findall(unicodedata.normalize("NFC", text)):
        letters += len(w)
        accented += sum(1 for c in w if c.lower() == "đ" or any(unicodedata.combining(m) for m in unicodedata.normalize("NFD", c)))
    return (accented / letters if letters else 0.0), letters


def count_forms(texts: Iterable[str], min_diacritic: float = MIN_DOC_DIACRITIC, min_letters: int = MIN_DOC_LETTERS) -> tuple[Counter, Counter, int]:
    """
    Đếm dạng có dấu và dạng trần theo khoá bỏ dấu trên các văn bản đủ chữ có dấu.

    Returns:
        (accented, bare, số văn bản được dùng): accented[khoá] = số lần dạng CÓ dấu cho ra khoá, bare[khoá] = số lần chính khoá xuất hiện trần (chỉ khoá toàn ASCII).
    """
    accented: Counter = Counter()
    bare: Counter = Counter()
    used = 0
    for text in texts:
        share, letters = diacritic_share(text or "")
        if letters < min_letters or share < min_diacritic:
            continue
        used += 1
        for w in _LETTERS.findall(unicodedata.normalize("NFC", text).lower()):
            if len(w) < 2:
                continue
            key = strip_marks(w)
            if not key.isascii() or not key.isalpha():
                continue
            if key == w:
                bare[key] += 1
            else:
                accented[key] += 1
    return accented, bare, used


def select_words(accented: Counter, bare: Counter, min_accented: int = MIN_ACCENTED, max_bare_share: float = MAX_BARE_SHARE,
                 top: int = TOP_WORDS, stoplist: frozenset[str] = FOREIGN_STOPLIST) -> list[tuple[str, int, int]]:
    """
    Chọn các khoá mà chữ đúng gần như không bao giờ viết trần.

    Returns:
        [(khoá, số lần dạng có dấu, số lần dạng trần)] theo số lần dạng có dấu giảm dần, tối đa `top` mục.
    """
    keep = []
    for key, n_acc in accented.items():
        n_bare = bare.get(key, 0)
        if n_acc < min_accented or key in stoplist or len(key) < 2:
            continue
        if n_bare / (n_bare + n_acc) <= max_bare_share:
            keep.append((key, n_acc, n_bare))
    keep.sort(key=lambda t: (-t[1], t[0]))
    return keep[:top]


def build_unaccented(texts: Iterable[str], min_accented: int = MIN_ACCENTED, max_bare_share: float = MAX_BARE_SHARE, top: int = TOP_WORDS,
                     extra_stoplist: Iterable[str] = ()) -> dict:
    """
    Sinh danh sách từ không dấu từ các văn bản (xem đầu file).

    Returns:
        {"docs": số văn bản được dùng, "params": {...}, "words": [khoá, theo thứ tự chữ cái], "counts": {khoá: [số lần có dấu, số lần trần]}}; "words" là thứ
        vi_corpus.common.garbled.load_unaccented đọc.
    """
    accented, bare, used = count_forms(texts)
    chosen = select_words(accented, bare, min_accented, max_bare_share, top, FOREIGN_STOPLIST | frozenset(extra_stoplist))
    return {"docs": used, "params": {"min_accented": min_accented, "max_bare_share": max_bare_share, "top": top,
                                     "min_doc_diacritic": MIN_DOC_DIACRITIC, "min_doc_letters": MIN_DOC_LETTERS},
            "words": sorted(k for k, _, _ in chosen), "counts": {k: [a, b] for k, a, b in chosen}}


def save_unaccented(path: Path, result: dict) -> None:
    """Ghi kết quả build_unaccented ra file JSON (tạo thư mục nếu cần)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
