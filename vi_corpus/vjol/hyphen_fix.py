"""
Khôi phục chữ "ư" cho bài VJOL bị font đời cũ vẽ "ư" thành dấu gạch nối (J-33 A): "tr-ờng", "ng-ời", "đ-ợc", "Nh-ng", "t- liệu".

Căn cứ J-32: OCR VLM KHÔNG sửa được lỗi này (495 / 699 trang có tín hiệu hyphen đã qua OCR vẫn còn >= 4 chỗ), vì ảnh trang của
các PDF không nhúng font cũng vẽ dấu "-" thay cho "ư". Quy tắc cố định theo dữ liệu: "-" nằm trong chữ luôn là "ư". Chạy lúc sinh
bản ghi (vi_corpus.vjol.clean.clean_paper_pages), nên đổi quy tắc không phải trích lại PDF.

An toàn nhờ ba lớp kiểm:
1. Chữ đứng trước "-" phải là phụ âm đầu thật của tiếng Việt (ngh, ng, nh, ch, kh, th, tr, gh, gi, ph, qu hoặc một phụ âm đơn), nên "x-ray",
   "e-mail", "Nam-Á", "kinh tế-xã hội" không bao giờ bị xét.
2. Kết quả thay "-" bằng "ư" phải là âm tiết thật trong tập âm tiết tham chiếu (processed/ocr_bakeoff/syllables.json); không thì giữ nguyên.
3. Cả bài phải có ít nhất MIN_DOC_HITS chỗ thoả hai điều trên mới sửa (font hỏng lặp hàng chục chỗ mỗi trang; một hai chỗ lẻ là gạch nối thật).

Hai dạng lỗi:
- liền chữ: "tr-ờng" -> "trường", "Nh-ng" -> "Nhưng", "D-ỚI" -> "DƯỚI";
- có khoảng trắng sau dấu: "t- liệu" -> "tư liệu" (giữ khoảng trắng) và "h- ớng" -> "hướng" (bỏ khoảng trắng, vì "ư" + "ơ" là một nguyên âm đôi "ươ").
"""

import re
import unicodedata

# Phụ âm đầu của tiếng Việt đứng trước "ư" bị mất. Chữ đơn dạng viết hoa ("N- TiO2", "C- section") là tiếng Anh / hoá học nên chỉ nhận chữ đơn viết thường trong dạng có khoảng trắng (J-28).
_MULTI = ("ngh", "ng", "nh", "ch", "kh", "th", "tr", "gh", "gi", "ph", "qu")
_SINGLE = frozenset("bcdđghklmnpqrstvx")
_LETTERS = r"[^\W\d_]"
# "tr-ờng": đầu từ (không dính chữ / gạch nối trước), phụ âm đầu tối đa 3 chữ, "-", phần còn lại của từ (không dính chữ / gạch nối sau).
_GLUED = re.compile(rf"(?<![\w-])({_LETTERS}{{1,3}})-({_LETTERS}+)(?![\w-])")
# "t- liệu", "h- ớng": phụ âm đầu, "-", đúng một khoảng trắng, từ kế tiếp.
_SPACED = re.compile(rf"(?<![\w-])({_LETTERS}{{1,3}})- ({_LETTERS}+)(?![\w-])")
_HORN_O = frozenset("ơờớởỡợ")  # nguyên âm đứng sau "ư" tạo nguyên âm đôi "ươ": dạng có khoảng trắng phải bỏ khoảng trắng
MIN_DOC_HITS = 3  # cả bài cần ít nhất ngần ấy chỗ sửa được thì mới sửa (tránh sửa nhầm gạch nối thật lẻ tẻ)


def _is_initial(lead: str, spaced: bool) -> bool:
    """True nếu `lead` là phụ âm đầu hợp lệ; dạng có khoảng trắng chỉ nhận chữ đơn viết thường (J-28)."""
    low = lead.lower()
    if len(low) == 1:
        return low in _SINGLE and (not spaced or lead == low)
    return low in _MULTI


def _upper_u(tail: str) -> str:
    """"Ư" nếu phần còn lại của từ toàn chữ hoa (từ viết hoa cả chữ: "D-ỚI"), ngược lại "ư"."""
    letters = [c for c in tail if c.isalpha()]
    return "Ư" if letters and all(c.isupper() for c in letters) else "ư"


def _restore_glued(m: re.Match, vocab: frozenset[str]) -> str | None:
    """Từ sửa được cho một chỗ "tr-ờng" (None nếu không thoả quy tắc)."""
    lead, tail = m.group(1), m.group(2)
    if not _is_initial(lead, spaced=False):
        return None
    fixed = lead + _upper_u(tail) + tail
    return fixed if unicodedata.normalize("NFC", fixed).lower() in vocab else None


def _restore_spaced(m: re.Match, vocab: frozenset[str]) -> str | None:
    """
    Từ sửa được cho một chỗ "t- liệu" / "h- ớng" (None nếu không thoả quy tắc): nguyên âm đứng sau thuộc _HORN_O thì ghép liền thành một âm tiết ("hướng"),
    còn lại giữ khoảng trắng và cả hai âm tiết "tư", "liệu" đều phải có trong tập tham chiếu.
    """
    lead, tail = m.group(1), m.group(2)
    if not _is_initial(lead, spaced=True):
        return None
    u = _upper_u(tail)
    if tail[0].lower() in _HORN_O:
        fixed = lead + u + tail
        return fixed if unicodedata.normalize("NFC", fixed).lower() in vocab else None
    head = lead + u
    ok = unicodedata.normalize("NFC", head).lower() in vocab and unicodedata.normalize("NFC", tail).lower() in vocab
    return f"{head} {tail}" if ok else None


def restore_u_in_text(text: str, vocab: frozenset[str]) -> tuple[str, int]:
    """
    Sửa mọi chỗ "-" thay "ư" trong một đoạn văn bản (không áp ngưỡng cả bài, xem restore_u_in_pages).

    Args:
        text:  Văn bản (đã NFC hoặc chưa; kết quả đã NFC).
        vocab: Tập âm tiết thật (chữ thường, NFC).

    Returns:
        (văn bản đã sửa, số chỗ đã sửa).
    """
    text = unicodedata.normalize("NFC", text)
    n = 0

    def repl(restore):
        """Hàm thay cho re.sub: gọi `restore`, đếm số chỗ sửa được, không sửa được thì giữ nguyên."""
        def inner(m: re.Match) -> str:
            """Thay một chỗ khớp."""
            nonlocal n
            fixed = restore(m, vocab)
            if fixed is None:
                return m.group(0)
            n += 1
            return fixed
        return inner

    text = _SPACED.sub(repl(_restore_spaced), text)
    text = _GLUED.sub(repl(_restore_glued), text)
    return text, n


def restore_u_in_pages(pages: list[str], vocab: frozenset[str] | None, min_hits: int = MIN_DOC_HITS) -> tuple[list[str], int]:
    """
    Khôi phục "ư" trong các trang của MỘT bài: chỉ sửa khi cả bài có ít nhất `min_hits` chỗ sửa được (font hỏng lặp nhiều, gạch nối thật thì lẻ tẻ).

    Args:
        pages:    Text từng trang.
        vocab:    Tập âm tiết thật; None thì không làm gì (thiếu syllables.json).
        min_hits: Ngưỡng số chỗ sửa được của cả bài.

    Returns:
        (các trang đã sửa, số chỗ đã sửa); bài không đủ ngưỡng trả về nguyên trang và 0.
    """
    if not vocab:
        return pages, 0
    fixed = [restore_u_in_text(p, vocab) for p in pages]
    total = sum(n for _, n in fixed)
    if total < min_hits:
        return pages, 0
    return [t for t, _ in fixed], total
