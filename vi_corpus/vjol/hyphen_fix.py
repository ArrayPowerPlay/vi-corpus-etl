"""
Khôi phục chữ "ư" cho bài VJOL bị font đời cũ vẽ "ư" thành dấu gạch nối (J-33 A, J-35): "tr-ờng", "ng-ời", "đ-ợc", "Nh-ng", "t- liệu".

Căn cứ J-32 và ảnh trang của S-18: OCR VLM KHÔNG sửa được lỗi này vì chính ảnh trang cũng in dấu "-" (hoặc "- " cách một khoảng) thay cho "ư"; ở font TCVN3 mã của "ư"
(0xAD, mã "gạch nối mềm") được PyMuPDF trả thành "-". Quy tắc cố định theo dữ liệu: "-" nằm trong chữ là "ư". Chạy lúc sinh bản ghi
(vi_corpus.vjol.clean.clean_paper_pages), nên đổi quy tắc không phải trích lại PDF.

An toàn nhờ các lớp kiểm:
1. Chữ đứng trước "-" phải là phụ âm đầu thật của tiếng Việt (ngh, ng, nh, ch, kh, th, tr, gh, gi, ph, qu hoặc một phụ âm đơn), nên "x-ray", "e-mail", "Nam-Á", "kinh tế-xã hội" không bao giờ bị xét.
2. Kết quả phải là âm tiết thật trong tập âm tiết tham chiếu (processed/ocr_bakeoff/syllables.json); không thì giữ nguyên ("t- Statistics", "n- butan", "a- Điabas" của thống kê / hoá học không khớp).
3. Dạng có khoảng trắng (dễ trùng cách viết thật) chỉ sửa khi cả bài có ít nhất MIN_DOC_HITS chỗ sửa được; dạng liền chữ ("ng-ời") gần như không thể là gạch nối thật nên sửa luôn.

Các dạng lỗi, theo thứ tự thử (cái đầu tiên cho âm tiết thật thì dùng):
- liền chữ: "tr-ờng" -> "trường", "Nh-ng" -> "Nhưng", "D-ỚI" -> "DƯỚI"; hoặc BỎ "-" khi phần sau đã mang chữ có móc ("t-ướng" -> "tướng", "x-ướng" -> "xướng", "s-ở" -> "sở", J-35);
- có khoảng trắng sau dấu: "h- ớng" -> "hướng" (bỏ khoảng trắng, vì "ư" + "ơ" là một nguyên âm đôi "ươ"), "đ- ược" -> "được" (bỏ cả "- ", phần sau đã có "ư"), "m- u" -> "mưu", "t- liệu" -> "tư liệu" (giữ khoảng trắng, cả hai âm tiết đều thật), "nh- ở" -> "như ở".
"""

import re
import unicodedata

# Phụ âm đầu của tiếng Việt đứng trước "ư" bị mất. Chữ đơn dạng viết hoa ("N- TiO2", "C- section") là tiếng Anh / hoá học nên chỉ nhận chữ đơn viết thường trong dạng có khoảng trắng (J-28).
_MULTI = ("ngh", "ng", "nh", "ch", "kh", "th", "tr", "gh", "gi", "ph", "qu")
_SINGLE = frozenset("bcdđghklmnpqrstvx")
_LETTERS = r"[^\W\d_]"
# "tr-ờng": đầu từ (không dính chữ / gạch nối trước), phụ âm đầu tối đa 3 chữ, "-", phần còn lại của từ (không dính chữ / gạch nối sau).
# Số chú thích dính cuối từ ("ng-ời5", "t-ợng19") không chặn khớp; chữ cái hoặc gạch nối kế tiếp thì chặn ("ch-ởng-anh").
_GLUED = re.compile(rf"(?<![\w-])({_LETTERS}{{1,3}})-({_LETTERS}+)(?!{_LETTERS}|-)")
# "t- liệu", "h- ớng": phụ âm đầu, "-", đúng một khoảng trắng, từ kế tiếp.
_SPACED = re.compile(rf"(?<![\w-])({_LETTERS}{{1,3}})- ({_LETTERS}+)(?!{_LETTERS}|-)")
_HORN_O = frozenset("ơờớởỡợ")  # nguyên âm đứng sau "ư" tạo nguyên âm đôi "ươ": dạng có khoảng trắng phải bỏ khoảng trắng
_HORN_U = frozenset("ưừứửữự")  # phần sau đã mang "ư": "-" là dấu thừa, bỏ đi ("đ- ược" -> "được")
MIN_DOC_HITS = 2  # dạng có khoảng trắng chỉ sửa khi cả bài có ít nhất ngần ấy chỗ sửa được (tránh sửa nhầm cách viết thật lẻ tẻ)


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


def _valid(word: str, vocab: frozenset[str]) -> bool:
    """True nếu `word` (NFC, chữ thường) là âm tiết thật trong `vocab`."""
    return unicodedata.normalize("NFC", word).lower() in vocab


def _restore_glued(m: re.Match, vocab: frozenset[str]) -> str | None:
    """
    Từ sửa được cho một chỗ "tr-ờng" (None nếu không thoả quy tắc): thử thay "-" bằng "ư"; không ra âm tiết thật và phần sau bắt đầu bằng chữ có móc thì
    thử bỏ "-" ("t-ướng" -> "tướng").
    """
    lead, tail = m.group(1), m.group(2)
    if not _is_initial(lead, spaced=False):
        return None
    fixed = lead + _upper_u(tail) + tail
    if _valid(fixed, vocab):
        return fixed
    joined = lead + tail
    return joined if tail[0].lower() in _HORN_O | _HORN_U and _valid(joined, vocab) else None


def _restore_spaced(m: re.Match, vocab: frozenset[str]) -> str | None:
    """
    Từ sửa được cho một chỗ "t- liệu" / "h- ớng" / "đ- ược" (None nếu không thoả quy tắc), thử theo thứ tự: (1) phần sau bắt đầu bằng ơ-họ thì ghép liền với "ư"
    ("hướng"); (2) phần sau bắt đầu bằng ư-họ thì bỏ "- " ("được"); (3) ghép liền với "ư" ("mưu"); (4) giữ khoảng trắng khi "tư" và "liệu" đều là âm tiết thật (phần sau >= 2 chữ hoặc có dấu).
    """
    lead, tail = m.group(1), m.group(2)
    if not _is_initial(lead, spaced=True):
        return None
    u = _upper_u(tail)
    merged = lead + u + tail
    first = tail[0].lower()
    if first in _HORN_O and _valid(merged, vocab):
        return merged
    if first in _HORN_U and _valid(lead + tail, vocab):
        return lead + tail
    if _valid(merged, vocab):
        return merged
    head = lead + u
    ok = (len(tail) >= 2 or not tail.isascii()) and _valid(head, vocab) and _valid(tail, vocab)  # phần sau một chữ ASCII ("t", "n") là ký hiệu, không phải âm tiết
    return f"{head} {tail}" if ok else None


def restore_u_in_text(text: str, vocab: frozenset[str], spaced: bool = True) -> tuple[str, int]:
    """
    Sửa mọi chỗ "-" thay "ư" trong một đoạn văn bản (không áp ngưỡng cả bài, xem restore_u_in_pages).

    Args:
        text:   Văn bản (đã NFC hoặc chưa; kết quả đã NFC).
        vocab:  Tập âm tiết thật (chữ thường, NFC).
        spaced: False thì chỉ sửa dạng liền chữ ("tr-ờng"), bỏ dạng có khoảng trắng ("t- liệu").

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

    if spaced:
        text = _SPACED.sub(repl(_restore_spaced), text)
    text = _GLUED.sub(repl(_restore_glued), text)
    return text, n


def restore_u_in_pages(pages: list[str], vocab: frozenset[str] | None, min_hits: int = MIN_DOC_HITS) -> tuple[list[str], int]:
    """
    Khôi phục "ư" trong các trang của MỘT bài: dạng liền chữ ("ng-ời") sửa luôn; dạng có khoảng trắng ("t- liệu") chỉ sửa khi cả bài có ít nhất `min_hits` chỗ sửa được
    (tính cả dạng liền chữ), vì một chỗ lẻ có thể là cách viết thật.

    Args:
        pages:    Text từng trang.
        vocab:    Tập âm tiết thật; None thì không làm gì (thiếu syllables.json).
        min_hits: Ngưỡng số chỗ sửa được của cả bài cho dạng có khoảng trắng.

    Returns:
        (các trang đã sửa, số chỗ đã sửa).
    """
    if not vocab:
        return pages, 0
    glued = [restore_u_in_text(p, vocab, spaced=False) for p in pages]
    full = [restore_u_in_text(p, vocab) for p in pages]
    n_glued, n_full = sum(n for _, n in glued), sum(n for _, n in full)
    if n_full >= min_hits:
        return [t for t, _ in full], n_full
    return ([t for t, _ in glued], n_glued) if n_glued else (pages, 0)
