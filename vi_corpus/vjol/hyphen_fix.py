"""
Khôi phục chữ "ư" cho bài VJOL bị font đời cũ vẽ "ư" thành dấu gạch nối (J-33 A, J-35, J-37): "tr-ờng", "ng-ời", "đ-ợc", "Nh-ng", "t- liệu", "trung - ơng", "xư a".

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

J-37 thêm các dạng còn sót ở v7 (đo trên 04_quality.parquet của v7, 140 đoạn):
- (A1) chữ đầu viết hoa tách khoảng trắng: "V- ơng" -> "Vương", "Đ- ờng" -> "Đường", "H- ng" -> "Hưng", "M- ỜNG" -> "MƯỜNG" (chữ hoa chỉ nhận khi ghép liền ra âm tiết thật; không giữ khoảng trắng nếu phần sau là chữ ASCII hoặc chữ hoa đứng riêng: "C- section", "T- ơ" giữ khoảng trắng nguyên);
- (A2) "-" đứng một mình là cả âm tiết ư-: "trung - ơng" -> "trung ương", "hương - ớc" -> "hương ước", "quy - ước" -> "quy ước" (bỏ dấu thừa); chữ đứng trước phải là chữ Latinh thường và phần sau thuộc danh sách đóng (ớc, ơng, ớp, ợc, ờng, ởng, ớng, ợng; ước, ưu) để dấu ngắt " - " thật ("σ - ứng suất", "A - ớt") giữ nguyên;
- (A3) "tối - u" -> "tối ưu", "các - u điểm" -> "các ưu điểm" (chữ "u" rời; công thức "u^{(k)} - u^{(k-1)}" giữ nguyên vì chữ đứng trước là ")");
- (A4) "nh-" đứng trước dấu câu: "nh- “bức" -> "như “bức".
A1-A3 thuộc nhóm dạng có khoảng trắng nên cũng cần MIN_DOC_HITS chỗ sửa được trong cả bài; riêng "nh- " (A4 và "nh- người", "nh- trước") sửa luôn vì "nh" + "- " gần như chắc chắn là "như".

(B) "ư" bị tách rời khỏi âm tiết (không có gạch nối), quy tắc chặt vì nhiều cặp "X ư + Y" là câu văn thật ("như sau", "tư liệu", "cư dân"):
- tail "a" / "u" chỉ ghép khi ra từ thật thuộc SPLIT_WORDS: "xư a" -> "xưa", "chư a" -> "chưa", "đư a" -> "đưa", "lư u" -> "lưu";
- tail là phần còn lại có chữ ơ-họ + c / ng / i: "hư ớng" -> "hướng", "nư ớc" -> "nước", "đư ợc" -> "được", "ngư ời" -> "người";
- tail là chữ o / ó / ố thay cho ơ / ớ: "nư óc" -> "nước", "lư ối" -> "lưới", "tư ong" -> "tương" (không áp cho "như", "sư", "ngư", "thư" vì "như ong", "sư ông", "ngư ông", "ung thư ống" là câu thật; không ghép tail có dấu khác như ồ / ô vì không biết dấu thật: "tư ông" là tương hay tướng);
- ư lặp trước nguyên âm đôi ươ + ng: "hư ưởng" -> "hưởng" (không ghép "tư ứng", "như ứng", "tư ước": câu thật).
KHÔNG ghép "phư ng", "tư ng", "trư c" ...: đó là chữ ơ / ợ / ứ bị mất (ph-ương, tư-ợng), ghép thành "phưng" là sai chữ; để lỗi lost_letters / quality xử lý.
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
_LATIN_LOWER = frozenset("abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
# A2: "-" đứng một mình là cả âm tiết ư-. Phần sau (không có ư) nằm trong danh sách đóng: toàn âm tiết có phụ âm cuối nên không thể là từ đứng riêng; HORN_U là phần sau đã mang ư (dấu "-" chỉ thừa).
_LONE_O = re.compile(r"[ơờớởỡợ](?:ng|c|p)")
_LONE_U = frozenset({"ước", "ưu"})
_LONE = re.compile(rf" -(?: ?)([ơờớởỡợưừứửữự]{_LETTERS}*)(?!{_LETTERS}|-)")
# A3: chữ "u" rời sau dấu "-" ("tối - u tiên"). Phía sau không được là chỉ số / dấu mở ngoặc của công thức.
_U_ALONE = re.compile(rf" - u(?!{_LETTERS}|[\d^_{{(\\])")
# "nh- người", "nh- trước": "nh" + "- " gần như luôn là "như" (không chữ nào bắt đầu bằng "nh-" rồi cách ra), nên sửa luôn không cần ngưỡng cả bài (J-37: v7 còn nhiều bài chỉ có 1 chỗ).
_NH_SPACED = re.compile(rf"(?<![\w-])([Nn]h)- ({_LETTERS}+)(?!{_LETTERS}|-)")
# A4: "nh-" đứng trước dấu câu / dấu nháy: "nh- “bức" -> "như “bức"
_NH_PUNCT = re.compile(r"(?<![\w-])([Nn]h)- (?=[^\w\s])")
# B: "ư" bị tách rời (không có gạch nối). Đầu từ = chữ cái kết thúc bằng ư; phần sau 1-5 chữ cái.
_SPLIT = re.compile(rf"(?<![\w-])({_LETTERS}*[ưƯ]) ({_LETTERS}{{1,5}})(?!{_LETTERS})")
SPLIT_WORDS = frozenset({"xưa", "chưa", "thưa", "đưa", "dưa", "cưa", "mưa", "trưa", "lưu", "hưu", "sưu", "bưu", "mưu", "ưu"})
_SPLIT_HORN = re.compile(r"[ơờớởỡợ](?:ng|c|i)")  # phần còn lại của âm tiết ư-ơ: ớng, ớc, ợc, ời ...
_SPLIT_DUP = re.compile(r"ư[ơờớởỡợ]ng")
_SPLIT_O = re.compile(r"(?:[oó](?:ng|c|i)|ối)")  # ơ bị đọc thành o / ó / ố: ong, oi, óc, óng, ối ...; dấu thanh khác (ồ, ộ, ô ...) không dùng vì không biết dấu thật ("tư ông" có thể là tương hay tướng)
_O_TO_HORN = {"o": "ơ", "ó": "ớ", "ố": "ớ"}
_SPLIT_O_SKIP = frozenset({"như", "sư", "ngư", "thư"})  # "như ong", "sư ong", "ngư ông", "ung thư óc / ống" là câu thật
MIN_DOC_HITS = 2  # dạng có khoảng trắng chỉ sửa khi cả bài có ít nhất ngần ấy chỗ sửa được (tránh sửa nhầm cách viết thật lẻ tẻ)


def _is_initial(lead: str, spaced: bool, tail: str = "") -> bool:
    """
    True nếu `lead` là phụ âm đầu hợp lệ. Dạng có khoảng trắng nhận chữ thường, chữ viết hoa chữ đầu ("V- ơng", "Đ- ờng", "Kh- ") và dạng toàn chữ hoa khi phần sau cũng toàn chữ hoa
    ("M- ỜNG"); chữ hoa đứng riêng bị loại ở J-28 nay chỉ còn bị chặn bởi phép kiểm âm tiết và `_restore_spaced` ("N- TiO2", "C- section" không ra âm tiết).
    """
    low = lead.lower()
    if not ((low in _SINGLE) if len(low) == 1 else low in _MULTI):
        return False
    if not spaced or lead == low:
        return True
    return lead[1:] == low[1:] or (lead.isupper() and tail.isupper())


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
    ("hướng"); (2) phần sau bắt đầu bằng ư-họ thì bỏ "- " ("được"); (3) ghép liền với "ư" ("mưu"); (4) giữ khoảng trắng khi "tư" và "liệu" đều là âm tiết thật (phần sau >= 2 chữ hoặc có dấu; đầu viết hoa phải từ hai chữ trở lên và phần sau phải có dấu: "Xa N- ở" không đổi).
    """
    lead, tail = m.group(1), m.group(2)
    if not _is_initial(lead, spaced=True, tail=tail):
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
    # phần sau một chữ ASCII ("t", "n") là ký hiệu, không phải âm tiết; đầu viết hoa chỉ giữ khoảng trắng khi phần sau có dấu ("T- liệu"), vì "C- section", "D- ong" là tiếng Anh / tên riêng
    ok = (len(tail) >= 2 or not tail.isascii()) and (lead == lead.lower() or (len(lead) > 1 and not tail.isascii())) and _valid(head, vocab) and _valid(tail, vocab)
    return f"{head} {tail}" if ok else None


def _restore_lone(m: re.Match, vocab: frozenset[str]) -> str | None:
    """
    Từ sửa được cho " - ơng" / " - ớc" (dấu "-" đứng một mình là cả âm tiết ư-): "trung - ơng" -> "trung ương" (A2). Chữ đứng trước phải là chữ Latinh thường; phần sau mang ơ-họ
    phải thuộc danh sách đóng `_LONE_O` và ghép "ư" ra âm tiết thật; phần sau đã mang ư-họ chỉ nhận `_LONE_U` (dấu "-" thừa: "quy - ước" -> "quy ước").
    """
    i = m.start()
    if i == 0 or m.string[i - 1] not in _LATIN_LOWER:
        return None
    tail = m.group(1)
    low = tail.lower()
    if low[0] in _HORN_U:
        return f" {tail}" if low in _LONE_U else None
    if tail == low and _LONE_O.fullmatch(low) and _valid("ư" + low, vocab):
        return f" ư{tail}"
    return None


def _restore_u_alone(m: re.Match, vocab: frozenset[str]) -> str | None:
    """" - u" -> " ưu" (A3: "tối - u tiên" -> "tối ưu tiên"); chữ đứng trước phải là chữ Latinh thường (công thức "u^{(k)} - u^{(k-1)}" có ")" đứng trước nên giữ nguyên)."""
    i = m.start()
    return " ưu" if i > 0 and m.string[i - 1] in _LATIN_LOWER and "ưu" in vocab else None


def _restore_nh(m: re.Match, vocab: frozenset[str]) -> str | None:
    """"nh- “bức" -> "như “bức" (A4): "nh-" đứng trước dấu câu nên không thể là phụ âm đầu của chữ khác."""
    return ("Như " if m.group(1)[0] == "N" else "như ") if "như" in vocab else None


def _restore_split(m: re.Match, vocab: frozenset[str]) -> str | None:
    """
    Từ ghép lại được cho "ư" bị tách rời (B): "xư a" -> "xưa", "hư ớng" -> "hướng", "nư óc" -> "nước", "hư ưởng" -> "hưởng"; None nếu không thoả quy tắc (docstring đầu file).
    Chữ hoa toàn bộ và phần sau không phải chữ thường thì bỏ qua.
    """
    head, tail = m.group(1), m.group(2)
    low = head.lower()
    if (head.isupper() and len(head) > 1) or tail != tail.lower():
        return None
    if tail in ("a", "u"):
        return head + tail if low + tail in SPLIT_WORDS else None
    if _SPLIT_DUP.match(tail):  # "hư ưởng": ư lặp ("ưở" là nguyên âm đôi ươ), bỏ ư của đầu từ; "đầu tư ứng", "như ứng" là câu thật nên không nằm ở đây
        stem = head[:-1]
        ok = stem.lower() in _MULTI or stem.lower() in _SINGLE
        return stem + tail if ok and _valid(stem + tail, vocab) else None
    if _SPLIT_HORN.fullmatch(tail):
        return head + tail if _valid(head + tail, vocab) else None
    if _SPLIT_O.fullmatch(tail) and low not in _SPLIT_O_SKIP:
        fixed = head + _O_TO_HORN[tail[0]] + tail[1:]
        return fixed if _valid(fixed, vocab) else None
    return None


def restore_u_in_text(text: str, vocab: frozenset[str], spaced: bool = True) -> tuple[str, int]:
    """
    Sửa mọi chỗ "-" thay "ư" trong một đoạn văn bản (không áp ngưỡng cả bài, xem restore_u_in_pages).

    Args:
        text:   Văn bản (đã NFC hoặc chưa; kết quả đã NFC).
        vocab:  Tập âm tiết thật (chữ thường, NFC).
        spaced: False thì chỉ sửa dạng liền chữ ("tr-ờng") và ư bị tách rời ("xư a"), bỏ dạng có khoảng trắng ("t- liệu", "trung - ơng", "tối - u"). Dạng "nh- " luôn được sửa.

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

    text = _NH_SPACED.sub(repl(_restore_spaced), text)
    if spaced:
        text = _SPACED.sub(repl(_restore_spaced), text)
        text = _LONE.sub(repl(_restore_lone), text)
        text = _U_ALONE.sub(repl(_restore_u_alone), text)
    text = _NH_PUNCT.sub(repl(_restore_nh), text)
    text = _GLUED.sub(repl(_restore_glued), text)
    text = _SPLIT.sub(repl(_restore_split), text)
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
