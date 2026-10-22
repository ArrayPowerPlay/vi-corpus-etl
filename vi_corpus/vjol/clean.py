"""
Làm sạch riêng của bài báo khoa học trước khi ghép các trang thành một văn bản (R-26 mục 3, J-05 b): bỏ trang bìa, trang
mục lục và phần "Tài liệu tham khảo". Dùng heuristic theo tiêu đề, chạy lúc sinh bản ghi (không lưu vào interim) nên đổi
ngưỡng không phải trích lại PDF. Bước làm sạch chung (tiêu đề chạy, số trang, ghép dòng) vẫn là
vi_corpus.common.pdf_text.clean_pages, chạy sau bước này.

Các hằng số là điểm khởi đầu, chưa kiểm trên bài thật; số trang / dòng bị bỏ được trả về để ghi vào meta và xem lại.

J-22 (6), J-23: ngoài ba việc trên còn
- bỏ dòng khuôn mẫu của tạp chí ("Ngày nhận bài", "Ngày duyệt đăng", "Received", "Email:" ...) và khối tác giả ở đầu bài (dòng có địa chỉ email trong
  AUTHOR_BLOCK_LINES dòng đầu), vì chúng lặp ở hàng trăm đoạn;
- bỏ tiêu đề trang tạp chí dính giữa dòng ("Tạp chí Nghiên cứu Tôn giáo, Số 3 - 2015") bằng regex dựng từ tên tạp chí trong metadata;
- che email còn lại trong thân bài bằng <EMAIL> và số điện thoại bằng <PHONE> (che số điện thoại là suy luận khi ghi J-22, chưa được chủ dự án nói riêng);
- nhận tiêu đề "Tài liệu tham khảo" có hậu tố ("chính", "tiếng Việt") hoặc dính liền mục 1 ("Tài liệu tham khảo 1. Nguyễn ...").
"""

import re
import unicodedata

# Tiêu đề phần tài liệu tham khảo (đã chuẩn hoá bằng heading_key). Chỉ khớp CẢ DÒNG, nên câu văn nhắc tới "tài liệu tham khảo" an toàn.
REF_HEADINGS = frozenset({
    "tài liệu tham khảo", "danh mục tài liệu tham khảo", "danh sách tài liệu tham khảo", "tài liệu trích dẫn",
    "danh mục tham khảo", "tham khảo", "references", "reference", "bibliography", "literature cited",
    "references and notes",
})
# Tiêu đề của phần đứng sau tài liệu tham khảo mà vẫn là nội dung bài (tóm tắt tiếng Anh, phụ lục, lời cảm ơn, thông tin tác giả):
# gặp thì hết phần tham khảo, giữ lại từ đó.
AFTER_REF_HEADINGS = frozenset({
    "phụ lục", "appendix", "appendices", "abstract", "tóm tắt", "summary", "lời cảm ơn", "cảm ơn", "acknowledgment",
    "acknowledgments", "acknowledgement", "acknowledgements", "thông tin tác giả", "thông tin về tác giả",
    "about the authors", "author contributions",
})
TOC_HEADINGS = frozenset({"mục lục", "table of contents", "contents"})
REF_MIN_POS = 0.3  # tiêu đề "Tài liệu tham khảo" phải nằm sau 30% số dòng của bài (tránh khớp nhầm ở đầu bài)
TOC_SCAN_PAGES = 3  # chỉ tìm trang mục lục trong ngần ấy trang đầu
TOC_LEADER_FRAC = 0.5  # trang có >= tỉ lệ này số dòng kết thúc bằng "......... 12" là trang mục lục
TOC_MIN_LINES = 5
COVER_MAX_WORDS = 40  # trang đầu ít hơn ngần ấy từ (bài >= COVER_MIN_PAGES trang) là trang bìa
COVER_MIN_PAGES = 3

# số thứ tự đầu tiêu đề: "5." "2.1" "IV)" (số La Mã viết hoa, hoặc viết thường nhưng phải có dấu . / ) để không cắt nhầm từ như "civil")
_NUMBERING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+[.)]?|[ivxlc]+[.)])\s+")
# Dòng kết thúc phần tài liệu tham khảo (khớp trên khoá heading_key, đã chữ thường): từ khoá phải đứng một mình, hoặc theo sau là dấu
# câu (": . - – — )"), nên "Abstract: this paper ..." và "Phụ lục 1: Bảng" kết thúc phần tham khảo còn mục tham khảo như "Abstract algebra,
# Dummit (2004)" hay "Summary of the report, World Bank" thì không. Phụ lục / Appendix có thể kèm nhãn ngắn (số, một chữ cái, số La Mã <= 4 ký tự).
_APPENDIX = "phụ lục|appendix|appendices"
_OTHER_END = "|".join(sorted(AFTER_REF_HEADINGS - set(_APPENDIX.split("|")), key=len, reverse=True))
_AFTER_REF = re.compile(rf"^(?:(?:{_APPENDIX})(?:\s+(?:\d+|[a-z]|[ivxlc]{{1,4}}))?|{_OTHER_END})\s*(?:$|[:.\-–—)])")
_LEADER = re.compile(r"(?:\.{4,}|…{2,}|(?:\. ){4,})\s*\d+\s*$")
# tiêu đề tham khảo có hậu tố hoặc dính liền mục 1: "tài liệu tham khảo chính", "tài liệu tham khảo 1. nguyễn văn a (2010)" (khoá heading_key đã chữ thường,
# nên số thứ tự đầu dòng "5." đã bị bỏ; chỉ nhận số 1 để câu văn "Tham khảo 2 nguồn ..." không khớp)
_REF_SUFFIX = r"(?:chính|chủ yếu|chọn lọc|bổ sung|tiếng việt|tiếng anh)"
_REF_STUCK = re.compile(r"^(?P<h>{heads})(?:\s+{suf})*(?:\s+\[?1[\].):]?(?:\s.*)?)?$".format(
    heads="|".join(sorted((re.escape(h) for h in REF_HEADINGS), key=len, reverse=True)), suf=_REF_SUFFIX))

AUTHOR_BLOCK_LINES = 40  # khối tác giả chỉ tìm trong ngần ấy dòng không rỗng đầu bài
# dòng khuôn mẫu của tạp chí; dòng ngắn (<= BOILER_MAX_WORDS từ) mới bị bỏ để câu văn nhắc tới "email" không mất
BOILER_MAX_WORDS = 25
AUTHOR_LINE_MAX_WORDS = 12  # dòng của khối tác giả (tên, cơ quan, email) ngắn; câu văn dài có email thì chỉ che email
_BOILER = re.compile(r"^\W*(?:ngày\s+(?:nhận|duyệt|gửi|chấp nhận|đăng|sửa)\b|received\b|accepted\b|revised\b|submitted\b|published\b|"
                     r"(?:e-?mail|thư điện tử|(?:số\s+)?điện thoại|tel|phone|fax)\s*[:：])", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# số điện thoại Việt Nam: 0 hoặc +84, 9-10 chữ số sau đó, có thể ngăn bằng dấu cách / chấm / gạch; không dính vào số dài hơn
PHONE = re.compile(r"(?<![\w.])(?:\+84|0)(?:[\s.-]?\d){8,10}(?![\w])")


def heading_key(line: str) -> str:
    """Khoá so sánh tiêu đề: NFC, chữ thường, bỏ số thứ tự đầu dòng ("5.", "IV)", "2.1") và dấu :. cuối dòng, gộp khoảng trắng."""
    text = _NUMBERING.sub("", unicodedata.normalize("NFC", line).strip()).casefold()
    return re.sub(r"\s+", " ", text).strip(" :.\t")


def is_ref_heading(key: str) -> bool:
    """True nếu khoá tiêu đề (heading_key) là tiêu đề phần tài liệu tham khảo: khớp cả dòng, có hậu tố (_REF_SUFFIX), hoặc dính liền mục 1."""
    return key in REF_HEADINGS or bool(_REF_STUCK.match(key))


def is_toc_page(text: str) -> bool:
    """Trang mục lục: có tiêu đề "Mục lục" / "Contents" trong 3 dòng đầu, hoặc phần lớn dòng kết thúc bằng dấu chấm dẫn + số trang."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if any(heading_key(ln) in TOC_HEADINGS for ln in lines[:3]):
        return True
    return len(lines) >= TOC_MIN_LINES and sum(bool(_LEADER.search(ln)) for ln in lines) / len(lines) >= TOC_LEADER_FRAC


def drop_front_matter(pages: list[str]) -> tuple[list[str], dict]:
    """
    Bỏ trang bìa (trang đầu rất ngắn, bài từ COVER_MIN_PAGES trang) và trang mục lục (trong TOC_SCAN_PAGES trang đầu).

    Trang bị bỏ thành chuỗi rỗng (giữ nguyên số trang). Trang rỗng sẵn không tính là bị bỏ.

    Returns:
        (các trang sau khi bỏ, {"cover_pages": n, "toc_pages": n}).
    """
    out = list(pages)
    stats = {"cover_pages": 0, "toc_pages": 0}
    for i, text in enumerate(pages[:TOC_SCAN_PAGES]):
        if not text.strip():
            continue
        if i == 0 and len(pages) >= COVER_MIN_PAGES and len(text.split()) < COVER_MAX_WORDS:
            out[i], stats["cover_pages"] = "", stats["cover_pages"] + 1
        elif is_toc_page(text):
            out[i], stats["toc_pages"] = "", stats["toc_pages"] + 1
    return out, stats


def strip_references(pages: list[str]) -> tuple[list[str], int]:
    """
    Bỏ mọi phần tài liệu tham khảo: mỗi đoạn bắt đầu ở một dòng khớp REF_HEADINGS (nằm sau REF_MIN_POS số dòng) và kết thúc ở dòng
    bắt đầu bằng một tiêu đề AFTER_REF_HEADINGS (Abstract, Phụ lục, ...) hoặc hết bài. Bài thường có hai danh sách (tiếng Việt rồi
    "References" tiếng Anh sau phần Abstract), nên xử lý hết chứ không chỉ tiêu đề cuối.

    Returns:
        (các trang sau khi bỏ, số dòng không rỗng đã bỏ). Không thấy tiêu đề thì giữ nguyên.
    """
    lines = [(p, i, ln) for p, text in enumerate(pages) for i, ln in enumerate(text.splitlines()) if ln.strip()]
    total = len(lines)
    dropped: set[tuple[int, int]] = set()
    in_refs = False
    for k, (p, i, ln) in enumerate(lines):
        key = heading_key(ln)
        if in_refs and _AFTER_REF.match(key):
            in_refs = False
        if not in_refs and k >= REF_MIN_POS * total and is_ref_heading(key):
            in_refs = True
        if in_refs:
            dropped.add((p, i))
    if not dropped:
        return pages, 0
    out = ["\n".join(ln for i, ln in enumerate(text.splitlines()) if (p, i) not in dropped) for p, text in enumerate(pages)]
    return out, len(dropped)


def journal_header_pattern(journal_title: str | None) -> re.Pattern | None:
    """
    Regex bắt tiêu đề trang tạp chí ("<tên tạp chí>[, ] Số n - năm"), dựng từ tên tạp chí trong metadata; None nếu không có tên.

    Khớp cả bản có và không có tiền tố "Tạp chí", không phân biệt hoa thường và khoảng trắng; phần "Số n", "No. n", "Tập n" và năm theo sau là tuỳ chọn.
    """
    title = re.sub(r"\s+", " ", unicodedata.normalize("NFC", journal_title or "")).strip()
    if len(title) < 5:
        return None
    bare = re.sub(r"^tạp chí\s+", "", title, flags=re.IGNORECASE)
    names = {title, bare} if len(bare) >= 5 else {title}
    body = "|".join(r"\s+".join(re.escape(w) for w in n.split()) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"(?:{body})(?:[\s,;|–—-]*(?:số|no\.?|tập|vol\.?)\s*\d+[\w/.]*)?(?:[\s,;|–—-]*[(]?\b(?:19|20)\d{{2}}[)]?)?", re.IGNORECASE)


def strip_boilerplate(pages: list[str], journal_title: str | None = None) -> tuple[list[str], dict]:
    """
    Bỏ khuôn mẫu của tạp chí trong các trang (J-22 (6)), giữ nguyên số trang.

    Việc làm, theo thứ tự: (1) bỏ dòng khuôn mẫu ngắn (_BOILER: "Ngày nhận bài", "Received", "Email:" ...); (2) bỏ dòng có email và không quá AUTHOR_LINE_MAX_WORDS từ trong AUTHOR_BLOCK_LINES
    dòng không rỗng đầu bài (khối tác giả); (3) bỏ tiêu đề trang tạp chí khỏi dòng (cả khi dính giữa dòng), dòng thành rỗng thì bỏ; (4) che email
    còn lại bằng <EMAIL> và số điện thoại bằng <PHONE>.

    Returns:
        (các trang sau khi làm sạch, {"boilerplate_lines", "author_block_lines", "journal_headers", "emails_masked", "phones_masked"}).
    """
    header = journal_header_pattern(journal_title)
    stats = {"boilerplate_lines": 0, "author_block_lines": 0, "journal_headers": 0, "emails_masked": 0, "phones_masked": 0}
    seen = 0  # số dòng không rỗng đã duyệt (cho khối tác giả)
    out = []
    for text in pages:
        kept = []
        for line in text.splitlines():
            if not line.strip():
                kept.append(line)
                continue
            seen += 1
            short = len(line.split()) <= BOILER_MAX_WORDS
            if short and _BOILER.match(line):
                stats["boilerplate_lines"] += 1
                continue
            if len(line.split()) <= AUTHOR_LINE_MAX_WORDS and seen <= AUTHOR_BLOCK_LINES and EMAIL.search(line):
                stats["author_block_lines"] += 1
                continue
            if header is not None:
                line, n = header.subn(" ", line)
                stats["journal_headers"] += n
                if n and not re.search(r"\w", line):
                    continue
                line = re.sub(r"[ \t]{2,}", " ", line).strip()
            line, n_mail = EMAIL.subn("<EMAIL>", line)
            line, n_tel = PHONE.subn("<PHONE>", line)
            stats["emails_masked"] += n_mail
            stats["phones_masked"] += n_tel
            kept.append(line)
        out.append("\n".join(kept))
    return out, stats


def clean_paper_pages(pages: list[str], journal_title: str | None = None) -> tuple[list[str], dict]:
    """
    Bỏ trang bìa, mục lục, phần tài liệu tham khảo và khuôn mẫu tạp chí của một bài báo.

    Args:
        pages:         Text từng trang.
        journal_title: Tên tạp chí trong metadata (để bỏ tiêu đề trang tạp chí dính giữa dòng); None thì không bỏ.

    Returns:
        (các trang đã làm sạch, {"cover_pages", "toc_pages", "reference_lines", và các số đếm của strip_boilerplate}).
    """
    pages, stats = drop_front_matter(pages)
    pages, stats["reference_lines"] = strip_references(pages)
    pages, boiler = strip_boilerplate(pages, journal_title)
    return pages, {**stats, **boiler}
