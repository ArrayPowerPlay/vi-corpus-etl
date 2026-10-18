"""
Làm sạch riêng của bài báo khoa học trước khi ghép các trang thành một văn bản (R-26 mục 3, J-05 b): bỏ trang bìa, trang
mục lục và phần "Tài liệu tham khảo". Dùng heuristic theo tiêu đề, chạy lúc sinh bản ghi (không lưu vào interim) nên đổi
ngưỡng không phải trích lại PDF. Bước làm sạch chung (tiêu đề chạy, số trang, ghép dòng) vẫn là
vi_corpus.common.pdf_text.clean_pages, chạy sau bước này.

Các hằng số là điểm khởi đầu, chưa kiểm trên bài thật; số trang / dòng bị bỏ được trả về để ghi vào meta và xem lại.
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
# phần đứng sau tài liệu tham khảo: khớp ĐẦU dòng ("Phụ lục 1", "Abstract: this paper ...", "Tóm tắt: ...")
_AFTER_REF = re.compile(r"^(?:" + "|".join(sorted(AFTER_REF_HEADINGS, key=len, reverse=True)) + r")\b")
_LEADER = re.compile(r"(?:\.{4,}|…{2,}|(?:\. ){4,})\s*\d+\s*$")


def heading_key(line: str) -> str:
    """Khoá so sánh tiêu đề: NFC, chữ thường, bỏ số thứ tự đầu dòng ("5.", "IV)", "2.1") và dấu :. cuối dòng, gộp khoảng trắng."""
    text = _NUMBERING.sub("", unicodedata.normalize("NFC", line).strip()).casefold()
    return re.sub(r"\s+", " ", text).strip(" :.\t")


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
        if not in_refs and k >= REF_MIN_POS * total and key in REF_HEADINGS:
            in_refs = True
        if in_refs:
            dropped.add((p, i))
    if not dropped:
        return pages, 0
    out = ["\n".join(ln for i, ln in enumerate(text.splitlines()) if (p, i) not in dropped) for p, text in enumerate(pages)]
    return out, len(dropped)


def clean_paper_pages(pages: list[str]) -> tuple[list[str], dict]:
    """
    Bỏ trang bìa, mục lục và phần tài liệu tham khảo của một bài báo.

    Returns:
        (các trang đã làm sạch, {"cover_pages", "toc_pages", "reference_lines"}).
    """
    pages, stats = drop_front_matter(pages)
    pages, stats["reference_lines"] = strip_references(pages)
    return pages, stats
