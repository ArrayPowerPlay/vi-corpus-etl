"""
Trích text HTML (trafilatura: lấy phần nội dung chính, bỏ menu / quảng cáo / chân trang) và EPUB (ebooklib, mỗi chương
theo thứ tự spine là một trang; chương là HTML nhưng không có menu nên đọc toàn bộ chữ bằng BeautifulSoup).
"""

import re
from pathlib import Path

_BLANKS = re.compile(r"\n\s*\n+")  # nhiều dòng trống liền nhau -> một dòng trống (giữ ranh giới đoạn văn)


def html_to_text(html: str | bytes) -> str:
    """
    Toàn bộ chữ của một trang HTML, mỗi khối một dòng (BeautifulSoup), bỏ script / style.

    Nhận bytes thì BeautifulSoup tự dò bảng mã (meta charset, BOM), không ép UTF-8.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    lines = (ln.strip() for ln in soup.get_text("\n").splitlines())
    return "\n".join(ln for ln in lines if ln)


def extract_html(path: Path) -> list[dict]:
    """
    Trích phần nội dung chính của file HTML bằng trafilatura; trafilatura không tìm được thì lấy toàn bộ chữ.

    Returns:
        Một trang [{"text", "method": "html"}].
    """
    import trafilatura

    raw = path.read_bytes()  # để trafilatura / BeautifulSoup tự dò bảng mã (trang cũ windows-1258, TCVN...)
    text = trafilatura.extract(raw, include_tables=True, include_comments=False, favor_recall=True)
    return [{"text": _BLANKS.sub("\n\n", text or html_to_text(raw)).strip(), "method": "html"}]


def extract_epub(path: Path) -> list[dict]:
    """
    Trích text EPUB: mỗi tài liệu trong spine (thường là một chương) một trang.

    Returns:
        [{"text", "method": "epub"}] theo thứ tự đọc; chương rỗng bị bỏ.
    """
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    order = [book.get_item_with_id(idref) for idref, _ in book.spine]
    items = [it for it in order if it is not None and it.get_type() == ebooklib.ITEM_DOCUMENT]
    pages = []
    for item in items:
        text = html_to_text(item.get_content())
        if text:
            pages.append({"text": text, "method": "epub"})
    return pages
