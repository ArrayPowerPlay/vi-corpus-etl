"""
Trích text từ file tài liệu nhiều định dạng (D-05, R-14), dùng chung cho giáo trình và các nguồn file sau này.

Định dạng nhận theo NỘI DUNG đầu file (magic bytes), đuôi file chỉ dùng khi nội dung không đủ để phân biệt:
    pdf   "%PDF"                                  -> vi_corpus.common.pdf_text.extract_pdf (trang scan thì OCR)
    docx  zip có word/document.xml                -> python-docx
    pptx  zip có ppt/presentation.xml             -> python-pptx (mỗi slide một trang)
    epub  zip có mimetype "application/epub+zip"  -> ebooklib (mỗi chương một trang)
    doc / ppt  OLE2 (D0 CF 11 E0), phân biệt theo đuôi -> LibreOffice headless đổi sang docx / pptx rồi đọc như trên
    djvu  "AT&TFORM"                              -> djvulibre (djvutxt từng trang; trang không có chữ thì ddjvu + OCR)
    html  "<!doctype html" / "<html"              -> trafilatura (bỏ menu, quảng cáo), dự phòng BeautifulSoup
Nội dung không khớp mẫu nào thì đoán theo đuôi file (vd file .pdf hỏng vẫn được thử như PDF để báo lỗi rõ ràng).

parse_file trả về {"format", "pages", "error", "skipped", "parser_version"} và không ném lỗi:
- định dạng không hỗ trợ: skipped = "unsupported_format";
- thiếu công cụ ngoài (LibreOffice, djvulibre): skipped = "needs_libreoffice" / "needs_djvulibre" — cài xong thì chạy lại
  với --retry-skipped (docs/README.md Phần E có lệnh cài);
- file hỏng / có mật khẩu / lỗi chuyển đổi: error là thông báo lỗi.
Mỗi trang là {"text", "method"}; method của định dạng mới là tên định dạng ("docx", "html", "epub", "djvu"), trang
DJVU OCR có method "ocr", chưa OCR được thì "needs_ocr" (giống PDF).
"""

import zipfile
from collections.abc import Callable
from pathlib import Path

PARSER_VERSION = "2"  # 1 = chỉ pdf + pptx theo đuôi file; 2 = D-05 (magic bytes, docx/doc/ppt/djvu/html/epub)
SUPPORTED = ("pdf", "docx", "pptx", "doc", "ppt", "djvu", "html", "epub")
_SUFFIX = {".pdf": "pdf", ".docx": "docx", ".pptx": "pptx", ".doc": "doc", ".ppt": "ppt", ".djvu": "djvu",
           ".djv": "djvu", ".html": "html", ".htm": "html", ".epub": "epub"}


class MissingTool(RuntimeError):
    """Thiếu công cụ ngoài (LibreOffice, djvulibre); args[0] là mã skipped."""


def _zip_format(path: Path) -> str | None:
    """Định dạng của file zip theo các file bên trong (docx / pptx / epub), None nếu không nhận ra."""
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if "mimetype" in names and z.read("mimetype").strip() == b"application/epub+zip":
                return "epub"
    except (zipfile.BadZipFile, OSError):
        return None
    if "word/document.xml" in names:
        return "docx"
    if "ppt/presentation.xml" in names:
        return "pptx"
    return None


def detect_format(path: Path) -> str:
    """
    Định dạng của file theo nội dung đầu file (xem docstring module), dự phòng theo đuôi file.

    Returns:
        Một trong SUPPORTED, hoặc đuôi file (không dấu chấm) / "unknown" nếu không hỗ trợ.
    """
    with open(path, "rb") as f:
        head = f.read(512)
    suffix = _SUFFIX.get(path.suffix.lower())
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        found = _zip_format(path)
        if found:
            return found
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):  # OLE2: Word / PowerPoint / Excel 97-2003
        return suffix if suffix in ("doc", "ppt") else "ole"
    if head.startswith(b"AT&TFORM"):
        return "djvu"
    lowered = head.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if lowered.startswith((b"<!doctype html", b"<html")):
        return "html"
    return suffix or (path.suffix.lower().lstrip(".") or "unknown")


def parse_file(path: Path, get_ocr: Callable[[], object | None]) -> dict:
    """
    Trích text một file theo định dạng nhận được; không ném lỗi ra ngoài (xem docstring module).

    Args:
        path:    File cần đọc.
        get_ocr: Hàm trả bộ OCR (có .page_text(ảnh)) hoặc None; chỉ gọi khi gặp trang scan (PDF, DJVU).
    """
    from vi_corpus.common.parsers import djvu, office, web
    from vi_corpus.common.pdf_text import extract_pdf

    fmt = detect_format(path)
    out = {"format": fmt, "pages": [], "error": None, "skipped": None, "parser_version": PARSER_VERSION}
    handlers = {"pdf": lambda: extract_pdf(path, get_ocr), "docx": lambda: office.extract_docx(path),
                "pptx": lambda: office.extract_pptx(path), "doc": lambda: office.extract_legacy(path, "docx"),
                "ppt": lambda: office.extract_legacy(path, "pptx"), "djvu": lambda: djvu.extract_djvu(path, get_ocr),
                "html": lambda: web.extract_html(path), "epub": lambda: web.extract_epub(path)}
    if fmt not in handlers:
        out["skipped"] = "unsupported_format"
        return out
    try:
        out["pages"] = handlers[fmt]()
    except MissingTool as exc:
        out["skipped"] = exc.args[0]
    except Exception as exc:  # noqa: BLE001 — một file lỗi không được dừng cả lô
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out
