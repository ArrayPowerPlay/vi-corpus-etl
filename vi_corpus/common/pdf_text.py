"""
Trích text từ PDF, dùng chung cho giáo trình (và VJOL/VISTA sau này).

Mỗi trang đi theo đường rẻ nhất còn cho ra text:
    "text"      : PDF có sẵn lớp chữ Unicode, lấy thẳng.
    "tcvn3"     : PDF gõ bằng font cũ TCVN3 (.VnTime, ...), lớp chữ ra dạng "§iÖn tö";
                  đổi sang Unicode ("Điện tử"). Nhận diện theo cả tài liệu.
    "ocr"       : trang scan (gần như không có chữ nhưng có ảnh), OCR bằng vi_corpus.common.ocr.
    "ocr_vlm"   : trang scan OCR bằng chuỗi engine VLM (vi_corpus.common.ocr_cascade): thêm "tier" (engine được chọn) và
                  "status" ("ok" | "blank" | "flagged"); trang flagged có text rỗng, văn bản bị loại nằm ở "rejected_text".
    "needs_ocr" : trang scan (hoặc trang lỗi font, xem dưới) nhưng lần chạy này không OCR được (tắt OCR / chưa cài nhóm ocr).
    "empty"     : trang trắng.

Trang lỗi font (J-10, chỉ khi extract_pdf(detect_garbled=True), VJOL dùng): lớp chữ toàn chữ Latinh có dấu nhưng sai ("Baỷng, hieọu"),
phát hiện bằng vi_corpus.common.garbled; trang đó (và cả bài nếu >= 30% trang có chữ bị nghi) được OCR lại như trang scan, vẽ lại cả
trang. Mục trang có thêm "reason" ("garbled_text_layer" | "garbled_doc"), "signals" và "garbled_text" (lớp chữ hỏng, giữ để kiểm tra, không dùng);
OCR bị flagged thì trang rỗng, không quay lại dùng chữ hỏng.

Các lỗi thật đã gặp trên giáo trình và được xử lý ở đây:
- Slide xuất từ PowerPoint hay mất dấu cách sau chữ có dấu ("cụthểmà"): lớp chữ không có ký tự
  cách nhưng vẫn có khoảng trống giữa hai chữ, nên page_text chèn lại dấu cách theo khoảng trống,
  đo dọc theo hướng dòng chữ (slide xoay dọc thì đo theo trục y, không phải x).
- Font TCVN3: xem tcvn3_to_unicode. Sách Unicode vẫn hay có bìa/tiêu đề gõ bằng font TCVN3
  (VnArialH, .VnTime...), nên span nào có tên font TCVN3 thì đổi riêng span đó; tài liệu mà
  tỉ lệ ký tự TCVN3 cao (font không mang tên .Vn...) thì đổi toàn bộ.

Ngoài trích text, file này có clean_pages: làm sạch text theo trang (bỏ tiêu đề chạy, số trang,
ghép dòng thành đoạn) để dùng khi ghép các trang thành một văn bản.
"""

import re
from collections.abc import Callable

import pymupdf

from vi_corpus.common.garbled import DEFAULT_THRESHOLDS, Thresholds, page_signals, plan_garbled

# Khoảng trống giữa 2 ký tự lớn hơn tỉ lệ này của cỡ chữ thì coi là có dấu cách.
SPACE_GAP_RATIO = 0.15
# Trang có ít hơn số ký tự này (không tính khoảng trắng) mà có ảnh thì coi là trang scan.
MIN_TEXT_CHARS = 30

# Bảng mã TCVN3 (ABC): ký tự mà lớp chữ PDF trả ra -> chữ Unicode. Chữ hoa có dấu nằm ở font
# riêng (.VnTimeH) dùng chung mã với chữ thường, nên chỉ khôi phục được thành chữ thường.
_TCVN3 = dict(zip(
    "¸µ¶·¹¨¾»¼½Æ©ÊÇÈÉËÐÌÎÏÑªÕÒÓÔÖÝ×ØÜÞãßáâä«èåæçé¬íêëìîóïñòô\xadøõö÷ùýúûüþ®¡¢£¤¥¦§",
    "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĂÂÊÔƠƯĐ",
))
_TCVN3["−"] = "ư"  # PyMuPDF trả mã 0xAD (ư) của font TCVN3 thành dấu trừ U+2212
_TCVN3["μ"] = "à"  # và mã 0xB5 (à) thành chữ Hy Lạp mu U+03BC ("NHμ XUÊT B¶N")
# Tên font TCVN3 (bỏ tiền tố subset "ABCDEF+"): .VnTime, VnArial, VnSouthernH... (không khớp VNI-Times).
_TCVN3_FONT = re.compile(r"^\.?Vn[A-Z]")
# Font TCVN3 chữ hoa có hậu tố H (.VnTimeH, VnArialH, VnBahamasBHBold): mã chữ thường hiện ra chữ hoa.
_TCVN3_UPPER_FONT = re.compile(r"H(?:[,-]?(?:Bold|Italic))*$")
# Ký tự chỉ có trong TCVN3, không xuất hiện trong văn bản tiếng Việt Unicode bình thường.
_TCVN3_MARKERS = set("¸µ¶·¹¨¾»¼½Æ©ÇÈÉËÐÎÏÑªÖ×ØÜÞßä«åæç¬ëîïñö÷øûüþ®§¡¢£¤¥¦")
# Số trang tối thiểu mà một dòng đầu/cuối trang (đã thay chữ số bằng #) lặp lại thì coi là tiêu đề chạy.
HEADER_MIN_PAGES = 3
_DIGITS_ONLY = re.compile(r"\d+")
_SENTENCE_END = (".", ":", "?", "!", ";")
_BULLETS = ("-", "•", "–")
TCVN3_MIN_RATIO = 0.05  # đo trên mẫu: tài liệu TCVN3 ~12%, tài liệu Unicode <= 1,3%


def page_text(page: pymupdf.Page, tcvn3_all: bool = False) -> str:
    """
    Text của một trang theo thứ tự trong PDF, mỗi dòng một dòng, chèn lại dấu cách bị mất.

    Span gõ bằng font TCVN3 (theo tên font) được đổi sang Unicode; `tcvn3_all=True` thì đổi mọi span.
    """
    lines = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            dx, dy = line["dir"]  # hướng dòng chữ: (1, 0) ngang, (0, -1) xoay dọc từ dưới lên

            def along(b):
                """(đầu, cuối) của bbox chiếu lên hướng dòng chữ."""
                p, q = b[0] * dx + b[1] * dy, b[2] * dx + b[3] * dy
                return min(p, q), max(p, q)

            out: list[str] = []
            prev = None
            for span in line["spans"]:
                font = span["font"].split("+")[-1]
                convert = tcvn3_all or _TCVN3_FONT.match(font)
                upper = convert and _TCVN3_UPPER_FONT.search(font)
                for ch in span["chars"]:
                    c = ch["c"]
                    if (prev is not None and not c.isspace() and not prev["c"].isspace()
                            and along(ch["bbox"])[0] - along(prev["bbox"])[1] > SPACE_GAP_RATIO * span["size"]):
                        out.append(" ")
                    if convert:
                        c = tcvn3_to_unicode(c)
                        out.append(c.upper() if upper else c)
                    else:
                        out.append(c)
                    prev = ch
            lines.append("".join(out))
    return "\n".join(lines)


def is_tcvn3(text: str) -> bool:
    """True nếu text trông như lớp chữ của font TCVN3 (tỉ lệ ký tự đặc trưng >= TCVN3_MIN_RATIO)."""
    letters = sum(c.isalpha() for c in text)
    return letters > 0 and sum(c in _TCVN3_MARKERS for c in text) / letters >= TCVN3_MIN_RATIO


def tcvn3_to_unicode(text: str) -> str:
    """Đổi text TCVN3 sang Unicode, vd "M«men tõ: §iÖn tö" -> "Mômen từ: Điện tử"."""
    return "".join(_TCVN3.get(c, c) for c in text)


def extract_pdf(path, get_ocr: Callable[[], object | None], detect_garbled: bool = False,
                garbled_vocab: frozenset[str] | None = None, garbled_th: Thresholds = DEFAULT_THRESHOLDS) -> list[dict]:
    """
    Trích text từng trang của một PDF.

    Args:
        path:    Đường dẫn PDF.
        get_ocr: Hàm trả về bộ OCR hoặc None nếu không OCR được; chỉ được gọi khi gặp trang cần OCR (scan hoặc lỗi font), để tài
                 liệu có sẵn chữ không phải nạp mô hình OCR. Bộ OCR có .page_text(ảnh PIL) -> str (OCR tuần tự từng trang), hoặc
                 .ocr_pages(các trang pymupdf, render=cờ từng trang) -> danh sách {"text", "method", ...} theo thứ tự (OCR song
                 song cả lô; render=True là vẽ lại cả trang thay vì lấy ảnh nhúng).
        detect_garbled: True thì trang có lớp chữ hỏng do font (vi_corpus.common.garbled) cũng được OCR lại (J-10); mặc định tắt,
                 nguồn khác (giáo trình) chạy y như trước.
        garbled_vocab: Tập âm tiết cho tín hiệu "unknown" của bộ dò (None = bỏ tín hiệu này, J-16).
        garbled_th: Ngưỡng các tín hiệu của bộ dò.

    Returns:
        Danh sách {"text": ..., "method": ...} theo thứ tự trang (method xem đầu file). Trang lỗi font có thêm "reason",
        "signals" (tín hiệu nào vượt ngưỡng; rỗng với trang chỉ bị kéo theo cả bài) và "garbled_text".

    Raises:
        ValueError: PDF bị cắt cụt / hỏng (PyMuPDF phải tự sửa khi mở) hoặc có mật khẩu.
    """
    from vi_corpus.common.ocr import page_image

    with pymupdf.open(path) as doc:
        if doc.is_repaired:
            raise ValueError("PDF hỏng hoặc tải thiếu")
        if doc.needs_pass:
            raise ValueError("PDF có mật khẩu")
        texts = [page_text(page) for page in doc]
        tcvn3 = is_tcvn3("".join(texts))
        if tcvn3:  # font TCVN3 không mang tên .Vn...: đổi mọi span (đọc lại, tránh đổi 2 lần)
            texts = [page_text(page, tcvn3_all=True) for page in doc]
        pages: list[dict | None] = []
        scan_idx: list[int] = []
        layer: list[str | None] = []  # text lớp chữ của trang chữ (None cho trang scan / trang trắng), để dò chữ hỏng
        for i, (page, text) in enumerate(zip(doc, texts)):
            if len("".join(text.split())) >= MIN_TEXT_CHARS or not page.get_images():
                if not text.strip():
                    pages.append({"text": "", "method": "empty"})
                    layer.append(None)
                else:
                    pages.append({"text": text, "method": "tcvn3" if tcvn3 else "text"})
                    layer.append(text)
                continue
            pages.append(None)  # trang scan: điền sau khi OCR
            scan_idx.append(i)
            layer.append(None)
        reasons: dict[int, str] = {}  # trang chữ bị lỗi font -> lý do (J-10), cũng đi OCR như trang scan
        if detect_garbled:
            reasons = {i: r for i, r in enumerate(plan_garbled(layer, garbled_th, vocab=garbled_vocab)) if r}
            for i in reasons:
                pages[i] = None
        todo = sorted([*scan_idx, *reasons])
        if todo:
            ocr = get_ocr()
            if ocr is None:
                for i in todo:
                    pages[i] = {"text": texts[i], "method": "needs_ocr", **({"reason": reasons[i]} if i in reasons else {})}
            else:
                if hasattr(ocr, "ocr_pages"):  # OCR nhận cả lô trang để gửi song song (CascadePageOcr)
                    kwargs = {"render": [i in reasons for i in todo]} if reasons else {}
                    results = ocr.ocr_pages([doc[i] for i in todo], **kwargs)
                else:
                    results = [{"text": ocr.page_text(page_image(doc[i], render=i in reasons)), "method": "ocr"} for i in todo]
                for i, entry in zip(todo, results):
                    pages[i] = ({**entry, "reason": reasons[i], "signals": page_signals(texts[i], garbled_th, garbled_vocab),
                                 "garbled_text": texts[i]} if i in reasons else entry)
        return pages


def clean_pages(pages: list[str], slides: bool = False, stats: dict | None = None) -> str:
    """
    Làm sạch text theo trang rồi ghép mọi trang thành một văn bản (đoạn cách nhau bằng dòng trống).

    1. Bỏ tiêu đề/chân trang chạy: dòng nằm trong 2 dòng không trống đầu hoặc 2 dòng cuối của trang,
       sau khi thay chữ số bằng "#" mà lặp ở >= HEADER_MIN_PAGES trang.
    2. Bỏ dòng chỉ có chữ số (số trang lẻ).
    3. Ghép dòng thành đoạn (xuyên qua ranh giới trang): chỉ xuống đoạn khi dòng trước kết thúc bằng
       . : ? ! ; và dòng sau bắt đầu bằng chữ hoa hoặc gạch đầu dòng; từ tiếng Anh bị gạch nối cuối
       dòng ("exam-" + "ple") thì nối liền. Nội dung không bị sửa gì khác.

    Args:
        pages:      Text từng trang, mỗi dòng một dòng.
        slides:     True cho slide (pptx): bỏ bước 1 (slide lặp cấu trúc như "Bài 1", "Bài 2" nên dễ bị bỏ nhầm)
                    và ở bước 3 mỗi dòng là một đoạn (tiêu đề, gạch đầu dòng không có dấu câu).
        stats:      Nếu có, cộng số dòng bị bỏ ở bước 1 + 2 vào stats["lines_removed"] (D-04 mục B).
    """
    page_lines = [[ln.strip() for ln in text.splitlines() if ln.strip()] for text in pages]

    def edge_idx(lines: list[str]) -> set[int]:
        """Chỉ số các dòng thuộc 2 dòng đầu / 2 dòng cuối của trang."""
        n = len(lines)
        return set(range(min(2, n))) | set(range(max(n - 2, 0), n))

    def key(line: str) -> str:
        """Khoá so sánh tiêu đề chạy: chữ số -> #."""
        return _DIGITS_ONLY.sub("#", line)

    seen: dict[str, int] = {}
    for lines in [] if slides else page_lines:
        for k in {key(lines[i]) for i in edge_idx(lines)}:
            seen[k] = seen.get(k, 0) + 1

    paragraphs: list[str] = []
    for lines in page_lines:
        edges = edge_idx(lines)
        for i, line in enumerate(lines):
            if line.isdigit() or (not slides and i in edges and seen[key(line)] >= HEADER_MIN_PAGES):
                if stats is not None:
                    stats["lines_removed"] = stats.get("lines_removed", 0) + 1
                continue
            if not paragraphs or slides:
                paragraphs.append(line)
                continue
            prev = paragraphs[-1]
            if prev.endswith(_SENTENCE_END) and (line[0].isupper() or line.startswith(_BULLETS)):
                paragraphs.append(line)
            elif (prev.endswith("-") and len(prev) > 1 and prev[-2].isascii() and prev[-2].isalpha()
                  and line[0].isascii() and line[0].islower()):
                paragraphs[-1] = prev[:-1] + line
            else:
                paragraphs[-1] = prev + " " + line
    return "\n\n".join(paragraphs)
