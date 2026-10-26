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
Trang không có lớp chữ và không có ảnh nhúng chưa chắc là trang trắng (ảnh nằm trong form XObject, trang vẽ bằng nét): với extract_pdf(rescue_empty=True)
(VJOL dùng, J-33 B) trang như vậy được vẽ ra thử, vẽ ra có nội dung thì OCR như trang lỗi font (reason "no_text_layer"), vẽ ra trắng thì "empty" kèm reason "blank_page".

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
from collections.abc import Callable, Sequence

import pymupdf

from vi_corpus.common.garbled import (DEFAULT_THRESHOLDS, REASON_BLANK, REASON_FONT_GROUP, REASON_NO_TEXT, REASON_PRODUCER, Thresholds, page_signals,
                                      plan_garbled)
from vi_corpus.common.ocr_noise import strip_vlm_noise

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
# Kiểm trang có nội dung hay không (J-33 B): vẽ ra ảnh xám tỉ lệ PROBE_ZOOM, pixel tối hơn PROBE_DARK chiếm >= PROBE_MIN_INK thì có nội dung.
PROBE_ZOOM = 0.5
PROBE_DARK = 230
PROBE_MIN_INK = 0.001


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


def _producer_matches(meta: dict | None, rules: Sequence[tuple[str, str]]) -> bool:
    """True nếu metadata PDF (producer, creator) khớp một quy tắc (đầu producer, chuỗi con creator); quy tắc rỗng không khớp gì."""
    producer = ((meta or {}).get("producer") or "").strip()
    creator = ((meta or {}).get("creator") or "").strip()
    return any(producer.startswith(p) and c in creator for p, c in rules)


def _group_matches(meta: dict | None, groups: Sequence[tuple[str, str]]) -> bool:
    """True nếu (producer, creator) của PDF BẰNG một nhóm trong `groups` (so khớp nguyên, khác _producer_matches); nhóm có cả hai trường rỗng bị bỏ vì sẽ khớp mọi PDF thiếu metadata."""
    key = (((meta or {}).get("producer") or "").strip(), ((meta or {}).get("creator") or "").strip())
    return key != ("", "") and key in {(p, c) for p, c in groups}


def _scrub_vlm(entry: dict) -> dict:
    """
    Lọc câu "nói về chính ảnh" của OCR VLM khỏi một kết quả OCR (vi_corpus.common.ocr_noise, J-33 C); không sửa `entry` (có thể là mục của bộ nhớ đệm).
    Trang chỉ còn câu đó thì text rỗng và status "blank"; số câu đã bỏ ghi ở "vlm_noise_removed".
    """
    if entry.get("method") != "ocr_vlm" or not entry.get("text"):
        return entry
    text, n = strip_vlm_noise(entry["text"])
    if not n:
        return entry
    out = {**entry, "text": text, "vlm_noise_removed": n}
    if not text.strip() and out.get("status") == "ok":
        out["status"] = "blank"
    return out


def page_has_ink(page: pymupdf.Page) -> bool:
    """
    True nếu trang vẽ ra có nội dung: có ảnh thật sự được vẽ lên trang (kể cả ảnh nằm trong form XObject mà page.get_images() không thấy) hoặc ảnh xám thu nhỏ
    có đủ pixel tối (PROBE_*). Dùng cho trang không có lớp chữ để phân biệt trang ảnh / nét vẽ cần OCR với trang trắng (J-33 B).
    """
    if page.get_image_info():
        return True
    pix = page.get_pixmap(matrix=pymupdf.Matrix(PROBE_ZOOM, PROBE_ZOOM), colorspace=pymupdf.csGRAY, alpha=False)
    samples = pix.samples
    return bool(samples) and sum(b < PROBE_DARK for b in samples) / len(samples) >= PROBE_MIN_INK


def extract_pdf(path, get_ocr: Callable[[], object | None], detect_garbled: bool = False,
                garbled_vocab: frozenset[str] | None = None, garbled_th: Thresholds = DEFAULT_THRESHOLDS,
                garbled_unaccented: frozenset[str] | None = None, ocr_cache=None,
                ocr_all_producers: Sequence[tuple[str, str]] = (), rescue_empty: bool = False,
                ocr_all_groups: Sequence[tuple[str, str]] = (),
                ocr_all_if: Callable[[Sequence[str | None]], str | None] | None = None) -> list[dict]:
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
        garbled_unaccented: Danh sách từ không dấu cho tín hiệu "nodiac" (None = garbled.UNACCENTED_WORDS, J-25 A).
        ocr_cache: vi_corpus.common.ocr_cache.OcrCache của PDF này (J-25 Bước 2): trang đã có kết quả OCR trong cache thì không OCR lại và get_ocr
                 không được gọi nếu mọi trang cần OCR đều có sẵn; kết quả OCR mới được ghi vào cache (save) khi xong cả lô. None = không dùng cache.
        ocr_all_producers: Các cặp (đầu của `producer`, chuỗi con của `creator`) trong metadata PDF; PDF khớp một cặp thì MỌI trang có chữ đi OCR (lý do
                 REASON_PRODUCER), vì lớp chữ do phần mềm OCR đời cũ tạo ra (J-29, D2). Chỉ có tác dụng khi detect_garbled bật.
        ocr_all_groups: Các cặp (producer, creator) của nhóm font xấu, so khớp NGUYÊN (không phải đầu / chuỗi con như ocr_all_producers), do phép thử scripts/vjol/font_canary.py tìm ra (J-42 (4)); lý do REASON_FONT_GROUP.
        ocr_all_if: Hàm nhận lớp chữ từng trang (None cho trang không có lớp chữ) và trả về một mã lý do nếu CẢ BÀI phải OCR lại, None nếu không (J-42 (3): vi_corpus.vjol.reocr.reocr_reason);
                 chỉ gọi khi bài chưa bị đưa đi OCR cả bài bởi producer / nhóm font. Chỉ có tác dụng khi detect_garbled bật.
        rescue_empty: True thì trang không có chữ và không có ảnh nhúng được vẽ ra thử (page_has_ink): có nội dung thì OCR (REASON_NO_TEXT), trắng thì
                 method "empty" kèm reason REASON_BLANK (J-33 B). Mặc định tắt, giáo trình chạy y như trước.

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
        no_text: set[int] = set()  # trang không có lớp chữ nhưng vẽ ra có nội dung (rescue_empty)
        scan_idx: list[int] = []
        layer: list[str | None] = []  # text lớp chữ của trang chữ (None cho trang scan / trang trắng), để dò chữ hỏng
        for i, (page, text) in enumerate(zip(doc, texts)):
            if len("".join(text.split())) >= MIN_TEXT_CHARS or not page.get_images():
                if not text.strip():
                    if rescue_empty and page_has_ink(page):  # J-33 B: có nội dung nhưng không có lớp chữ nào: OCR vẽ lại cả trang
                        pages.append(None)
                        no_text.add(i)
                    else:
                        pages.append({"text": "", "method": "empty", **({"reason": REASON_BLANK} if rescue_empty else {})})
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
            reasons = {i: r for i, r in enumerate(plan_garbled(layer, garbled_th, vocab=garbled_vocab, unaccented=garbled_unaccented)) if r}
            whole_doc: str | None = None
            if _producer_matches(doc.metadata, ocr_all_producers):
                whole_doc = REASON_PRODUCER
            elif _group_matches(doc.metadata, ocr_all_groups):
                whole_doc = REASON_FONT_GROUP
            elif ocr_all_if is not None:
                whole_doc = ocr_all_if(layer)
            if whole_doc:
                reasons.update({i: reasons.get(i, whole_doc) for i, t in enumerate(layer) if t is not None})
            for i in reasons:
                pages[i] = None
        reasons.update({i: REASON_NO_TEXT for i in no_text})
        todo = sorted([*scan_idx, *reasons])
        if todo:
            cached = {i: ocr_cache.get(i, i in reasons) for i in todo} if ocr_cache is not None else {}
            miss = [i for i in todo if cached.get(i) is None]
            fresh: dict[int, dict] = {}
            ocr = get_ocr() if miss else None
            if miss and ocr is not None:
                if hasattr(ocr, "ocr_pages"):  # OCR nhận cả lô trang để gửi song song (CascadePageOcr)
                    kwargs = {"render": [i in reasons for i in miss]} if reasons else {}
                    results = ocr.ocr_pages([doc[i] for i in miss], **kwargs)
                else:
                    results = [{"text": ocr.page_text(page_image(doc[i], render=i in reasons)), "method": "ocr"} for i in miss]
                fresh = dict(zip(miss, results))
                if ocr_cache is not None and hasattr(ocr, "ocr_pages"):  # chỉ bộ OCR chuỗi engine có chữ ký đúng với cache
                    for i, entry in fresh.items():
                        ocr_cache.put(i, i in reasons, entry)
                    ocr_cache.save()
            for i in todo:
                entry = cached.get(i) or fresh.get(i)
                if entry is None:  # không OCR được (tắt OCR / chưa cài nhóm ocr) và cache cũng không có
                    pages[i] = {"text": texts[i], "method": "needs_ocr", **({"reason": reasons[i]} if i in reasons else {})}
                elif i in no_text:
                    pages[i] = {**_scrub_vlm(entry), "reason": REASON_NO_TEXT}
                else:
                    pages[i] = ({**_scrub_vlm(entry), "reason": reasons[i], "signals": page_signals(texts[i], garbled_th, garbled_vocab, garbled_unaccented),
                                 "garbled_text": texts[i]} if i in reasons else _scrub_vlm(entry))
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
