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

J-25 (sửa theo nguyên nhân gốc, chạy lúc sinh bản ghi nên không phải trích lại):
- C: strip_running_headers bỏ tiêu đề / chân trang LẶP giữa các trang của cùng PDF, kể cả khi dính đầu dòng thân bài ("Tạp chí Dân tộc học số 3 - 2005 Nội dung ..."),
  không phụ thuộc tên tạp chí trong metadata (vốn là tên tiếng Anh nên không khớp tiêu đề tiếng Việt in trên trang). Regex theo metadata giữ làm phần bổ sung.
- E: tiêu đề tham khảo chấp nhận dạng song ngữ "TÀI LIỆU THAM KHẢO/ REFERENCES"; strip_trailing_references đi ngược từ cuối bài để cắt khối mục tham khảo không có tiêu đề
  (nhận mục theo mẫu đầu dòng, vi_corpus.common.references).
J-33 (chạy lúc sinh bản ghi, không phải trích lại):
- A: restore_u_in_pages khôi phục "ư" cho trang font gạch nối ("tr-ờng" -> "trường"), kiểm bằng tập âm tiết (vi_corpus.vjol.hyphen_fix).
- C: strip_vlm_noise bỏ câu "There is no text in the image" của OCR VLM (vi_corpus.common.ocr_noise).
- E (J-35): GENERIC_HEADER bỏ tiêu đề trang tiếng Việt dạng "Tạp chí X số N - năm" ở mọi vị trí trong dòng (kể cả số trang đứng sau), vì tiêu đề nằm ngoài 3 dòng đầu / cuối của trang thì strip_running_headers không học được.
- E: strip_running_headers học tiêu đề từ các trang lặp rồi bỏ nó ở MỌI vị trí trong dòng (đầu trang đầu bài không lặp, giữa đoạn do thứ tự khối chữ), không chỉ ở đầu / cuối dòng.
- F: drop_front_matter nhận trang ban biên tập / bìa mục lục (chức danh, hội đồng biên tập, ISSN) ở vài trang đầu và trang cuối.
J-37 (C), chạy lúc sinh bản ghi:
- strip_front_headers cắt tiêu đề tạp chí ở ĐẦU dòng đầu bài (FRONT_SCAN_LINES dòng không rỗng của trang đầu; TOP_SCAN_LINES dòng đầu mỗi trang sau): "Tạp chí ... Số N - năm" mọi kiểu
  viết (kể cả chữ hoa lẫn thường, "Tập N, Số M (năm): tr", ISSN, trang, địa chỉ web), "Nghiên cứu Tôn giáo. Số 6 - 2001 35", ". Số 3 - 2005 45", "SỐ 12 (206)-2012 NGÔN NGỮ & ĐỜI SỐNG 27",
  "© Học viện Ngân hàng ISSN 1859 - 011X", "Vietnam J. Agri. Sci. 2016, Vol. 14, No. 6: 843-851", "http://... Bài báo khoa học". Câu trích dẫn ("Tạp chí Nghiên cứu Tôn giáo, số 4/2000, tr. 5")
  không bị cắt: dạng có dấu phẩy ngay trước "số" chỉ nhận ở trang đầu và không có "tr." mà thiếu ISSN.
J-42 (6), P8 (cắt mức dòng thay vì loại cả đoạn, chạy lúc sinh bản ghi):
- strip_mid_headers cắt tiêu đề tạp chí ở MỌI vị trí trong dòng (find_journal_headers: "Tạp chí X số N - năm", ISSN, số trang; câu trích dẫn không bị cắt), phần còn lại của dòng giữ nguyên (khác dòng chỉ gồm tiêu đề thì bỏ);
- strip_junk_lines bỏ dòng rác (>= JUNK_LINE_MIN_TOKENS từ, >= JUNK_LINE_RATIO là từ rác của defects.gibberish_ratio và không có từ Latinh >= 3 chữ cái): "/, N9 = < 0 e H ; :"; một dòng rác không còn làm mất cả đoạn 1.000 token sạch.
  Rủi ro đã biết: công thức dàn phẳng toàn ký hiệu đơn cũng thoả điều kiện; số dòng bỏ ghi ở meta.repairs để đo trước khi tin.
- F (J-25): dòng khuôn mẫu "ngày + động từ + ngày tháng" nói chung ("Ngày phản biện: 12/05/2015"), số điện thoại có mã nước không có dấu + ("84-983226880") có kiểm đầu số
  và loại số liệu bảng.
"""

import math
import re
import unicodedata

from vi_corpus.common.defects import gibberish_ratio
from vi_corpus.common.ocr_noise import strip_vlm_noise
from vi_corpus.common.references import is_reference_entry, reference_flags
from vi_corpus.vjol.hyphen_fix import restore_u_in_pages

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
# Tiêu đề / chân trang lặp (J-25 C, điểm khởi đầu chưa đo): tiền tố hoặc hậu tố >= 3 từ lặp ở >= 30% số trang (và >= 3 trang) của bài
HEADER_MIN_PAGES = 3
HEADER_MIN_SHARE = 0.3
HEADER_MIN_TOKENS = 3
HEADER_MAX_TOKENS = 14
HEADER_SCAN_LINES = 3
HEADER_ROUNDS = 4
# Khối tham khảo không tiêu đề ở cuối bài (J-25 E): >= 6 mục, >= 60% dòng của khối thuộc phần tham khảo, khối bắt đầu sau 50% số dòng của bài
TAIL_REF_MIN_ENTRIES = 6
TAIL_REF_MIN_SHARE = 0.6
TAIL_REF_MIN_POS = 0.5
TOC_LEADER_FRAC = 0.5  # trang có >= tỉ lệ này số dòng kết thúc bằng "......... 12" là trang mục lục
TOC_MIN_LINES = 5
# Trang ban biên tập / bìa mục lục (J-33 F): chứa >= EDITORIAL_MIN_MARKERS loại chức danh / mục khác nhau (EDITORIAL_MARKERS) và ngắn (<= EDITORIAL_MAX_WORDS từ), ở TOC_SCAN_PAGES trang đầu
# hoặc EDITORIAL_TAIL_PAGES trang cuối. Điểm khởi đầu chưa đo; trang bài thật dài hơn nhiều ngưỡng từ nên không bị cắt nhầm.
EDITORIAL_MARKERS = (
    r"tổng biên tập", r"hội đồng biên tập", r"ban biên tập", r"hội đồng khoa học", r"thư ký (?:tòa soạn|toà soạn|hội đồng|biên tập|tạp chí)", r"trị sự",
    r"(?:tòa|toà) soạn", r"giấy phép xuất bản", r"chủ tịch hội đồng", r"(?:uỷ|ủy) viên", r"editor[- ]in[- ]chief", r"editorial board", r"managing editor", r"issn\b",
    r"chịu trách nhiệm xuất bản", r"cơ quan chủ quản",
)
_EDITORIAL = tuple(re.compile(rf"(?<!\w){m}", re.IGNORECASE) for m in EDITORIAL_MARKERS)
EDITORIAL_MIN_MARKERS = 3
EDITORIAL_MAX_WORDS = 350
EDITORIAL_TAIL_PAGES = 1
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

# Tiêu đề trang tạp chí tiếng Việt in sẵn trên trang, không cần tên trong metadata (J-35): "Tạp chí Dân tộc học số 3 - 2004", "Tạp chí Dân tộc học - Số 2 - 1994", có thể kèm số trang đứng ngay sau năm.
# Chỉ khớp dạng "số N - năm" (gạch nối giữa số và năm); câu trích dẫn "Tạp chí Dân chủ và Pháp luật Kỳ 2 (Số 407), tháng 6/2024" hay "Tạp chí X, số 3 (2005)" có dấu câu nên không khớp.
GENERIC_HEADER = re.compile(r"Tạp chí\s+[^\n.,;:()\[\]]{3,60}?\s*(?:[-–]\s*)?số\s*\d+\s*[-–]\s*(?:19|20)\d{2}(?:\s+\d{1,3}(?=\s|$))?", re.IGNORECASE)
# Dạng viết hoa toàn bộ ("TẠP CHÍ THÔNG TIN KHOA HỌC & CÔNG NGHỆ QUẢNG BÌNH - SỐ 5/2016"): phân biệt hoa thường để câu trích dẫn không khớp.
GENERIC_HEADER_UPPER = re.compile(r"TẠP CHÍ\s+[A-ZÀ-Ỹ&\s]{5,80}?\s*(?:[-–]\s*)?SỐ\s*\d+\s*[/-]\s*(?:19|20)\d{2}(?:\s+\d{1,3}(?=\s|$))?")
# Tiêu đề tạp chí ở ĐẦU bài (J-37 C). Chỉ xét FRONT_SCAN_LINES dòng không rỗng đầu của trang đầu và TOP_SCAN_LINES dòng đầu của các trang sau, và chỉ khi dòng BẮT ĐẦU bằng tiêu đề.
FRONT_SCAN_LINES = 8
TOP_SCAN_LINES = 2
_HEAD_LEAD = re.compile(r"^[\d\s.©•*|:]*(?:BÀI BÁO KHOA HỌC\s+)?(?:Tạp chí|TẠP CHÍ)\s+")
_HEAD_NUM = r"[\dIVXLC]+"
_HEAD_MARK = rf"(?:số|tập|quyển|năm thứ|tháng|vol\.?|no\.?|t\.)\s*{_HEAD_NUM}(?:\s*[(/&.,-]\s*\d+\)?)*"
_HEAD_ISSN = r"ISSN[:\s]*\d{4}\s*-\s*\d{3}[\dXx]"
_HEAD_FIRST = re.compile(rf"\b(?:{_HEAD_MARK}|{_HEAD_ISSN})", re.IGNORECASE)
_HEAD_TOKEN = re.compile(rf"(?:{_HEAD_MARK}|{_HEAD_ISSN}|(?:19|20)\d{{2}}|\([^)\n]{{1,30}}\)|\d{{1,4}}\s*[-–]\s*\d{{1,4}}|(?:tr|trang|pp)\.?\s*\d+(?:\s*[-–]\s*\d+)?|https?://\S+|www\.\S+|©[^\W\d_][^\n]{{0,40}}?(?=ISSN))", re.IGNORECASE)
_HEAD_GAP = re.compile(r"[\s,.:;|/&–—-]*")
_HEAD_YEAR = re.compile(r"(?:19|20)\d{2}")
_HEAD_PAGE = re.compile(r"\d{1,3}(?=\s|$)")
_HEAD_LEFT = re.compile(r"^(?:\s|[,.:;|–—-]|o0o)*")
_YEAR = r"(?:19|20)\d{2}"
# Các kiểu tiêu đề không bắt đầu bằng "Tạp chí" (mỗi mẫu neo ở đầu dòng, chỉ dùng ở trang đầu).
FRONT_PATTERNS = (
    re.compile(rf"^[\W\d]{{0,6}}(?:Số|SỐ)\s*\d+\s*(?:\(\d+\))?\s*[-–]\s*{_YEAR}(?:\s+[A-ZÀ-Ỹ&]+(?:\s+[A-ZÀ-Ỹ&]+){{0,5}}\s+\d{{1,3}}(?=\s|$)|\s+\d{{1,3}}(?=\s|$))?"),  # ". Số 3 - 2005 45", "SỐ 12 (206)-2012 NGÔN NGỮ & ĐỜI SỐNG 27"
    re.compile(rf"^Nghiên cứu\s+[^\W\d_]+(?:\s+[^\W\d_]+){{0,3}}\s*[.,]\s*[Ss]ố\s*\d+\s*(?:\(\d+\))?\s*[-–]\s*{_YEAR}(?:\s+\d{{1,3}}(?=\s|$))?"),  # "Nghiên cứu Tôn giáo. Số 6 - 2001 35"
    re.compile(r"^©[^\n]{3,60}?ISSN[:\s]*\d{4}\s*-\s*\d{3}[\dXx]"),  # "© Học viện Ngân hàng ISSN 1859 - 011X"
    re.compile(rf"^(?:[A-Z&][\w.]*\s+){{1,6}}{_YEAR},\s*Vol\.\s*\d+,\s*No\.\s*\d+\s*:\s*\d+\s*[-–]\s*\d+"),  # "Vietnam J. Agri. Sci. 2016, Vol. 14, No. 6: 843-851", "J. Sci. & Devel. 2015, Vol. 13, No. 8: 1488-1496"
    re.compile(rf"^(?:(?:[A-Z&][\w.]*|of|and)\s+){{1,9}}ISSN[:\s]*\d{{4}}\s*-\s*\d{{3}}[\dXx]\s+[A-Z]+\s+{_YEAR};\s*\d+\(\d+\)\s*:\s*\d+\s*-\s*\d+(?:\s+https?://\S+)?"),  # "Vietnam Journal of Agricultural Sciences ISSN 2588-1299 VJAS 2018; 1(4): 289-304 https://doi.org/..."
    re.compile(rf"^(?:(?:[A-Z&][\w.]*|of|and)\s+){{1,9}}Vol\.\s*\d+\s*\({_YEAR}\),\s*pp\.\s*\d+\s*-\s*\d+"),  # "Journal of Science of Lac Hong University Vol. 5 (2016), pp. 19-24"
    re.compile(r"^(?:https?://|www\.)\S+(?:\s+Bài báo khoa học)?"),  # "http://tapchikttv.vn/ Bài báo khoa học"
    re.compile(r"^Tạp chí Khoa học (?:ĐHQGHN|DHQGHN|Đại học Quốc gia Hà Nội)\s*,"),  # "Tạp chí Khoa học ĐHQGHN," dính đầu đoạn
)


def _journal_header_end(line: str, front: bool) -> int | None:
    """
    Vị trí kết thúc tiêu đề tạp chí nếu `line` BẮT ĐẦU bằng "Tạp chí <tên> <số / tập / ISSN ... năm ...>" (J-37 C), None nếu không.

    Tiêu đề = "Tạp chí" + tên (tối đa 110 ký tự) + một chuỗi liền các mục (số N, tập N, tháng N, năm, "(trang)", "ISSN ...", "tr. 22-31", địa chỉ web) cách nhau tối đa dấu câu; phải có năm
    (kể cả năm trong tên) hoặc ISSN thì mới cắt ("Tạp chí X số 3 ..." không năm thì giữ). Dạng có dấu phẩy ngay trước mục đầu ("Tạp chí X, số 4/2000, tr. 5") là câu trích dẫn: chỉ nhận ở
    trang đầu (`front`) và khi không có "tr." hoặc có ISSN.
    """
    lead = _HEAD_LEAD.match(line)
    if not lead:
        return None
    window = line[lead.end():lead.end() + 160]
    first = _HEAD_FIRST.search(window)
    if not first or first.start() < 3 or first.start() > 110:
        return None
    end = first.end()
    while True:
        gap = _HEAD_GAP.match(window, end)
        tok = _HEAD_TOKEN.match(window, gap.end())
        if not tok and _HEAD_YEAR.search(window[max(0, end - 6):end]):  # số trang in sau năm: "Số 6 – 2005 47"
            tok = _HEAD_PAGE.match(window, gap.end())
        if not tok or tok.end() == end:
            break
        end = tok.end()
    run = window[first.start():end]
    has_issn = bool(re.search(_HEAD_ISSN, run, re.IGNORECASE))
    if not (_HEAD_YEAR.search(window[:end]) or has_issn):  # năm có thể nằm trong tên ("... Nam 2016, tập 14, số 6")
        return None
    if re.search(r"[,;]\s*$", window[:first.start()]):
        if not front or (re.search(r"\b(?:tr|trang|pp)\.?\s*\d", run, re.IGNORECASE) and not has_issn):
            return None
    return lead.end() + end


_JOURNAL_WORD = re.compile(r"(?:Tạp chí|TẠP CHÍ)\s+")


def find_journal_headers(text: str) -> list[tuple[int, int]]:
    """
    Vị trí các tiêu đề tạp chí ("Tạp chí <tên> số N - năm", ISSN, số trang ...) ở BẤT KỲ chỗ nào trong `text` (J-42 (1c)), theo quy tắc của `_journal_header_end` ở chế độ không phải trang đầu:
    câu trích dẫn có dấu phẩy ngay trước "số" ("Tạp chí Nghiên cứu Tôn giáo, số 8, 2008") và cụm viết thường giữa câu ("trên tạp chí ...") không được tính.

    Returns:
        [(đầu, cuối)] theo thứ tự xuất hiện.
    """
    spans = []
    for m in _JOURNAL_WORD.finditer(text):
        end = _journal_header_end(text[m.start():], front=False)
        if end is not None:
            spans.append((m.start(), m.start() + end))
    return spans


JUNK_LINE_MIN_TOKENS = 8  # dòng rác phải có ít nhất ngần này từ (không tính số / dấu câu thuần)
JUNK_LINE_RATIO = 0.85  # và tỷ lệ từ rác (gibberish_ratio) từ mức này
_LONG_WORD = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]{3,}")


def strip_mid_headers(pages: list[str]) -> tuple[list[str], int]:
    """
    Cắt tiêu đề tạp chí dính ở bất kỳ chỗ nào trong dòng (J-42 (6), find_journal_headers); dòng chỉ còn khoảng trắng / dấu câu thì bỏ. Lặp tối đa 3 lượt cho dòng có nhiều tiêu đề.

    Returns:
        (các trang đã cắt, số chỗ đã cắt).
    """
    out, total = [], 0
    for text in pages:
        lines = []
        for line in text.splitlines():
            cut = 0
            for _ in range(3):
                spans = find_journal_headers(line)
                if not spans:
                    break
                a, b = spans[0]
                line = re.sub(r"^[\s.,;:|–—-]+", "", re.sub(r"[ \t]{2,}", " ", line[:a] + " " + line[b:]).strip())
                cut += 1
            total += cut
            if cut and not line:  # dòng chỉ gồm tiêu đề: bỏ (dòng trống sẵn có vẫn giữ)
                continue
            lines.append(line)
        out.append("\n".join(lines))
    return out, total


def is_junk_line(line: str) -> bool:
    """True nếu `line` là dòng rác (J-42 (6)): >= JUNK_LINE_MIN_TOKENS từ, >= JUNK_LINE_RATIO là từ rác và không có từ Latinh nào >= 3 chữ cái."""
    ratio, n = gibberish_ratio(line)
    return n >= JUNK_LINE_MIN_TOKENS and ratio >= JUNK_LINE_RATIO and not _LONG_WORD.search(unicodedata.normalize("NFC", line))


def strip_junk_lines(pages: list[str]) -> tuple[list[str], int]:
    """Bỏ các dòng rác (is_junk_line) khỏi các trang, giữ nguyên số trang và dòng trống; trả về (các trang, số dòng đã bỏ)."""
    out, total = [], 0
    for text in pages:
        kept = []
        for line in text.splitlines():
            if line.strip() and is_junk_line(line):
                total += 1
                continue
            kept.append(line)
        out.append("\n".join(kept))
    return out, total


def _strip_front_line(line: str, front: bool) -> tuple[str, int]:
    """Cắt tiêu đề ở đầu một dòng (lặp tối đa 4 lượt cho tiêu đề nhiều mảnh); trả về (dòng còn lại, số lần cắt)."""
    cut = 0
    for _ in range(4):
        end = _journal_header_end(line, front)
        if end is None and front:
            end = next((m.end() for m in (p.match(line) for p in FRONT_PATTERNS) if m), None)
        if end is None:
            break
        line = _HEAD_LEFT.sub("", line[end:], count=1)
        cut += 1
        if not line.strip():
            break
    return line, cut


def strip_front_headers(pages: list[str]) -> tuple[list[str], int]:
    """
    Cắt tiêu đề tạp chí / số / ISSN ở đầu dòng đầu bài (J-37 C): FRONT_SCAN_LINES dòng không rỗng đầu của trang có chữ đầu tiên, TOP_SCAN_LINES dòng đầu của mỗi trang sau. Dòng chỉ gồm tiêu đề thì bị bỏ.
    Trang sau chỉ nhận kiểu "Tạp chí ..." (không có dạng phẩy và FRONT_PATTERNS) vì dòng đầu trang sau có thể là câu văn.

    Returns:
        (các trang đã cắt, số chỗ đã cắt).
    """
    out, total, first_seen = [], 0, False
    for text in pages:
        lines = text.splitlines()
        front = not first_seen and bool(text.strip())
        first_seen = first_seen or bool(text.strip())
        scan = FRONT_SCAN_LINES if front else TOP_SCAN_LINES
        drop, seen = set(), 0
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            if seen >= scan:
                break
            seen += 1
            lines[i], n = _strip_front_line(line, front)
            total += n
            if n and not re.search(r"\w", lines[i]):
                drop.add(i)
        out.append("\n".join(ln for i, ln in enumerate(lines) if i not in drop))
    return out, total


AUTHOR_BLOCK_LINES = 40  # khối tác giả chỉ tìm trong ngần ấy dòng không rỗng đầu bài
# dòng khuôn mẫu của tạp chí; dòng ngắn (<= BOILER_MAX_WORDS từ) mới bị bỏ để câu văn nhắc tới "email" không mất
BOILER_MAX_WORDS = 25
AUTHOR_LINE_MAX_WORDS = 12  # dòng của khối tác giả (tên, cơ quan, email) ngắn; câu văn dài có email thì chỉ che email
_DATE = r"\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{2,4}"
_BOILER = re.compile(r"^\W*(?:ngày\s+(?:nhận|duyệt|gửi|chấp nhận|đăng|sửa|phản biện|hoàn thiện|chỉnh sửa|xuất bản|phản hồi)\b|"
                     rf"ngày\s+(?:\S+\s+){{0,4}}?[:：]?\s*{_DATE}|"  # "Ngày <vài từ> [:] dd/mm/yyyy" nói chung (J-25 F)
                     r"received\b|accepted\b|revised\b|submitted\b|published\b|date\s+(?:received|accepted|revised)\b|article history\b|"
                     r"(?:e-?mail|thư điện tử|(?:số\s+)?điện thoại|tel|phone|fax)\s*[:：])", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# số điện thoại Việt Nam: 0 hoặc +84, 9-10 chữ số sau đó, có thể ngăn bằng dấu cách / chấm / gạch; không dính vào số dài hơn
PHONE = re.compile(r"(?<![\w.])(?:\+84|0)(?:[\s.-]?\d){8,10}(?![\w])")
# có mã nước nhưng không có dấu + ("84-983226880", "84 983 226 880"): đầu số di động 3/5/7/8/9, tách bằng gạch / khoảng trắng (không dùng dấu chấm: "84.567.890.123" là số liệu bảng)
PHONE_INTL = re.compile(r"(?<![\w.,/+-])84[\s-]?[35789](?:[\s-]?\d){8}(?![\w.,])")
_THOUSANDS = re.compile(r"^\d{1,3}(?:\s\d{3}){2,}$")  # "84 500 000 000": số có nhóm nghìn, không phải số điện thoại


def heading_key(line: str) -> str:
    """Khoá so sánh tiêu đề: NFC, chữ thường, bỏ số thứ tự đầu dòng ("5.", "IV)", "2.1") và dấu :. cuối dòng, gộp khoảng trắng."""
    text = _NUMBERING.sub("", unicodedata.normalize("NFC", line).strip()).casefold()
    return re.sub(r"\s+", " ", text).strip(" :.\t")


def is_ref_heading(key: str) -> bool:
    """
    True nếu khoá tiêu đề (heading_key) là tiêu đề phần tài liệu tham khảo: khớp cả dòng, có hậu tố (_REF_SUFFIX), dính liền mục 1, hoặc là dạng song ngữ
    mà mọi phần cách nhau bởi "/" hay "|" đều là tiêu đề tham khảo ("tài liệu tham khảo/ references", J-25 E).
    """
    if key in REF_HEADINGS or _REF_STUCK.match(key):
        return True
    parts = [p.strip(" :.\t") for p in re.split(r"[/|]", key)]
    return len(parts) > 1 and all(p in REF_HEADINGS or bool(_REF_STUCK.match(p)) for p in parts)


def is_toc_page(text: str) -> bool:
    """Trang mục lục: có tiêu đề "Mục lục" / "Contents" trong 3 dòng đầu, hoặc phần lớn dòng kết thúc bằng dấu chấm dẫn + số trang."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if any(heading_key(ln) in TOC_HEADINGS for ln in lines[:3]):
        return True
    return len(lines) >= TOC_MIN_LINES and sum(bool(_LEADER.search(ln)) for ln in lines) / len(lines) >= TOC_LEADER_FRAC


def is_editorial_page(text: str) -> bool:
    """Trang ban biên tập / bìa mục lục (J-33 F): ngắn (<= EDITORIAL_MAX_WORDS từ) và có >= EDITORIAL_MIN_MARKERS loại chức danh / mục khác nhau (EDITORIAL_MARKERS)."""
    if len(text.split()) > EDITORIAL_MAX_WORDS:
        return False
    flat = unicodedata.normalize("NFC", text)
    return sum(bool(rx.search(flat)) for rx in _EDITORIAL) >= EDITORIAL_MIN_MARKERS


def drop_front_matter(pages: list[str]) -> tuple[list[str], dict]:
    """
    Bỏ trang bìa (trang đầu rất ngắn, bài từ COVER_MIN_PAGES trang), trang mục lục và trang ban biên tập (TOC_SCAN_PAGES trang đầu; trang ban biên tập còn xét cả
    EDITORIAL_TAIL_PAGES trang cuối, J-33 F).

    Trang bị bỏ thành chuỗi rỗng (giữ nguyên số trang). Trang rỗng sẵn không tính là bị bỏ.

    Returns:
        (các trang sau khi bỏ, {"cover_pages": n, "toc_pages": n, "editorial_pages": n}).
    """
    out = list(pages)
    stats = {"cover_pages": 0, "toc_pages": 0, "editorial_pages": 0}
    for i, text in enumerate(pages[:TOC_SCAN_PAGES]):
        if not text.strip():
            continue
        if i == 0 and len(pages) >= COVER_MIN_PAGES and len(text.split()) < COVER_MAX_WORDS:
            out[i], stats["cover_pages"] = "", stats["cover_pages"] + 1
        elif is_toc_page(text):
            out[i], stats["toc_pages"] = "", stats["toc_pages"] + 1
        elif is_editorial_page(text):
            out[i], stats["editorial_pages"] = "", stats["editorial_pages"] + 1
    if len(pages) >= COVER_MIN_PAGES:  # trang cuối: chỉ nhận ban biên tập, không nhận bìa / mục lục (bài ngắn có thể kết thúc bằng trang rất ngắn)
        for i in range(max(TOC_SCAN_PAGES, len(pages) - EDITORIAL_TAIL_PAGES), len(pages)):
            if out[i].strip() and is_editorial_page(out[i]):
                out[i], stats["editorial_pages"] = "", stats["editorial_pages"] + 1
    return out, stats


def _is_intl_phone(match: re.Match) -> bool:
    """PHONE_INTL khớp mà không phải số có nhóm nghìn kiểu bảng số liệu ("84 500 000 000")."""
    return not _THOUSANDS.match(match.group(0).strip())


def _is_phone(match: re.Match) -> bool:
    """
    PHONE khớp mà đúng là số điện thoại, không phải dãy số liệu bảng (J-28).

    Số điện thoại hoặc liền một khối 10-11 chữ số ("0912345678"), hoặc là các khối cách nhau bằng dấu cách / chấm / gạch mà khối đầu là mã vùng / đầu số
    ("0903", "024", "+84") và các khối sau mỗi khối >= 3 chữ số ("0903 123 456", "024.3825.1234"). Dãy "0.0 0.2 0.4 0.6 0.8" hay "0.15 0.23 0.42 5.89" của bảng và
    trục biểu đồ có khối 1-2 chữ số nên bị loại; trên vjol_10k_v4a hơn một nửa trong 425 chỗ bị che là số liệu như vậy.
    """
    toks = [t for t in re.split(r"[\s.-]+", match.group(0).strip()) if t]
    if len(toks) == 1:
        return len(re.sub(r"\D", "", toks[0])) in (10, 11)
    head = toks[0].lstrip("+")
    return 2 <= len(head) <= 4 and all(len(t) >= 3 for t in toks[1:])


def find_phones(text: str) -> list[str]:
    """Các số điện thoại thật trong văn bản (PHONE hợp lệ theo _is_phone, và PHONE_INTL không phải số có nhóm nghìn)."""
    return [m.group(0) for m in PHONE.finditer(text) if _is_phone(m)] + [m.group(0) for m in PHONE_INTL.finditer(text) if _is_intl_phone(m)]


def mask_phones(line: str) -> tuple[str, int]:
    """Che số điện thoại (PHONE hợp lệ theo _is_phone, và PHONE_INTL) trong một dòng bằng <PHONE>; trả về (dòng mới, số số đã che)."""
    n = 0

    def repl(m: re.Match) -> str:
        """Thay số điện thoại hợp lệ bằng <PHONE>, giữ nguyên dãy số liệu."""
        nonlocal n
        if _is_phone(m):
            n += 1
            return "<PHONE>"
        return m.group(0)

    line = PHONE.sub(repl, line)
    out, last, extra = [], 0, 0
    for m in PHONE_INTL.finditer(line):
        if _is_intl_phone(m):
            out.append(line[last:m.start()] + "<PHONE>")
            last, extra = m.end(), extra + 1
    out.append(line[last:])
    return "".join(out), n + extra


def _token_keys(line: str) -> list[tuple[str, int, int]]:
    """Các từ của dòng: (khoá so sánh = chữ thường, mọi chuỗi chữ số thành "#"; vị trí đầu; vị trí cuối)."""
    return [(re.sub(r"\d+", "#", m.group(0).casefold()), m.start(), m.end()) for m in re.finditer(r"\S+", unicodedata.normalize("NFC", line))]


def strip_running_headers(pages: list[str]) -> tuple[list[str], int]:
    """
    Bỏ tiêu đề / chân trang lặp giữa các trang của cùng một bài (J-25 C), kể cả khi dính đầu (hoặc cuối) dòng thân bài.

    Với mỗi trang, xét HEADER_SCAN_LINES dòng không rỗng đầu (so tiền tố từ) và cuối (so hậu tố từ); khoá từ thay mọi chuỗi chữ số bằng "#", nên "số 3 - 2005" và
    "số 4 - 2005" trùng nhau. Tiền tố / hậu tố dài nhất (HEADER_MIN_TOKENS .. HEADER_MAX_TOKENS từ) lặp ở ít nhất max(HEADER_MIN_PAGES, HEADER_MIN_SHARE số trang) trang thì bị cắt khỏi
    mọi dòng mang nó; lặp tối đa HEADER_ROUNDS lượt cho trường hợp trang chẵn / lẻ khác tiêu đề. Bài dưới HEADER_MIN_PAGES trang giữ nguyên.

    J-33 E: sau các lượt trên, mỗi tiêu đề đã học còn bị cắt ở MỌI vị trí trong dòng của mọi trang (tiêu đề dính giữa đoạn do thứ tự khối chữ của PDF, hoặc ở trang đầu bài
    vốn không có tiêu đề lặp nên lượt đầu / cuối không thấy). Chuỗi từ khớp nguyên (>= HEADER_MIN_TOKENS từ, chữ số thành "#") nên câu trích dẫn tên tạp chí có dấu câu khác không bị cắt.

    Returns:
        (các trang sau khi bỏ, số dòng đã bị cắt).
    """
    out = [p for p in pages]
    n_pages = sum(bool(p.strip()) for p in out)
    if n_pages < HEADER_MIN_PAGES:
        return out, 0
    need = max(HEADER_MIN_PAGES, math.ceil(HEADER_MIN_SHARE * n_pages))
    cut = 0
    learned: list[tuple[str, ...]] = []
    for _ in range(HEADER_ROUNDS):
        counts: dict[tuple[str, tuple[str, ...]], set[int]] = {}
        for pi, text in enumerate(out):
            lines = text.splitlines()
            idx = [i for i, ln in enumerate(lines) if ln.strip()]
            for side, picks in (("head", idx[:HEADER_SCAN_LINES]), ("tail", idx[HEADER_SCAN_LINES:][-HEADER_SCAN_LINES:])):  # đầu và cuối không chung dòng
                for i in picks:
                    keys = [k for k, _, _ in _token_keys(lines[i])]
                    for n in range(HEADER_MIN_TOKENS, min(HEADER_MAX_TOKENS, len(keys)) + 1):
                        gram = tuple(keys[:n] if side == "head" else keys[-n:])
                        if any(k != "#" for k in gram):  # chỉ toàn số thì không phải tiêu đề
                            counts.setdefault((side, gram), set()).add(pi)
        best = max(((len(g), side, g) for (side, g), pgs in counts.items() if len(pgs) >= need), default=None)
        if best is None:
            break
        _, side, gram = best
        learned.append(gram)
        n = len(gram)
        for pi, text in enumerate(out):
            lines = text.splitlines()
            idx = [i for i, ln in enumerate(lines) if ln.strip()]
            for i in (idx[:HEADER_SCAN_LINES] if side == "head" else idx[HEADER_SCAN_LINES:][-HEADER_SCAN_LINES:]):
                toks = _token_keys(lines[i])
                keys = tuple(k for k, _, _ in toks)
                matches = keys[:n] == gram if side == "head" else keys[-n:] == gram  # len(keys) < n thì tuple ngắn hơn, không bao giờ bằng gram
                if matches:
                    line = unicodedata.normalize("NFC", lines[i])
                    lines[i] = line[toks[n - 1][2]:].strip() if side == "head" else line[:toks[-n][1]].strip()
                    cut += 1
            out[pi] = "\n".join(lines)
    for gram in learned:  # J-33 E: cắt tiêu đề đã học ở mọi vị trí còn sót
        for pi, text in enumerate(out):
            lines = text.splitlines()
            for i, line in enumerate(lines):
                lines[i], n_cut = _cut_gram_anywhere(line, gram)
                cut += n_cut
            out[pi] = "\n".join(lines)
    return out, cut


def _cut_gram_anywhere(line: str, gram: tuple[str, ...]) -> tuple[str, int]:
    """
    Cắt mọi chỗ chuỗi từ `gram` (khoá từ của _token_keys) xuất hiện trong `line`, kể cả giữa dòng; phần còn lại nối bằng một dấu cách.

    Returns:
        (dòng mới, số chỗ đã cắt); dòng không chứa gram trả về nguyên văn và 0.
    """
    toks = _token_keys(line)
    keys = [k for k, _, _ in toks]
    n = len(gram)
    spans, i = [], 0
    while i <= len(keys) - n:
        if tuple(keys[i:i + n]) == gram:
            spans.append((toks[i][1], toks[i + n - 1][2]))
            i += n
        else:
            i += 1
    if not spans:
        return line, 0
    text = unicodedata.normalize("NFC", line)
    out, last = [], 0
    for a, b in spans:
        out.append(text[last:a])
        last = b
    out.append(text[last:])
    return re.sub(r"[ \t]{2,}", " ", " ".join(x.strip() for x in out if x.strip())), len(spans)


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


def strip_trailing_references(pages: list[str]) -> tuple[list[str], int]:
    """
    Cắt khối mục tài liệu tham khảo Ở CUỐI BÀI khi không có tiêu đề (hoặc tiêu đề nằm trên trang bị bỏ / đọc hỏng) (J-25 E): đi ngược từ cuối bài, tìm khối cuối cùng bắt đầu bằng một mục
    tham khảo (vi_corpus.common.references) sau TAIL_REF_MIN_POS số dòng, có ít nhất TAIL_REF_MIN_ENTRIES mục và ít nhất TAIL_REF_MIN_SHARE dòng thuộc phần tham khảo.

    Returns:
        (các trang sau khi cắt, số dòng không rỗng đã bỏ). Không thấy khối nào thì giữ nguyên.
    """
    lines = [(p, i, ln) for p, text in enumerate(pages) for i, ln in enumerate(text.splitlines()) if ln.strip()]
    total = len(lines)
    if total < TAIL_REF_MIN_ENTRIES * 2:
        return pages, 0
    texts = [ln for _, _, ln in lines]
    flags = reference_flags(texts)
    entry = [is_reference_entry(t) for t in texts]
    start = None
    for s in range(int(TAIL_REF_MIN_POS * total), total):  # từ sớm đến muộn: lấy khối dài nhất thoả điều kiện
        if not entry[s]:
            continue
        n_entries = sum(entry[s:])
        if n_entries >= TAIL_REF_MIN_ENTRIES and sum(flags[s:]) / (total - s) >= TAIL_REF_MIN_SHARE:
            start = s
            break
    if start is None:
        return pages, 0
    dropped = {(p, i) for p, i, _ in lines[start:]}
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
            n = 0
            for pattern in (header, GENERIC_HEADER, GENERIC_HEADER_UPPER):
                if pattern is not None:
                    line, k = pattern.subn(" ", line)
                    n += k
            if n:
                stats["journal_headers"] += n
                if not re.search(r"\w", line):
                    continue
                line = re.sub(r"[ \t]{2,}", " ", line).strip()
            line, n_mail = EMAIL.subn("<EMAIL>", line)
            line, n_tel = mask_phones(line)
            stats["emails_masked"] += n_mail
            stats["phones_masked"] += n_tel
            kept.append(line)
        out.append("\n".join(kept))
    return out, stats


def clean_paper_pages(pages: list[str], journal_title: str | None = None, vocab: frozenset[str] | None = None) -> tuple[list[str], dict]:
    """
    Bỏ trang bìa, mục lục, trang ban biên tập, tiêu đề / chân trang lặp, phần tài liệu tham khảo và khuôn mẫu tạp chí của một bài báo; khôi phục "ư" cho trang font gạch nối
    và bỏ câu rác của OCR VLM trước tiên (J-33 A, C).

    Args:
        pages:         Text từng trang.
        journal_title: Tên tạp chí trong metadata (để bỏ tiêu đề trang tạp chí dính giữa dòng); None thì không bỏ.
        vocab:         Tập âm tiết thật để kiểm khi khôi phục "ư" (vi_corpus.vjol.hyphen_fix); None thì bỏ bước khôi phục.

    Returns:
        (các trang đã làm sạch, {"hyphen_restored", "hyphen_ambiguous" (chỗ bỏ qua vì mơ hồ, J-42 P6), "mid_headers", "junk_lines" (J-42 (6)), "vlm_noise_removed", "cover_pages", "toc_pages", "editorial_pages", "running_headers", "front_headers", "reference_lines", và các số đếm của strip_boilerplate}).
    """
    hyphen_stats: dict = {}
    pages, restored = restore_u_in_pages(pages, vocab, stats=hyphen_stats)
    noise = 0
    cleaned = []
    for text in pages:
        text, n = strip_vlm_noise(text)
        cleaned.append(text)
        noise += n
    pages, stats = drop_front_matter(cleaned)
    stats = {"hyphen_restored": restored, "hyphen_ambiguous": hyphen_stats.get("ambiguous", 0), "vlm_noise_removed": noise, **stats}
    pages, stats["running_headers"] = strip_running_headers(pages)
    pages, stats["front_headers"] = strip_front_headers(pages)
    pages, stats["mid_headers"] = strip_mid_headers(pages)
    pages, stats["junk_lines"] = strip_junk_lines(pages)
    pages, stats["reference_lines"] = strip_references(pages)
    pages, tail = strip_trailing_references(pages)
    stats["reference_lines"] += tail
    pages, boiler = strip_boilerplate(pages, journal_title)
    return pages, {**stats, **boiler}
