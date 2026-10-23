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
- F: dòng khuôn mẫu "ngày + động từ + ngày tháng" nói chung ("Ngày phản biện: 12/05/2015"), số điện thoại có mã nước không có dấu + ("84-983226880") có kiểm đầu số
  và loại số liệu bảng.
"""

import math
import re
import unicodedata

from vi_corpus.common.references import is_reference_entry, reference_flags

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


def _is_intl_phone(match: re.Match) -> bool:
    """PHONE_INTL khớp mà không phải số có nhóm nghìn kiểu bảng số liệu ("84 500 000 000")."""
    return not _THOUSANDS.match(match.group(0).strip())


def mask_phones(line: str) -> tuple[str, int]:
    """Che số điện thoại (PHONE và PHONE_INTL) trong một dòng bằng <PHONE>; trả về (dòng mới, số số đã che)."""
    line, n = PHONE.subn("<PHONE>", line)
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

    Returns:
        (các trang sau khi bỏ, số dòng đã bị cắt).
    """
    out = [p for p in pages]
    n_pages = sum(bool(p.strip()) for p in out)
    if n_pages < HEADER_MIN_PAGES:
        return out, 0
    need = max(HEADER_MIN_PAGES, math.ceil(HEADER_MIN_SHARE * n_pages))
    cut = 0
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
    return out, cut


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
            if header is not None:
                line, n = header.subn(" ", line)
                stats["journal_headers"] += n
                if n and not re.search(r"\w", line):
                    continue
                line = re.sub(r"[ \t]{2,}", " ", line).strip()
            line, n_mail = EMAIL.subn("<EMAIL>", line)
            line, n_tel = mask_phones(line)
            stats["emails_masked"] += n_mail
            stats["phones_masked"] += n_tel
            kept.append(line)
        out.append("\n".join(kept))
    return out, stats


def clean_paper_pages(pages: list[str], journal_title: str | None = None) -> tuple[list[str], dict]:
    """
    Bỏ trang bìa, mục lục, tiêu đề / chân trang lặp, phần tài liệu tham khảo và khuôn mẫu tạp chí của một bài báo.

    Args:
        pages:         Text từng trang.
        journal_title: Tên tạp chí trong metadata (để bỏ tiêu đề trang tạp chí dính giữa dòng); None thì không bỏ.

    Returns:
        (các trang đã làm sạch, {"cover_pages", "toc_pages", "running_headers", "reference_lines", và các số đếm của strip_boilerplate}).
    """
    pages, stats = drop_front_matter(pages)
    pages, stats["running_headers"] = strip_running_headers(pages)
    pages, stats["reference_lines"] = strip_references(pages)
    pages, tail = strip_trailing_references(pages)
    stats["reference_lines"] += tail
    pages, boiler = strip_boilerplate(pages, journal_title)
    return pages, {**stats, **boiler}
