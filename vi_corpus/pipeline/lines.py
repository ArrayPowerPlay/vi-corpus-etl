"""
Stage prepare, bước 2 (xoá dòng lặp, D-04): chạy sau normalize, trước cắt đoạn. has_code cũng được normalize gọi
(trên text còn thụt lề) để quyết định giữ khoảng trắng đầu dòng cho văn bản có code (C-09).

A. Trong một văn bản (mọi nguồn): gộp các dòng giống hệt nằm liền nhau; dòng ngắn (< line_short_words từ) lặp
   >= line_repeat_min lần thì chỉ giữ lần đầu; bỏ dòng chỉ là số trang ("12", "- 12 -", "Trang 12", "12/300").
   Văn bản có code hoặc bảng (has_code_or_table) được MIỄN mục A (C-07): trong code dòng "}" / "end" lặp là cú pháp,
   trong bảng ô số đứng một dòng hay ô giống nhau liền nhau là dữ liệu, xoá thì hỏng nội dung. Văn bản được miễn chỉ
   còn bị bỏ dòng nhãn số trang chắc chắn ("Trang 12", "- 12 -"), không bỏ số trơn.
B. Tiêu đề / chân trang sách: đã làm ở vi_corpus.common.pdf_text.clean_pages lúc ingest (không lặp lại ở đây).
C. Liên văn bản (chỉ nguồn có SourceProfile.cross_line_clean, tức nguồn web): lượt 1 đếm mỗi dòng xuất hiện ở bao nhiêu
   văn bản của cùng nguồn, lượt 2 xoá dòng xuất hiện ở >= max(cross_line_min_docs, cross_line_min_frac x số văn bản)
   văn bản (menu, chân trang, câu mời đăng ký lặp trên nhiều trang web). Trong văn bản có code / bảng, mục C vẫn chạy
   nhưng không xoá dòng trông như code hay dòng không có chữ cái ("}", "0", "---"), vì các dòng đó lặp ở nhiều trang
   code một cách tự nhiên.

Không sửa chữ trong dòng, không bỏ thụt lề đầu dòng (văn bản có code giữ thụt lề từ normalize, C-09); dòng trống
(ranh giới đoạn văn) được giữ. Mỗi bản ghi được cộng dồn lines_removed / chars_removed; thống kê trả về có danh sách các
dòng bị xoá nhiều nhất để báo cáo đọc lại. source_sha256 vẫn là hash của text gốc (để truy vết).
"""

import hashlib
import math
import re
from collections import Counter

from vi_corpus.pipeline.config import RunConfig

LINES_VERSION = "4"  # 2 = miễn mục A cho văn bản có code / bảng (C-07); 3 = nhận diện code / bảng chặt hơn;
# 4 = mục C không bỏ thụt lề dòng đầu (C-09)
TOP_REMOVED = 30  # số dòng bị xoá nhiều nhất ghi vào thống kê
_PAGE_NUMBER = re.compile(r"^(?:[-–—]\s*)?(?:(?:trang|page|tr\.)\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?(?:\s*[-–—])?$",
                          re.IGNORECASE)
CODE_MIN_LINES = 3  # số dòng code LIỀN NHAU tối thiểu (dòng trống không ngắt; ít nhất một dòng "chắc") để coi là có code
TABLE_PIPE_RUN = 3  # số dòng liền nhau có cùng số dấu "|" (>= 2) tối thiểu để coi là bảng khi không có dòng kẻ |---|
TABLE_RUN_LINES = 6  # số dòng ô ngắn liền nhau tối thiểu (bảng trích từ PDF: mỗi ô một dòng)
TABLE_CELL_WORDS = 4  # dòng ít hơn số từ này mới coi là một ô bảng
# Dòng trông như code: hàng rào ```, dòng chỉ có ngoặc / chấm phẩy, dòng kết thúc bằng "{", câu lệnh import, dòng chỉ là
# một lời gọi hàm "f(x)" / "a.b(1, 2)", dòng bắt đầu bằng từ khoá chữ thường VÀ kết thúc bằng ":" / ";" / "{", dòng gán
# "x = ...;". Không dùng "kết thúc bằng ;" một mình vì văn bản pháp luật tiếng Việt kết thúc các điểm a), b) bằng dấu
# chấm phẩy; không dùng "kết thúc bằng )" vì văn xuôi trích PDF xuống dòng giữa câu ("for ... (see Section 3.2)"). Một
# dòng lẻ vẫn có thể khớp nhầm, nên has_code_or_table đòi một khối dòng liền nhau. Thêm: câu lệnh gọi hàm kết thúc ";"
# ("System.out.println(x);", cần chữ liền trước "(" nên điểm "a) ...;" của văn bản pháp luật không khớp) và dòng SQL.
_CODE_LINE = re.compile(
    r"^\s*(?:```.*|[{}\[\]();,]+|.*\{|#include\s*[<\"].*|import\s+[\w.]+(?:\s+as\s+\w+)?|"
    r"from\s+[\w.]+\s+import\s+[\w.*, ()]+|[A-Za-z_][\w.]*\([^()]*\)|.*[A-Za-z_]\(.*\)\s*;|"
    r"(?:SELECT|FROM|WHERE|ORDER BY|GROUP BY|INSERT INTO|UPDATE|DELETE FROM|CREATE TABLE|JOIN|VALUES)\s+\S.*|"
    r"(?:def|class|return|function|var|let|const|public|private|static|void|elif|else|except|try|catch|printf?|"
    r"while|for|if|end|begin|switch|case)\b.*[:;{]|"
    r"[A-Za-z_][\w.\[\]]*\s*[+\-*/:]?=\s*[^=].*;)\s*$")  # ":=" là phép gán Pascal
# Dòng code "yếu": hay gặp trong thân hàm nhưng một mình chưa chắc là code (công thức toán "y = ax + b", văn xuôi
# "return to ..."). Chỉ được tính khi nằm trong khối có ít nhất một dòng _CODE_LINE. Không dùng thụt lề làm dấu hiệu:
# code trích từ PDF / OCR thường đã mất thụt lề, còn văn xuôi hay thụt đầu đoạn.
_CODE_WEAK = re.compile(
    r"^\s*(?:[A-Za-z_][\w.\[\]'\"]*\s*(?:[+\-*/%]|//)?=\s*[^=\s].*|return(?:\s+[^\s.][^.]*)?|"
    r"pass|break|continue|else|try|finally|begin|end[;.]?)\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")  # dòng kẻ bảng markdown |---|---|
# Nhãn số trang chắc chắn ("Trang 12", "page 3", "- 12 -"): vẫn xoá trong văn bản được miễn (khác số trơn, có thể là ô bảng).
_PAGE_LABEL = re.compile(r"^(?:[-–—]\s*\d{1,4}\s*[-–—]|(?:trang|page|tr\.)\s*\d{1,4}(?:\s*/\s*\d{1,4})?)$", re.IGNORECASE)


def _mark_removed(row: dict, removed: list[str], counter: Counter) -> None:
    """Cộng số dòng / ký tự bị xoá vào bản ghi và vào bộ đếm dòng bị xoá."""
    row["lines_removed"] = (row.get("lines_removed") or 0) + len(removed)
    row["chars_removed"] = (row.get("chars_removed") or 0) + sum(len(ln) for ln in removed)
    counter.update(removed)


def _has_code_block(lines: list[str]) -> bool:
    """
    Có >= CODE_MIN_LINES dòng code liền nhau (_CODE_LINE hoặc _CODE_WEAK), trong đó ít nhất một dòng _CODE_LINE. Dòng
    trống không ngắt khối, dòng văn xuôi thì ngắt.
    """
    run, strong = 0, False
    for ln in lines:
        if not ln.strip():
            continue
        if _CODE_LINE.match(ln):
            run, strong = run + 1, True
        elif _CODE_WEAK.match(ln):
            run += 1
        else:
            run, strong = 0, False
        if run >= CODE_MIN_LINES and strong:
            return True
    return False


def _has_pipe_table(lines: list[str]) -> bool:
    """
    Có bảng dạng "|": một dòng kẻ |---|---|, hoặc >= TABLE_PIPE_RUN dòng liền nhau có cùng số dấu "|" (>= 2). Một dòng
    menu / breadcrumb "Trang chủ | Tin tức | Thể thao" lẻ không phải bảng.
    """
    run, prev = 0, -1
    for ln in lines:
        if _TABLE_SEPARATOR.match(ln):
            return True
        n = ln.count("|")
        run = run + 1 if n >= 2 and n == prev else (1 if n >= 2 else 0)
        prev = n
        if run >= TABLE_PIPE_RUN:
            return True
    return False


def _has_cell_table(lines: list[str]) -> bool:
    """Có dải >= TABLE_RUN_LINES dòng ô ngắn liền nhau mà ít nhất một nửa có chữ số (bảng trích từ PDF)."""
    run: list[str] = []
    for ln in [*lines, ""]:
        if ln.strip() and len(ln.split()) < TABLE_CELL_WORDS:
            run.append(ln)
            continue
        if len(run) >= TABLE_RUN_LINES and 2 * sum(any(c.isdigit() for c in x) for x in run) >= len(run):
            return True
        run = []
    return False


def has_code(text: str) -> bool:
    """Văn bản có khối code không (xem _has_code_block); normalize dùng để giữ thụt lề (C-09)."""
    return _has_code_block(text.split("\n"))


def has_code_or_table(text: str) -> bool:
    """Văn bản có code hoặc bảng không (để miễn mục A, C-07): khối code, bảng "|" hoặc bảng ô ngắn (xem các hàm _has_*)."""
    lines = text.split("\n")
    return _has_code_block(lines) or _has_pipe_table(lines) or _has_cell_table(lines)


def strip_page_labels(text: str) -> tuple[str, list[str]]:
    """Văn bản được miễn mục A: chỉ bỏ dòng nhãn số trang chắc chắn (_PAGE_LABEL). Returns (text mới, dòng bị xoá)."""
    lines = text.split("\n")
    removed = [ln for ln in lines if _PAGE_LABEL.match(ln.strip())]
    if not removed:
        return text, []
    return "\n".join(ln for ln in lines if not _PAGE_LABEL.match(ln.strip())), removed


def clean_lines_in_doc(text: str, short_words: int = 10, repeat_min: int = 3) -> tuple[str, list[str]]:
    """
    Xoá dòng lặp trong một văn bản (mục A, xem docstring module).

    Returns:
        (text mới, danh sách dòng bị xoá theo thứ tự). Dòng trống liên tiếp sinh ra do xoá được gộp lại.
    """
    lines = text.split("\n")
    counts = Counter(ln for ln in lines if ln and len(ln.split()) < short_words)
    out: list[str] = []
    removed: list[str] = []
    seen_short: set[str] = set()
    prev = None
    for ln in lines:
        if not ln:
            if out and out[-1] != "":
                out.append("")
            prev = None
            continue
        short = len(ln.split()) < short_words
        if ln == prev or _PAGE_NUMBER.match(ln) or (short and counts[ln] >= repeat_min and ln in seen_short):
            removed.append(ln)
            continue
        if short:
            seen_short.add(ln)
        out.append(ln)
        prev = ln
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out), removed


def _line_key(line: str) -> str:
    """Khoá đếm dòng liên văn bản: băm ngắn của dòng (đỡ tốn RAM khi đếm trên nhiều văn bản)."""
    return hashlib.blake2b(line.encode("utf-8"), digest_size=8).hexdigest()


def _structural(line: str) -> bool:
    """Dòng là cú pháp code hoặc không có chữ cái (ô số, ngoặc, đường kẻ): mục C không xoá trong văn bản có code / bảng."""
    return bool(_CODE_LINE.match(line)) or not any(c.isalpha() for c in line)


def clean_lines_cross_doc(rows: list[dict], min_docs: int, min_frac: float,
                          protected: set[int] | None = None) -> tuple[dict, Counter]:
    """
    Xoá dòng lặp ở nhiều văn bản của cùng một nguồn (mục C), sửa text tại chỗ.

    Args:
        rows:      Bản ghi của MỘT nguồn.
        min_docs:  Ngưỡng số văn bản tối thiểu.
        min_frac:  Ngưỡng theo tỷ lệ số văn bản; ngưỡng thật = max(min_docs, ceil(min_frac x số văn bản)).
        protected: id() của bản ghi có code / bảng: không xoá dòng _structural trong các bản ghi này.

    Returns:
        ({"threshold", "lines_flagged", "lines_removed"}, bộ đếm các dòng bị xoá).
    """
    threshold = max(min_docs, math.ceil(min_frac * len(rows)))
    doc_freq: Counter = Counter()
    for row in rows:
        doc_freq.update({_line_key(ln) for ln in (row["text"] or "").split("\n") if ln})
    flagged = {k for k, n in doc_freq.items() if n >= threshold}
    removed_counter: Counter = Counter()
    if flagged:
        for row in rows:
            keep, removed = [], []
            guard = protected is not None and id(row) in protected
            for ln in (row["text"] or "").split("\n"):
                hit = ln and _line_key(ln) in flagged and not (guard and _structural(ln))
                (removed if hit else keep).append(ln)
            if removed:
                row["text"] = re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip("\n")  # giữ thụt lề dòng đầu
                _mark_removed(row, removed, removed_counter)
    return {"threshold": threshold, "lines_flagged": len(flagged),
            "lines_removed": sum(removed_counter.values())}, removed_counter


def clean_rows(rows: list[dict], cfg: RunConfig) -> tuple[list[dict], dict]:
    """
    Chạy mục A cho mọi bản ghi (trừ văn bản có code / bảng, C-07) rồi mục C cho từng nguồn có cross_line_clean (sửa tại
    chỗ).

    Returns:
        (rows, thống kê: số dòng / ký tự bị xoá theo nguồn và theo mục, ngưỡng mục C, các dòng bị xoá nhiều nhất).
    """
    per_source: dict[str, dict] = {}
    top: Counter = Counter()
    by_source: dict[str, list[dict]] = {}
    protected: set[int] = set()
    for row in rows:
        row["lines_removed"] = row["chars_removed"] = 0
        st = per_source.setdefault(row["source_key"], {"in_doc_lines": 0, "in_doc_exempt_docs": 0})
        if has_code_or_table(row["text"] or ""):  # C-07: miễn mục A, chỉ bỏ nhãn số trang chắc chắn
            st["in_doc_exempt_docs"] += 1
            protected.add(id(row))
            new, removed = strip_page_labels(row["text"] or "")
        else:
            new, removed = clean_lines_in_doc(row["text"] or "", cfg.line_short_words, cfg.line_repeat_min)
        row["text"] = new
        _mark_removed(row, removed, top)
        st["in_doc_lines"] += len(removed)
        by_source.setdefault(row["source_key"], []).append(row)
    for key, group in by_source.items():
        if cfg.profile(key).cross_line_clean:
            cstats, removed = clean_lines_cross_doc(group, cfg.cross_line_min_docs, cfg.cross_line_min_frac, protected)
            per_source[key]["cross_doc"] = cstats
            top.update(removed)
    for key, group in by_source.items():
        per_source[key]["chars_removed"] = sum(r["chars_removed"] for r in group)
    return rows, {"version": LINES_VERSION, "per_source": per_source,
                  "top_removed": [[ln[:200], n] for ln, n in top.most_common(TOP_REMOVED)]}
