"""
Adapter chung đưa đầu ra của các engine OCR về văn bản thuần trước khi chấm (F-11).

Định dạng đầu ra (khoá "output" của engine trong configs/ocr_bakeoff/engines.json):
    "plain"            : văn bản thuần (baseline Paddle + VietOCR), giữ nguyên.
    "markdown"         : markdown, có thể có bảng HTML hoặc bảng "|" (Qwen3-VL, PaddleOCR-VL).
    "dots_layout_json" : JSON bố cục của dots.ocr ([{"bbox", "category", "text"}], đã theo thứ tự đọc); bỏ phần tử
                         "Picture", bảng HTML, còn lại là markdown. JSON bị cắt cụt (hết max_tokens) thì vớt các trường
                         "text" đọc được.
Quy ước chung: bỏ cú pháp markdown (tiêu đề #, đậm / nghiêng, trích dẫn >, liên kết, ảnh, đường kẻ, hàng rào ```), bảng
(HTML hoặc "|") thành mỗi ô một dòng theo thứ tự hàng, giữ dấu gạch đầu dòng và mọi ký tự khác. Tiêu đề / chân trang vẫn
được xuất (clean_pages của pipeline lo bỏ), công thức LaTeX giữ nguyên chữ (bộ A ít công thức; là hạn chế đã biết).
"""

import json
import re
from html import unescape
from html.parser import HTMLParser

_TABLE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_FENCE_LINE = re.compile(r"^\s*```[\w-]*\s*$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_QUOTE = re.compile(r"^\s{0,3}>\s?")
_RULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")
_PIPE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>])")
_TEXT_FIELD = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')
_CATEGORY_FIELD = re.compile(r'"category"\s*:\s*"([^"]*)"')
# Mũ 2 / 3 viết bằng thẻ HTML hoặc LaTeX ("km<sup>2</sup>", "km$^{2}$") đổi về ký tự "²" / "³": văn bản thuần của corpus
# cũng sẽ đổi như vậy, nên engine biết đó là số mũ được tính đúng.
_SUPERSCRIPT = re.compile(r"<sup>\s*([23])\s*</sup>|\$\s*\^\s*\{?\s*([23])\s*\}?\s*\$")
_SUP_CHAR = {"2": "²", "3": "³"}


class _CellParser(HTMLParser):
    """Gom chữ từng ô <td> / <th> của bảng HTML theo thứ tự xuất hiện."""

    def __init__(self) -> None:
        """Khởi tạo danh sách ô rỗng."""
        super().__init__(convert_charrefs=True)
        self.cells: list[str] = []
        self._buf: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        """Mở ô mới."""
        if tag in ("td", "th"):
            self._buf = []
        elif tag == "br" and self._buf is not None:
            self._buf.append(" ")

    def handle_endtag(self, tag):
        """Đóng ô."""
        if tag in ("td", "th") and self._buf is not None:
            self.cells.append(" ".join("".join(self._buf).split()))
            self._buf = None

    def handle_data(self, data):
        """Chữ trong ô."""
        if self._buf is not None:
            self._buf.append(data)


def html_table_lines(table_html: str) -> list[str]:
    """Các ô (khác rỗng) của một bảng HTML, mỗi ô một phần tử, theo thứ tự hàng."""
    p = _CellParser()
    p.feed(table_html)
    p.close()
    return [c for c in p.cells if c]


def _pipe_row_cells(line: str) -> list[str]:
    """Các ô của một dòng bảng markdown "| a | b |"."""
    return [c.strip() for c in line.strip().strip("|").split("|") if c.strip()]


def markdown_to_plain(text: str) -> str:
    """Bỏ cú pháp markdown / HTML, bảng thành mỗi ô một dòng (xem docstring module)."""
    text = _SUPERSCRIPT.sub(lambda m: _SUP_CHAR[m.group(1) or m.group(2)], text or "")
    text = _TABLE.sub(lambda m: "\n" + "\n".join(html_table_lines(m.group(0))) + "\n", text)
    out: list[str] = []
    for line in text.split("\n"):
        if _FENCE_LINE.match(line) or _RULE.match(line) or _PIPE_SEP.match(line):
            continue
        if line.count("|") >= 2 and line.strip().startswith("|"):
            out.extend(_pipe_row_cells(line))
            continue
        line = _QUOTE.sub("", _HEADING.sub("", line))
        line = _LINK.sub(r"\1", _IMAGE.sub("", line))
        line = _ITALIC.sub(r"\1", _BOLD.sub(r"\2", line))
        line = unescape(_TAG.sub("", _ESCAPE.sub(r"\1", line)))
        out.append(line)
    return "\n".join(out).strip()


def _element_text(category: str, text: str) -> str:
    """Văn bản thuần của một phần tử bố cục dots.ocr."""
    if category == "Picture" or not text:
        return ""
    if category == "Table":
        return "\n".join(html_table_lines(text)) or markdown_to_plain(text)
    return markdown_to_plain(text)


def dots_layout_to_plain(raw: str) -> str:
    """
    Văn bản thuần từ JSON bố cục của dots.ocr (danh sách phần tử, hoặc {"layout": [...]}) theo thứ tự đã có.

    JSON hỏng / cắt cụt: lấy lần lượt các cặp "category" / "text" đọc được bằng biểu thức chính quy.
    """
    s = _strip_outer_fence(raw or "")
    try:
        data = json.loads(s)
        items = data.get("layout", data.get("elements", [])) if isinstance(data, dict) else data
        parts = [_element_text(str(it.get("category", "")), str(it.get("text") or ""))
                 for it in items if isinstance(it, dict)]
    except (json.JSONDecodeError, AttributeError, TypeError):
        parts = []
        for obj in re.split(r"(?=\{\s*\"bbox\")", s):
            t = _TEXT_FIELD.search(obj)
            if not t:
                continue
            cat = _CATEGORY_FIELD.search(obj)
            try:
                text = json.loads(f'"{t.group(1)}"')
            except json.JSONDecodeError:
                continue
            parts.append(_element_text(cat.group(1) if cat else "Text", text))
    return "\n".join(p for p in parts if p).strip()


def _strip_outer_fence(text: str) -> str:
    """Bỏ hàng rào ```markdown ... ``` bọc ngoài cả đầu ra (Qwen hay bọc)."""
    lines = text.strip().split("\n")
    if len(lines) >= 2 and _FENCE_LINE.match(lines[0]) and _FENCE_LINE.match(lines[-1]):
        return "\n".join(lines[1:-1])
    return text


def to_plain(raw: str, fmt: str) -> str:
    """
    Văn bản thuần từ đầu ra thô theo định dạng của engine.

    Raises:
        ValueError: định dạng lạ.
    """
    if fmt == "plain":
        return (raw or "").strip()
    if fmt == "markdown":
        return markdown_to_plain(_strip_outer_fence(raw or ""))
    if fmt == "dots_layout_json":
        return dots_layout_to_plain(raw)
    raise ValueError(f"Định dạng đầu ra lạ: {fmt!r} (plain | markdown | dots_layout_json)")
