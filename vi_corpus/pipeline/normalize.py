"""
Stage prepare, bước 1 (normalize) và đếm từ / token: chỉ chuẩn hóa Unicode và khoảng trắng, không sửa nội dung.

Theo nguyên tắc "không sửa nội dung" của docs/PIPELINE.md: không paraphrase, không sửa số liệu / tên / chính tả.
Việc làm: NFC, bỏ ký tự điều khiển và ký tự vô hình (zero-width, BOM, soft hyphen), đổi mọi loại khoảng trắng Unicode
về dấu cách, gộp dấu cách thừa, gộp dòng trống thừa. Vị trí dấu thanh (hòa / hoà) giữ nguyên.
Văn bản có code (vi_corpus.pipeline.lines.has_code, chạy trên text CÒN thụt lề) giữ khoảng trắng đầu dòng (C-09): thụt lề
là cú pháp của Python; chỉ gộp khoảng trắng giữa dòng và bỏ khoảng trắng cuối dòng. Văn bản khác bỏ cả khoảng trắng đầu dòng.
Việc bỏ tiêu đề / số trang của sách đã làm ở ingest (clean_pages); xoá dòng lặp ở vi_corpus.pipeline.lines.
Đếm token bằng vi_corpus.pipeline.tokens (tokenizer Qwen3 mặc định, D-01); word_count là số từ tách bằng khoảng trắng.
"""

import re
import unicodedata

from vi_corpus.pipeline.lines import has_code
from vi_corpus.pipeline.tokens import TokenCounter

NORMALIZER_VERSION = "2"  # 2 = giữ thụt lề cho văn bản có code (C-09)
_SPACES = re.compile(r"[ \t]+")
_INDENT = re.compile(r"^[ \t]*")
_BLANK_LINES = re.compile(r"\n{3,}")


def _normalize(text: str) -> tuple[str, bool]:
    """Chuẩn hóa một văn bản (xem docstring module). Returns (text mới, có giữ thụt lề vì là code không)."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "\n\t":
            out.append(ch)
        elif cat in ("Cc", "Cf"):  # điều khiển, zero-width, BOM, soft hyphen
            continue
        elif cat in ("Zs", "Zl", "Zp"):
            out.append(" ")
        else:
            out.append(ch)
    text = "".join(out)
    keep_indent = has_code(text)  # C-09: nhận diện trên text còn thụt lề, trước khi gộp khoảng trắng
    lines = []
    for line in text.split("\n"):
        indent = _INDENT.match(line).group() if keep_indent else ""
        body = _SPACES.sub(" ", line).strip()
        lines.append(indent + body if body else "")
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip("\n"), keep_indent


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa một văn bản (xem docstring module). Giữ ngắt dòng; kết quả không có khoảng trắng cuối dòng, không có dòng
    trống đầu / cuối, và không có khoảng trắng đầu dòng trừ khi văn bản có code (C-09).

    Ví dụ: "Việt" (chữ + dấu rời) -> "Việt" (NFC); "a​b" -> "ab"; "a  b" -> "a b".
    """
    return _normalize(text)[0]


def set_counts(rows: list[dict], counter: TokenCounter) -> None:
    """Điền char_count, word_count, token_count của các bản ghi từ text hiện tại (sửa tại chỗ, đếm token theo lô)."""
    tokens = counter.count_batch([row["text"] or "" for row in rows])
    for row, n in zip(rows, tokens):
        text = row["text"] or ""
        row["char_count"] = len(text)
        row["word_count"] = len(text.split())
        row["token_count"] = n


def normalize_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """
    Chuẩn hóa text của mọi bản ghi (sửa tại chỗ, source_sha256 vẫn là của text gốc để truy vết).

    Returns:
        (rows, thống kê: số ký tự trước / sau, số bản ghi bị đổi, số bản ghi giữ thụt lề vì có code).
    """
    before = after = changed = indented = 0
    for row in rows:
        old = row["text"] or ""
        row["text"], keep_indent = _normalize(old)
        before, after = before + len(old), after + len(row["text"])
        changed += old != row["text"]
        indented += keep_indent
    return rows, {"chars_before": before, "chars_after": after, "rows_changed": changed,
                  "indent_kept_docs": indented, "normalizer_version": NORMALIZER_VERSION}
