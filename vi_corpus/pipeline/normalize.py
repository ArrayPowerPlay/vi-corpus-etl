"""
Stage 2 (normalize) và đếm từ / token: chỉ chuẩn hóa Unicode và khoảng trắng, không sửa nội dung.

Theo nguyên tắc "không sửa nội dung" của docs/PIPELINE.md: không paraphrase, không sửa số liệu / tên / chính tả.
Việc làm: NFC, bỏ ký tự điều khiển và ký tự vô hình (zero-width, BOM, soft hyphen), đổi mọi loại khoảng trắng Unicode
về dấu cách, gộp dấu cách thừa, gộp dòng trống thừa. Vị trí dấu thanh (hòa / hoà) giữ nguyên.
Việc bỏ tiêu đề / số trang của sách đã làm ở ingest (clean_pages).
"""

import re
import unicodedata
from collections.abc import Callable

NORMALIZER_VERSION = "1"
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa một văn bản (xem docstring module). Giữ ngắt dòng; kết quả không có khoảng trắng đầu / cuối.

    Ví dụ: "Việt" (chữ + dấu rời) -> "Việt" (NFC); "a​b" -> "ab"; "a  b" -> "a b".
    """
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
    lines = (_SPACES.sub(" ", line).strip() for line in "".join(out).split("\n"))
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def make_token_counter(tokenizer: str | None) -> Callable[[str], int]:
    """
    Trả về hàm đếm token của một văn bản.

    tokenizer=None: đếm từ tách bằng khoảng trắng (với tiếng Việt là số âm tiết, ước lượng THẤP hơn số token BPE thật,
    chỉ để so sánh tương đối). Có tên / đường dẫn tokenizer HF thì dùng transformers.AutoTokenizer (cần cài sẵn).
    """
    if tokenizer is None:
        return lambda text: len(text.split())
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])


def set_counts(row: dict, counter: Callable[[str], int]) -> None:
    """Điền char_count, word_count, token_count của bản ghi từ text hiện tại (sửa tại chỗ)."""
    text = row["text"] or ""
    row["char_count"] = len(text)
    row["word_count"] = len(text.split())
    row["token_count"] = counter(text)


def normalize_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """
    Chuẩn hóa text của mọi bản ghi (sửa tại chỗ, source_sha256 vẫn là của text gốc để truy vết).

    Returns:
        (rows, thống kê: số ký tự trước / sau, số bản ghi bị đổi).
    """
    before = after = changed = 0
    for row in rows:
        old = row["text"] or ""
        row["text"] = normalize_text(old)
        before, after = before + len(old), after + len(row["text"])
        changed += old != row["text"]
    return rows, {"chars_before": before, "chars_after": after, "rows_changed": changed,
                  "normalizer_version": NORMALIZER_VERSION}
