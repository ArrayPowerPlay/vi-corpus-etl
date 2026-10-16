"""
Bộ kiểm từng trang cho OCR bằng mô hình thị giác-ngôn ngữ (O-18): nhận ra trang hỏng hẳn mà không cần đáp án.

Lúc chạy thật không có đáp án để so, nên chỉ bắt được lỗi hỏng hẳn mà nhìn đầu ra là thấy (không bắt được đọc sai lẻ tẻ).
Mỗi dấu hiệu trả về một mã lý do; danh sách rỗng nghĩa là trang ổn:
    length        : finish_reason = "length" hoặc số token sinh ra chạm trần max_tokens (cắt cụt: lặp vô tận, base64).
    empty         : đầu ra rỗng sau khi làm sạch (trang chỉ có ảnh, hoặc chuỗi dừng cắt mất cả trang).
    embedded_image: đầu ra thô có ảnh nhúng ("data:image", "base64,"): mô hình cố vẽ lại logo / hình.
    english_meta  : đầu ra mở đầu bằng lời suy nghĩ tiếng Anh ("The user wants me to transcribe...").
    loop          : một đoạn ngắn lặp liên tiếp phủ phần lớn trang (dãy " . . . ." hoặc "......").
    too_long      : dài vô lý (> MAX_CHARS ký tự) cho một trang.
Chi phí báo nhầm thấp (trang bị báo chỉ phải chạy lại), chi phí bỏ sót cao, nên ngưỡng đặt hơi nhạy. Ngưỡng là khởi điểm,
được đo bằng scripts/ocr_bakeoff/eval_guard.py trên đầu ra thật của đợt so sánh (số báo nhầm / bỏ sót nằm ở O-18).
"""

import re

MAX_CHARS = 12000  # trang sách dày đặc ~3.000-5.000 ký tự; trên mức này gần như chắc chắn là lặp
TOKEN_CAP_RATIO = 0.98  # completion_tokens >= tỷ lệ này x max_tokens coi như chạm trần
LOOP_UNIT_MAX = 8  # đơn vị lặp dài tối đa (ký tự), vd ". " hoặc "..."
LOOP_MIN_REPEATS = 25  # số lần lặp liên tiếp tối thiểu để tính là một dãy lặp
LOOP_COVERAGE_MIN = 0.35  # dãy lặp phủ >= tỷ lệ này số ký tự của trang
LOOP_RUN_MIN_CHARS = 300  # hoặc có một dãy lặp liền dài >= số ký tự này
META_HEAD_CHARS = 300  # chỉ xét phần đầu đầu ra khi tìm lời suy nghĩ tiếng Anh

_LOOP = re.compile(r"(.{1,%d}?)\1{%d,}" % (LOOP_UNIT_MAX, LOOP_MIN_REPEATS - 1), re.DOTALL)
_META = re.compile(
    r"^\s*(?:the user (?:wants|is asking|asked|has provided)|i need to (?:read|transcribe|extract|carefully)|"
    r"let me (?:transcribe|read|look)|okay,? (?:let me|i need|so))", re.IGNORECASE)
_IMAGE_MARK = ("data:image", "base64,")


def repeat_stats(text: str) -> tuple[float, int]:
    """
    Mức lặp của văn bản: (tỷ lệ ký tự nằm trong dãy lặp, độ dài dãy lặp liền dài nhất).

    Dãy lặp = một đơn vị 1-LOOP_UNIT_MAX ký tự lặp liên tiếp từ LOOP_MIN_REPEATS lần trở lên (dãy ". . . ." hay "......").
    Dấu chấm dẫn mục lục ("Chương 1 ........ 5") cũng khớp nhưng chỉ phủ một phần nhỏ trang, nên không vượt ngưỡng phủ.
    """
    if not text:
        return 0.0, 0
    covered = longest = 0
    for m in _LOOP.finditer(text):
        n = m.end() - m.start()
        covered += n
        longest = max(longest, n)
    return covered / len(text), longest


def guard_page(rec: dict, max_tokens: int | None = None) -> list[str]:
    """
    Các lý do một trang OCR bị coi là hỏng (danh sách rỗng = ổn).

    Args:
        rec:        Bản ghi trang như trong outputs.jsonl: "raw" (đầu ra thô), "text" (văn bản thuần), "finish_reason",
                    "completion_tokens", "error".
        max_tokens: max_tokens đã dùng khi gọi engine (để phát hiện chạm trần khi finish_reason thiếu); None = bỏ qua.

    Returns:
        Mã lý do theo thứ tự cố định (length, empty, embedded_image, english_meta, loop, too_long). Bản ghi có lỗi
        (rec["error"]) trả ["error"].
    """
    if rec.get("error"):
        return ["error"]
    raw, text = rec.get("raw") or "", rec.get("text") or ""
    reasons = []
    capped = (max_tokens and rec.get("completion_tokens") and rec["completion_tokens"] >= TOKEN_CAP_RATIO * max_tokens)
    if rec.get("finish_reason") == "length" or capped:
        reasons.append("length")
    if not text.strip():
        reasons.append("empty")
    if any(mark in raw for mark in _IMAGE_MARK):
        reasons.append("embedded_image")
    if _META.match(text[:META_HEAD_CHARS]):
        reasons.append("english_meta")
    coverage, longest = repeat_stats(text)
    if coverage >= LOOP_COVERAGE_MIN or longest >= LOOP_RUN_MIN_CHARS:
        reasons.append("loop")
    if len(text) > MAX_CHARS:
        reasons.append("too_long")
    return reasons
