"""
Quyết định OCR lại CẢ BÀI cho VJOL khi lớp chữ có lỗi mà quy tắc vá không an toàn (J-42 mục (2), (3), (4); P1, P2).

Thay vì vá chữ hỏng bằng regex ("sửa im lặng" có thể ra chữ sai nghĩa, J-41 (4)), bài có lớp chữ hỏng ở đủ nhiều trang được đưa qua chuỗi OCR VLM như bài nhiễu phông (J-10):
- page_defects / page_flags: số đo lỗi của MỘT trang (lệch dấu, chữ rơi mất, từ dính liền, rác, gạch nối thay ư) và các tín hiệu trang vượt ngưỡng.
- reocr_reason: từ lớp chữ các trang của một bài, trả về REASON_DEFECT nếu số trang trúng tín hiệu (REOCR_SIGNALS) >= max(REOCR_MIN_PAGES, REOCR_SHARE x số trang có chữ). Là hàm `ocr_all_if` của
  vi_corpus.common.pdf_text.extract_pdf; ngưỡng là điểm khởi đầu, scripts/vjol/plan_reocr.py quét ngưỡng trên dữ liệu thật để chốt (J-42 (2)).
- Dấu gạch nối thay ư ("hyphen") KHÔNG nằm trong REOCR_SIGNALS mặc định: ảnh trang cũng in dấu "-" nên OCR không đọc ra "ư" (J-32, J-37); gửi những bài đó đi OCR chỉ tốn GPU. plan_reocr vẫn đếm riêng.
  Dấu hiệu "gibberish" cũng bỏ ngoài mặc định vì công thức dàn phẳng là nguồn báo nhầm chính; đưa vào khi số đo plan_reocr cho thấy nên.
- Nhóm font xấu (J-42 (4)): font_canary.py ghi `font_groups_bad.json` (danh sách cặp producer / creator), load_font_groups đọc, font_groups_tag cho dấu nhận dạng ghi vào extract_version.
- pdf_producer_creator: đọc (producer, creator) của PDF; checkpoint không lưu hai trường này.
"""

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

from vi_corpus.common import defects as Df
from vi_corpus.common.garbled import REASON_DEFECT, hyphen_lost

REOCR_SIGNALS = ("tone_shift", "dropped_letters", "joined_words")  # tín hiệu trang quyết định OCR lại cả bài (xem docstring module)
ALL_SIGNALS = ("tone_shift", "dropped_letters", "joined_words", "gibberish", "hyphen")  # mọi tín hiệu plan_reocr đo
REOCR_MIN_PAGES = 2  # bài cần ít nhất ngần này trang trúng tín hiệu ...
REOCR_SHARE = 0.15  # ... và ít nhất tỷ lệ này số trang có chữ (điểm khởi đầu chưa đo; plan_reocr quét)
HYPHEN_PAGE_MIN = 3  # trang có ít nhất ngần này chỗ "-" thay ư thì tính là trang "hyphen" (chỉ để đo)
FONT_GROUPS_REL = "processed/vjol_font_canary/font_groups_bad.json"  # tương đối so với data-root (scripts/vjol/font_canary.py ghi)


def page_defects(text: str, vocab: frozenset[str] | None) -> dict[str, float]:
    """
    Số đo lỗi của một trang: {"tone_shift", "dropped_letters", "joined_words" (số chỗ; 0 nếu thiếu `vocab`), "gibberish" (tỷ lệ từ rác, 0 nếu dưới GIBBERISH_MIN_TOKENS từ), "hyphen" (số chỗ "-" thay ư)}.
    """
    ratio, n = Df.gibberish_ratio(text)
    return {"tone_shift": Df.tone_shift_hits(text, vocab) if vocab is not None else 0, "dropped_letters": Df.dropped_letters_runs(text),
            "joined_words": Df.joined_words_hits(text, vocab) if vocab is not None else 0,
            "gibberish": ratio if n >= Df.GIBBERISH_MIN_TOKENS else 0.0, "hyphen": hyphen_lost(text)[2]}


def page_flags(measures: dict[str, float]) -> list[str]:
    """Các tín hiệu (tên trong ALL_SIGNALS) mà số đo của một trang (page_defects) vượt ngưỡng của vi_corpus.common.defects."""
    limits = {"tone_shift": Df.SHIFT_MIN_HITS, "dropped_letters": Df.DROPPED_MIN_RUNS, "joined_words": Df.JOINED_MIN_HITS, "gibberish": Df.GIBBERISH_MIN_RATIO, "hyphen": HYPHEN_PAGE_MIN}
    return [name for name in ALL_SIGNALS if measures[name] >= limits[name] and measures[name] > 0]


def pages_needed(n_text_pages: int, min_pages: int = REOCR_MIN_PAGES, share: float = REOCR_SHARE) -> int:
    """Số trang trúng tín hiệu tối thiểu để cả bài đi OCR lại: max(min_pages, ceil(share x số trang có chữ))."""
    return max(min_pages, math.ceil(share * n_text_pages))


def reocr_reason(layer: Sequence[str | None], vocab: frozenset[str] | None, signals: Sequence[str] = REOCR_SIGNALS,
                 min_pages: int = REOCR_MIN_PAGES, share: float = REOCR_SHARE) -> str | None:
    """
    REASON_DEFECT nếu cả bài phải OCR lại, None nếu không (xem docstring module). Là hàm `ocr_all_if` của extract_pdf (qua functools.partial).

    Args:
        layer:     Lớp chữ từng trang (None cho trang không có lớp chữ; trang rỗng sau strip không được tính).
        vocab:     Tập âm tiết; None thì tone_shift và joined_words không đo được (chỉ còn dropped_letters).
        signals:   Các tín hiệu tính (mặc định REOCR_SIGNALS).
        min_pages: Số trang trúng tối thiểu tuyệt đối.
        share:     Tỷ lệ trang trúng tối thiểu trên số trang có chữ.
    """
    texts = [t for t in layer if t is not None and t.strip()]
    if not texts:
        return None
    hit = sum(any(sig in signals for sig in page_flags(page_defects(t, vocab))) for t in texts)
    return REASON_DEFECT if hit >= pages_needed(len(texts), min_pages, share) else None


def pdf_producer_creator(path: Path) -> tuple[str, str]:
    """(producer, creator) trong metadata của PDF ("" nếu thiếu hoặc PDF không mở được); checkpoint không lưu hai trường này nên phải đọc lại file gốc."""
    import pymupdf

    try:
        with pymupdf.open(path) as doc:
            meta = doc.metadata or {}
    except Exception:  # noqa: BLE001 — PDF hỏng chỉ thiếu thông tin nhóm, không dừng phép đo
        return "", ""
    return (meta.get("producer") or "").strip(), (meta.get("creator") or "").strip()


def load_font_groups(path: Path) -> list[tuple[str, str]]:
    """Đọc font_groups_bad.json ({"groups": [{"producer", "creator", ...}]}) thành danh sách (producer, creator); file không tồn tại thì []."""
    path = Path(path)
    if not path.is_file():
        return []
    pairs = [(g["producer"], g["creator"]) for g in json.loads(path.read_text(encoding="utf-8")).get("groups", [])]
    return [pair for pair in pairs if pair != ("", "")]  # nhóm không có metadata không phải một nhóm font


def font_groups_tag(groups: Sequence[tuple[str, str]]) -> str:
    """Dấu nhận dạng ngắn (băm) của danh sách nhóm font xấu, ghi vào extract_version để đổi danh sách thì checkpoint cũ bị làm lại; "" nếu không có nhóm."""
    return hashlib.sha1("\n".join(sorted(f"{p}\t{c}" for p, c in groups)).encode("utf-8")).hexdigest()[:6] if groups else ""
