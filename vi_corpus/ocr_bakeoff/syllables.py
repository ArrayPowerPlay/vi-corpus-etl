"""
Tập âm tiết tham chiếu cho bộ B (ảnh quét thật, không có đáp án) của đợt so sánh OCR (F-11).

Đếm các "từ chỉ gồm chữ cái" (chữ thường, NFC) trên văn bản SEA-PILE v2, giữ những từ xuất hiện >= min_freq lần. Âm tiết
OCR đọc sai dấu ("ngot", "bưc") hay chữ bịa thường không có trong tập này, nên tỷ lệ âm tiết lạ là số đo thay thế cho CER
khi không có đáp án. Từ tiếng Anh phổ biến cũng có trong SEA-PILE nên trang song ngữ không bị phạt nhiều.
Đọc parquet theo lô (pyarrow iter_batches), chỉ cột text, dừng sau max_docs văn bản: bộ nhớ không phụ thuộc cỡ file.
"""

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from vi_corpus.ocr_bakeoff.metrics import letter_words


def build_syllables(files: list[Path], text_col: str, max_docs: int = 300_000, min_freq: int = 20) -> dict:
    """
    Đếm từ chữ cái trên tối đa max_docs văn bản của các file parquet (theo thứ tự cho trước).

    Returns:
        {"docs", "files", "min_freq", "tokens", "syllables": [...]}.
    """
    counts: Counter = Counter()
    docs = used = 0
    for path in files:
        if docs >= max_docs:
            break
        used += 1
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=[text_col]):
            for text in batch.column(0).to_pylist():
                counts.update(letter_words(text or ""))
                docs += 1
                if docs >= max_docs:
                    break
            if docs >= max_docs:
                break
    keep = sorted(w for w, n in counts.items() if n >= min_freq)
    return {"docs": docs, "files": used, "min_freq": min_freq, "tokens": sum(counts.values()), "syllables": keep}


def load_syllables(path: Path) -> frozenset[str]:
    """Đọc tập âm tiết đã lưu bằng build_syllables."""
    return frozenset(json.loads(Path(path).read_text(encoding="utf-8"))["syllables"])


def unknown_rate(text: str, vocab: frozenset[str]) -> tuple[int, int]:
    """(số từ chữ cái không có trong vocab, tổng số từ chữ cái) của một văn bản."""
    ws = letter_words(text)
    return sum(w not in vocab for w in ws), len(ws)
