"""
Stage 3 (chunk): cắt sách / tài liệu dài thành các đoạn ~600 từ ở ranh giới đoạn văn, rồi lấy mẫu xuống đúng số đoạn.

Chỉ áp dụng cho nguồn có SourceProfile.chunked (stbook, giao_trinh); nguồn khác đi qua nguyên vẹn. Mỗi đoạn là một mẫu
độc lập, giữ source_path / source_sha256 của sách gốc (để truy vết), thêm parent_doc_id và chunk_index.
Stage này cũng đặt số đếm (char / word / token) cho mọi bản ghi.
"""

import random
import re
from collections.abc import Callable

from vi_corpus.pipeline.config import RunConfig
from vi_corpus.pipeline.normalize import set_counts

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _split_long(paragraph: str, max_words: int) -> list[str]:
    """Cắt một đoạn văn dài hơn max_words thành các phần <= max_words, ưu tiên ranh giới câu, cuối cùng mới cắt cứng theo từ."""
    if len(paragraph.split()) <= max_words:
        return [paragraph]
    parts, cur = [], []
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        words = sentence.split()
        while len(words) > max_words:  # một câu dài quá mức: cắt cứng
            if cur:
                parts.append(" ".join(cur))
                cur = []
            parts.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if len(cur) + len(words) > max_words and cur:
            parts.append(" ".join(cur))
            cur = []
        cur.extend(words)
    if cur:
        parts.append(" ".join(cur))
    return parts


def chunk_text(text: str, target_words: int, max_words: int, min_words: int) -> list[str]:
    """
    Cắt text thành các đoạn, mỗi đoạn gồm nguyên các đoạn văn (cách nhau bằng dòng trống).

    Đoạn được chốt khi đạt target_words, hoặc khi thêm đoạn văn kế tiếp sẽ vượt max_words. Đoạn cuối ngắn hơn
    min_words thì gộp vào đoạn trước. Text rỗng trả về [].
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    paragraphs = [p for para in text.split("\n\n") if para.strip() for p in _split_long(para.strip(), max_words)]
    for para in paragraphs:
        n = len(para.split())
        if cur and cur_words + n > max_words:
            chunks.append("\n\n".join(cur))
            cur, cur_words = [], 0
        cur.append(para)
        cur_words += n
        if cur_words >= target_words:
            chunks.append("\n\n".join(cur))
            cur, cur_words = [], 0
    if cur:
        if chunks and cur_words < min_words:
            chunks[-1] += "\n\n" + "\n\n".join(cur)
        else:
            chunks.append("\n\n".join(cur))
    return chunks


def chunk_rows(rows: list[dict], cfg: RunConfig, counter: Callable[[str], int]) -> tuple[list[dict], dict]:
    """
    Cắt đoạn các nguồn chunked, lấy mẫu xuống cfg.mix[nguồn] đoạn (ngẫu nhiên theo seed), rồi đếm từ / token mọi bản ghi.

    Returns:
        (rows, thống kê theo nguồn chunked: số sách, số đoạn trước khi lấy mẫu, số đoạn giữ lại).
    """
    out: list[dict] = []
    stats: dict[str, dict] = {}
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        if cfg.profile(row["source_key"]).chunked:
            by_source.setdefault(row["source_key"], []).append(row)
        else:
            out.append(row)
    for key, books in by_source.items():
        chunks = []
        for book in books:
            for i, text in enumerate(chunk_text(book["text"], cfg.chunk_target_words, cfg.chunk_max_words,
                                                cfg.chunk_min_words)):
                chunks.append({**book, "doc_id": f"{book['doc_id']}#c{i}", "text": text,
                               "parent_doc_id": book["doc_id"], "chunk_index": i})
        rng = random.Random(f"{cfg.seed}:{key}:chunk")
        n = cfg.mix.get(key)
        kept = sorted(rng.sample(chunks, n), key=lambda r: r["doc_id"]) if n is not None and n < len(chunks) else chunks
        stats[key] = {"books": len(books), "chunks_total": len(chunks), "chunks_kept": len(kept)}
        out.extend(kept)
    for row in out:
        set_counts(row, counter)
    return out, stats
