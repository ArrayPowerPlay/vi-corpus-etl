"""
Stage prepare, bước 3 (chunk): cắt văn bản thành các đoạn (mẫu) theo token Qwen3 (D-10), rồi lấy mẫu xuống đúng số đoạn.

Cỡ đoạn: mục tiêu chunk_target_tokens (~1.024), tối đa chunk_max_tokens (2.048); đoạn cuối < chunk_min_tokens (256)
gộp vào đoạn trước nếu tổng không vượt mức tối đa, nếu không thì nhận thêm các đoạn văn cuối của đoạn trước cho tới khi
đủ chunk_min_tokens. Thứ tự ưu tiên chỗ cắt:
1. Dòng tiêu đề ("Chương 3", "CHƯƠNG III", "Bài 5", "Mục 2.1", "Phần I"; chỉ khi dòng < 15 từ và không kết thúc bằng
   dấu chấm): đoạn đang gom đã đủ chunk_min_tokens thì cắt ngay trước tiêu đề, để đoạn không vắt qua hai chương.
2. Dòng trống (hết đoạn văn): cắt khi đoạn đang gom đạt mục tiêu, hoặc khi thêm đoạn văn kế tiếp sẽ vượt mức tối đa.
3. Hết câu: đoạn văn dài hơn mức tối đa được tách theo dòng, rồi theo câu; các mảnh liền nhau được gom tới chunk_target_tokens
   (không phải tới mức tối đa). Bài web thường ngăn đoạn văn bằng một dấu xuống dòng, nên cả bài là một "đoạn văn": gom tới
   mức tối đa thì bài 2.120 token thành [2.040, 80] và đoạn đuôi 80 token không gộp ngược được (vượt mức tối đa), bị loại
   too_short (report run 10.000 mẫu lần 2); gom tới mục tiêu thì đoạn đuôi luôn còn chỗ để gộp.
4. Cắt cứng theo token (một câu dài hơn mức tối đa), chia thành các phần đều nhau thay vì [max, max, ..., phần thừa].

Nguồn chunked (stbook, giao_trinh) luôn cắt và được lấy mẫu theo số đoạn. Nguồn khác (bài web) giữ nguyên, chỉ bài dài hơn
chunk_max_tokens mới cắt theo cùng quy tắc (SourceProfile.split_long; hội thoại SEA-Instruct không cắt). Mỗi đoạn là một
mẫu độc lập, giữ source_path / source_sha256 của văn bản gốc (truy vết), thêm parent_doc_id và chunk_index. Khi xuất CPT
các đoạn liền nhau còn giữ được nối lại (vi_corpus.pipeline.knowledge.pack_cpt_blocks).

Trước khi cắt, mỗi văn bản gốc được gắn doc_sha256 và doc_minhash (chữ ký MinHash của CẢ văn bản): dedup gần trùng chạy
theo văn bản, không theo đoạn (R-19). Stage này cũng đặt số đếm (char / word / token) cho mọi bản ghi.
"""

import hashlib
import re

from vi_corpus.common.schema import text_sha256
from vi_corpus.pipeline.config import RunConfig
from vi_corpus.pipeline.dedup import minhash_bytes
from vi_corpus.pipeline.normalize import set_counts
from vi_corpus.pipeline.tokens import TokenCounter

CHUNK_VERSION = "5"  # 1 = cắt theo số từ; 2 = cắt theo token (D-10); 3 = giữ thụt lề dòng đầu đoạn văn (C-09);
# 4 = đoạn văn quá dài gom mảnh tới mục tiêu, cắt cứng chia đều, đoạn đuôi ngắn mượn đoạn văn của đoạn trước;
# 5 = lấy mẫu đoạn theo băm (seed, nguồn, parent_doc_id, chunk_index) thay cho random.sample, để sửa chữ trong một bài không làm đổi đoạn được chọn của bài khác (J-40 a)
HEADING_MAX_WORDS = 15
_HEADING = re.compile(r"^(?:chương|bài|mục|phần|chapter|part|section)\s+(?:\d+(?:\.\d+)*|[ivxlcdm]+)\b",
                      re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def is_heading(paragraph: str) -> bool:
    """Đoạn văn bắt đầu bằng một dòng tiêu đề chương / bài / mục (xem docstring module, mức cắt 1)."""
    first = paragraph.split("\n", 1)[0].strip()
    return (bool(_HEADING.match(first)) and len(first.split()) < HEADING_MAX_WORDS
            and not first.endswith("."))


def _split_long(paragraph: str, max_tokens: int, counter: TokenCounter, join_tokens: int | None = None) -> list[str]:
    """
    Tách một đoạn văn dài hơn max_tokens thành các phần <= max_tokens: theo dòng, rồi theo câu, cuối cùng cắt cứng.

    Các phần nhỏ liền nhau được gom lại cho tới gần join_tokens (mặc định max_tokens; chunk_text truyền mục tiêu, xem mức
    cắt 3 ở docstring module). Cắt cứng chia đoạn thành ceil(n / max_tokens) phần đều nhau.
    """
    join_tokens = join_tokens or max_tokens
    n = counter.count(paragraph)
    if n <= max_tokens:
        return [paragraph]
    for sep, pieces in (("\n", paragraph.split("\n")), (" ", _SENTENCE_SPLIT.split(paragraph))):
        if len(pieces) > 1:
            out: list[str] = []
            for piece in pieces:
                out += _split_long(piece, max_tokens, counter, join_tokens)
            return _greedy_join(out, sep, join_tokens, counter)
    parts = -(-n // max_tokens)
    return counter.hard_split(paragraph, -(-n // parts))


def _greedy_join(pieces: list[str], sep: str, max_tokens: int, counter: TokenCounter) -> list[str]:
    """Gom các mảnh liền nhau (nối bằng sep) thành phần lớn nhất không vượt max_tokens."""
    out: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for piece, n in zip(pieces, counter.count_batch(pieces)):
        if cur and cur_n + n > max_tokens:
            out.append(sep.join(cur))
            cur, cur_n = [], 0
        cur.append(piece)
        cur_n += n
    if cur:
        out.append(sep.join(cur))
    return out


def chunk_text(text: str, counter: TokenCounter, target: int, max_tokens: int, min_tokens: int) -> list[str]:
    """
    Cắt text thành các đoạn theo quy tắc ở docstring module. Text rỗng trả về [].

    Số token của đoạn được ước bằng tổng số token các đoạn văn (ranh giới "\\n\\n" thêm khoảng 1 token mỗi chỗ nối),
    số token thật được đếm lại sau khi cắt.
    """
    paragraphs = [p for para in text.split("\n\n") if para.strip()  # strip("\n"): giữ thụt lề code (C-09)
                  for p in _split_long(para.strip("\n").rstrip(), max_tokens, counter, target)]
    groups: list[list[tuple[str, int]]] = []
    cur: list[tuple[str, int]] = []
    cur_n = 0
    for para, n in zip(paragraphs, counter.count_batch(paragraphs)):
        if cur and ((is_heading(para) and cur_n >= min_tokens) or cur_n + n > max_tokens or cur_n >= target):
            groups.append(cur)
            cur, cur_n = [], 0
        cur.append((para, n))
        cur_n += n
    if cur:
        prev = groups[-1] if groups else None
        prev_n = sum(n for _, n in prev) if prev else 0
        if prev and cur_n < min_tokens and prev_n + cur_n <= max_tokens:
            prev.extend(cur)
        else:
            # đoạn đuôi ngắn không gộp được (gộp sẽ vượt max): chuyển các đoạn văn cuối của đoạn trước sang cho tới khi
            # đuôi đủ min, miễn đoạn trước còn ít nhất một đoạn văn và đuôi không vượt max
            while prev and cur_n < min_tokens and len(prev) > 1 and cur_n + prev[-1][1] <= max_tokens:
                moved = prev.pop()
                cur.insert(0, moved)
                cur_n += moved[1]
            groups.append(cur)
    return ["\n\n".join(p for p, _ in g) for g in groups]


def chunk_rank(seed: int | str, source_key: str, parent_doc_id: str, chunk_index: int) -> str:
    """
    Khoá xếp hạng ổn định của một đoạn khi lấy mẫu (J-40 a): băm SHA-256 của (seed, nguồn, parent_doc_id, chunk_index).

    Khoá chỉ phụ thuộc vào định danh của đoạn, không phụ thuộc vào văn bản hay số đoạn của bài khác; vì vậy sửa chữ trong một bài (đổi số đoạn của bài đó) không làm
    đổi khoá, cũng không đổi việc chọn đoạn của bài khác. Lấy n đoạn có khoá nhỏ nhất thì mẫu của n nhỏ luôn nằm trong mẫu của n lớn hơn.
    """
    return hashlib.sha256(f"{seed}:{source_key}:{parent_doc_id}:{chunk_index}".encode()).hexdigest()


def sample_chunks(pieces: list[dict], n: int, seed: int | str, source_key: str) -> list[dict]:
    """
    Lấy n đoạn có chunk_rank nhỏ nhất (J-40 a), trả về theo thứ tự doc_id. Bản ghi chưa cắt (không có parent_doc_id / chunk_index) dùng doc_id và 0.
    """
    def rank(piece: dict) -> str:
        """Khoá xếp hạng của một đoạn."""
        return chunk_rank(seed, source_key, piece.get("parent_doc_id") or piece["doc_id"], piece.get("chunk_index") or 0)

    return sorted(sorted(pieces, key=rank)[:n], key=lambda r: r["doc_id"])


def _split_doc(row: dict, texts: list[str]) -> list[dict]:
    """Các bản ghi đoạn của một văn bản (doc_id "<doc_id>#c<i>", parent_doc_id, chunk_index)."""
    return [{**row, "doc_id": f"{row['doc_id']}#c{i}", "text": text, "parent_doc_id": row["doc_id"], "chunk_index": i}
            for i, text in enumerate(texts)]


def chunk_rows(rows: list[dict], cfg: RunConfig, counter: TokenCounter) -> tuple[list[dict], dict]:
    """
    Gắn vân tay văn bản gốc, cắt đoạn (xem docstring module), lấy mẫu nguồn chunked xuống cfg.mix[nguồn] đoạn (theo băm ổn định
    của seed và định danh đoạn, xem sample_chunks), rồi đếm char / word / token mọi bản ghi.

    Returns:
        (rows, thống kê theo nguồn: số văn bản, số văn bản bị cắt, số đoạn trước / sau khi lấy mẫu).
    """
    out: list[dict] = []
    stats: dict[str, dict] = {}
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        row["doc_sha256"] = text_sha256(row["text"] or "")
        row["doc_minhash"] = minhash_bytes(row["text"] or "")
        by_source.setdefault(row["source_key"], []).append(row)
    lengths = dict(zip((r["doc_id"] for r in rows), counter.count_batch([r["text"] or "" for r in rows])))
    args = (cfg.chunk_target_tokens, cfg.chunk_max_tokens, cfg.chunk_min_tokens)
    for key, docs in by_source.items():
        prof = cfg.profile(key)
        pieces: list[dict] = []
        split = 0
        for doc in docs:
            if prof.chunked or (prof.split_long and lengths[doc["doc_id"]] > cfg.chunk_max_tokens):
                parts = _split_doc(doc, chunk_text(doc["text"] or "", counter, *args))
                split += len(parts) > 1
                pieces.extend(parts)
            else:
                pieces.append(doc)
        st = {"docs": len(docs), "docs_split": split, "chunks_total": len(pieces)}
        n = cfg.mix.get(key)
        if prof.chunked and n is not None and n < len(pieces):
            pieces = sample_chunks(pieces, n, cfg.seed, key)
        st["chunks_kept"] = len(pieces)
        stats[key] = st
        out.extend(pieces)
    set_counts(out, counter)
    return out, {"version": CHUNK_VERSION, "tokenizer": counter.spec, "per_source": stats}
