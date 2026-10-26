"""
Bộ dò lỗi chữ còn lọt của văn bản tiếng Việt trích từ PDF (J-42 mục (1c), P4): năm số đo theo văn bản, chỉ CPU, không cần mô hình.

- gibberish_ratio: "rác" nhiều ký hiệu rời ("/, N9 = < 0 e H ; :" của phông dịch ký tự): tỷ lệ từ không có cụm >= 2 chữ cái Latinh (ký tự đơn, ký hiệu, chữ không Latinh).
- tone_shift_hits: chữ có dấu bị đổi thành chữ có dấu khác ("nghiờn" thay "nghiên", "Khỏnh" thay "Khánh"): từ không có trong tập âm tiết nhưng thành âm tiết thật khi đổi MỘT ký tự
  theo bảng đổi. Bảng đổi mặc định (TONE_SHIFT_MAP) là bảng ê / â / á đã thấy ở J-33 H; bảng đầy đủ do `learn_tone_shift_map` học từ dữ liệu (scripts/vjol/learn_tone_shift.py), không đoán.
- dropped_letters_runs: chữ có dấu bị rơi mất ("kh n th", "nh ng"): chuỗi >= 2 "từ" liền nhau gồm 1-2 chữ cái thường không có nguyên âm, trong đó có ít nhất một từ 2 chữ cái.
- joined_words_hits: mất khoảng trắng giữa hai âm tiết ("tươnghỗ", "Tómtắt"): từ có dấu, không có trong tập âm tiết, tách được thành hai âm tiết thật.

Tất cả là ĐO (không loại đoạn) cho tới khi precision >= 0,9 trên nhãn chuẩn (J-42 (5), vi_corpus.pipeline.quality.HARD_DETECTOR_THRESHOLDS). Ngưỡng ở đầu file là điểm khởi đầu chưa đo trên bài thật.
"""

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from vi_corpus.common.garbled import latin_tokens

# Điểm khởi đầu của các ngưỡng đoạn (chưa đo): chỉ dùng cho bộ dò của audit; ngưỡng loại đoạn nằm ở vi_corpus.pipeline.quality.
GIBBERISH_MIN_TOKENS = 30  # số "từ" tối thiểu (bỏ số / dấu câu thuần) để đo gibberish
GIBBERISH_MIN_RATIO = 0.25  # tỷ lệ từ rác từ mức này là đoạn rác
DROPPED_MIN_RUNS = 2  # số chuỗi chữ rơi mất tối thiểu
JOINED_MIN_HITS = 2  # số từ dính liền tối thiểu
SHIFT_MIN_HITS = 2  # số từ lệch dấu tối thiểu (một tên riêng lẻ tẻ không tính)
JOINED_MIN_LEN = 5  # từ dính phải dài ít nhất ngần này chữ cái
TONE_SHIFT_REL = "processed/ocr_bakeoff/tone_shift.json"  # bảng đổi học từ dữ liệu (scripts/vjol/learn_tone_shift.py), tương đối so với data-root

# Bảng đổi mặc định: khoá = ký tự sai, giá trị = ký tự đúng ("NGHIấN" -> "NGHIÊN", "nghiờn" -> "nghiên", "nhõn" -> "nhân", "Khỏnh" -> "Khánh"); J-33 H.
TONE_SHIFT_MAP = {"ấ": "ê", "ờ": "ê", "õ": "â", "ỏ": "á"}

_NEUTRAL = re.compile(r"^[\d.,;:()\[\]%+\-–—/\"'“”‘’…°*•]*$")  # số, dấu câu, gạch đầu dòng: không tính vào mẫu số của gibberish
_WORDLIKE = re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]{2,}")
_VOWELS = frozenset("aeiouyàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ")
# Đơn vị / viết tắt thường gồm 1-2 chữ cái không nguyên âm; không tính là chữ rơi mất.
_UNITS = frozenset("mm cm km kg mg ml dl pp tr tp vs st hz pm kv kw mw".split())
_VI_ALPHABET = frozenset("abcdefghijklmnopqrstuvxyzàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")


def gibberish_ratio(text: str) -> tuple[float, int]:
    """
    Tỷ lệ "từ rác" trên các từ không phải số / dấu câu thuần: từ rác là từ không có cụm >= 2 chữ cái Latinh liền nhau (ký tự đơn "e", "H", "N9", ký hiệu "=", "<", chữ Kirin / Hy Lạp...).

    Công thức dàn phẳng cũng có tỷ lệ cao; bộ dò này chỉ để đo, và nhãn chuẩn quyết định có dùng để loại không (xem docstring module).

    Returns:
        (tỷ lệ, số từ trong mẫu số); (0.0, 0) khi không có từ nào.
    """
    toks = [t for t in unicodedata.normalize("NFC", text).split() if not _NEUTRAL.match(t)]
    if not toks:
        return 0.0, 0
    return sum(not _WORDLIKE.search(t) for t in toks) / len(toks), len(toks)


def _consonant_only(token: str) -> bool:
    """True nếu `token` gồm đúng 1-2 chữ cái thường, không có nguyên âm, và không phải đơn vị / viết tắt (_UNITS)."""
    return token.isalpha() and 1 <= len(token) <= 2 and token == token.lower() and not any(c in _VOWELS for c in token) and token not in _UNITS


def dropped_letters_runs(text: str) -> int:
    """
    Số chuỗi chữ rơi mất ("kh n th", "nh ng"): chuỗi >= 2 từ liền nhau (không dấu câu dính vào) mà mỗi từ là 1-2 chữ cái thường không có nguyên âm, và có ít nhất một từ 2 chữ cái
    (chuỗi toàn chữ đơn "x y z" là ký hiệu toán).
    """
    runs = 0
    cur: list[str] = []
    for tok in unicodedata.normalize("NFC", text).split() + [""]:
        if _consonant_only(tok):
            cur.append(tok)
            continue
        runs += len(cur) >= 2 and any(len(t) == 2 for t in cur)
        cur = []
    return runs


def first_dropped_span(text: str) -> tuple[int, int] | None:
    """Vị trí (đầu, cuối) của chuỗi chữ rơi mất đầu tiên (xem dropped_letters_runs) trong `text` đã NFC; None nếu không có. Dùng để trích mẫu quanh chỗ lỗi."""
    text = unicodedata.normalize("NFC", text)
    cur: list[re.Match] = []
    for m in list(re.finditer(r"\S+", text)) + [None]:
        if m is not None and _consonant_only(m.group(0)):
            cur.append(m)
            continue
        if len(cur) >= 2 and any(len(x.group(0)) == 2 for x in cur):
            return cur[0].start(), cur[-1].end()
        cur = []
    return None


def first_gibberish_span(text: str) -> tuple[int, int] | None:
    """Vị trí (đầu, cuối) của từ rác đầu tiên (từ không phải số / dấu câu thuần mà không có cụm >= 2 chữ cái Latinh) trong `text` đã NFC; None nếu không có. Dùng để trích mẫu."""
    text = unicodedata.normalize("NFC", text)
    for m in re.finditer(r"\S+", text):
        if not _NEUTRAL.match(m.group(0)) and not _WORDLIKE.search(m.group(0)):
            return m.start(), m.end()
    return None


def joined_words_hits(text: str, vocab: frozenset[str]) -> int:
    """
    Số từ dính liền hai âm tiết ("tươnghỗ" = "tương" + "hỗ"): từ Latinh >= JOINED_MIN_LEN chữ cái, CÓ dấu / chữ đặc biệt (từ ASCII toàn chữ không xét vì tiếng Anh ghép sẵn nhiều),
    không có trong `vocab` và tách được thành hai phần, mỗi phần >= 2 chữ cái và đều là âm tiết trong `vocab`.
    """
    hits = 0
    for tok in latin_tokens(text):
        low = tok.lower()
        if len(low) < JOINED_MIN_LEN or low.isascii() or low in vocab:
            continue
        hits += any(low[:i] in vocab and low[i:] in vocab for i in range(2, len(low) - 1))
    return hits


def tone_shift_hits(text: str, vocab: frozenset[str], mapping: Mapping[str, str] | None = None) -> int:
    """
    Số từ lệch dấu: từ Latinh KHÔNG có trong `vocab` nhưng thành âm tiết thật khi đổi MỘT ký tự theo `mapping` (khoá = ký tự sai, giá trị = ký tự đúng; mặc định TONE_SHIFT_MAP).
    """
    table = TONE_SHIFT_MAP if mapping is None else mapping
    hits = 0
    for tok in latin_tokens(text):
        low = tok.lower()
        if low in vocab:
            continue
        hits += any(c in table and low[:i] + table[c] + low[i + 1:] in vocab for i, c in enumerate(low))
    return hits


def load_tone_shift_map(path: Path) -> dict[str, str] | None:
    """Đọc bảng đổi lệch dấu do learn_tone_shift_map sinh (khoá "map"); None nếu file không tồn tại."""
    path = Path(path)
    if not path.is_file():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8"))["map"])


def learn_tone_shift_map(texts: Iterable[str], vocab: frozenset[str], min_count: int = 30, min_share: float = 0.3, top: int = 60) -> dict:
    """
    Học bảng đổi lệch dấu từ dữ liệu (J-42 (1c): "chọn bảng đổi bằng dữ liệu, không đoán").

    Với mỗi từ Latinh không có trong `vocab` mà đúng MỘT phép đổi một ký tự (c -> c', cả hai là chữ cái tiếng Việt) cho âm tiết thật, đếm cặp (c, c'). Giữ cặp có count >= min_count và
    share >= min_share (share = số lần cặp đó / số từ lạ có chứa ký tự c): ký tự sai thật (ỏ thay á) hầu như chỉ đổi theo một cách, còn lỗi gõ ngẫu nhiên thì rải đều nhiều cặp.
    Mỗi ký tự sai chỉ giữ cặp có count lớn nhất.

    Returns:
        {"map": {ký tự sai: ký tự đúng}, "pairs": [{"wrong", "right", "count", "share"}] (tối đa `top` cặp theo count giảm dần), "unknown_tokens": số từ lạ đã xét}.
    """
    pairs: Counter = Counter()
    contain: Counter = Counter()
    unknown = 0
    for text in texts:
        for tok in latin_tokens(text):
            low = tok.lower()
            if low in vocab:
                continue
            unknown += 1
            for c in set(low):
                contain[c] += 1
            found = {(c, alt) for i, c in enumerate(low) if c in _VI_ALPHABET for alt in _VI_ALPHABET if alt != c and low[:i] + alt + low[i + 1:] in vocab}
            if len(found) == 1:
                pairs[next(iter(found))] += 1
    rows = [{"wrong": c, "right": alt, "count": n, "share": round(n / contain[c], 3)} for (c, alt), n in pairs.most_common()]
    chosen: dict[str, str] = {}
    for row in rows:
        if row["count"] >= min_count and row["share"] >= min_share and row["wrong"] not in chosen:
            chosen[row["wrong"]] = row["right"]
    return {"map": chosen, "pairs": rows[:top], "unknown_tokens": unknown}
