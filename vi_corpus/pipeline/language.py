"""
Stage language: nhận diện ngôn ngữ THEO ĐOẠN VĂN bằng fastText lid.176 (D-02, R-20), gắn lang_mix.

Cách tính (docs/DECISION_LOG.md, D-02):
1. Tách văn bản thành các đoạn văn (theo dòng trống; văn bản chỉ có một đoạn thì theo dòng). Văn bản rất dài chỉ lấy tối
   đa cfg.lang_max_segments đoạn rải đều để giữ chi phí cố định. Đoạn không có chữ cái bị bỏ.
2. Bộ nhận diện đoán từng đoạn: (nhãn, độ tin cậy).
3. lang_mix = phần độ dài (ký tự) thuộc mỗi ngôn ngữ; language = ngôn ngữ chiếm nhiều nhất;
   lang_score = tổng (độ dài x độ tin cậy) của các đoạn thuộc `language` / tổng độ dài. Cao khi văn bản vừa thuần một
   ngôn ngữ vừa được đoán chắc chắn. Ngưỡng min_lang_score nằm ở SourceProfile, được dùng ở stage quality.
Văn bản không có chữ cái: language = "other", lang_score = 0, lang_mix = {}.

SEA-Instruct: chỉ đưa nội dung các lượt không phải "system" vào nhận diện (bỏ lượt system và tiền tố vai "human: "),
vì lời nhắc hệ thống tiếng Anh ("You are an AI assistant...") làm cả hội thoại bị gọi là tiếng Anh (M-03).

Bộ nhận diện chọn bằng cfg.lang_model:
- "fasttext:lid.176": tải lid.176.bin (Facebook, ~126 MB) một lần vào <cache>/lid.176.bin (có khoá, ghi .part rồi đổi
  tên), <cache> = biến môi trường VI_CORPUS_MODEL_DIR hoặc ~/.cache/vi_corpus. "fasttext:<đường dẫn .bin>" dùng file có sẵn.
- "heuristic": tỷ lệ chữ có dấu tiếng Việt + stopword Việt / Anh (bản cũ), không cần mô hình; dùng cho test / máy không
  có mạng. Nhãn chỉ có "vi" / "en" / "other".
Nhãn của fastText là mã ISO (vi, en, zh, ...). Mô hình được nạp một lần mỗi tiến trình.
"""

import fcntl
import json
import os
import re
from functools import lru_cache
from pathlib import Path

LANG_VERSION = "2"  # 1 = heuristic cả văn bản; 2 = theo đoạn + lang_mix
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_MIN_BYTES = 100_000_000  # file lid.176.bin đủ: 131.266.198 byte; nhỏ hơn mức này coi như tải dở

_VI_CHARS = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
_VI_STOP = frozenset("và của là có không được những một các trong cho với này để người đã khi từ ra như cũng nhiều "
                     "theo về lại rất còn hay bị đến sẽ họ tôi chúng".split())
_EN_STOP = frozenset("the of and to in is that for it with as was on are by be this from or an at which not have "
                     "has had were been their its but they his her you".split())
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_SYSTEM_ROLES = frozenset({"system"})

MAX_CHARS = 20_000  # heuristic: chỉ xét phần đầu văn bản
MAX_SEGMENT_CHARS = 2_000  # mỗi đoạn chỉ đưa tối đa chừng này ký tự vào bộ nhận diện
MIXED_SHARE = 0.2  # ngôn ngữ thứ hai chiếm từ tỷ lệ này trở lên thì coi là văn bản trộn ngôn ngữ (mã mixed_language)


def detect_language(text: str) -> tuple[str, float]:
    """
    Heuristic nhận diện ngôn ngữ của một đoạn (không cần mô hình).

    Returns:
        (nhãn "vi" | "en" | "other", điểm tin cậy 0..1). Văn bản không có chữ cái trả ("other", 0.0).
    """
    sample = text[:MAX_CHARS].lower()
    words = _WORD.findall(sample)
    letters = sum(len(w) for w in words)
    if not words or letters == 0:
        return "other", 0.0
    diacritic = sum(ch in _VI_CHARS for w in words for ch in w) / letters
    vi_stop = sum(w in _VI_STOP for w in words) / len(words)
    en_stop = sum(w in _EN_STOP for w in words) / len(words)
    if diacritic >= 0.08 or (vi_stop >= 0.08 and vi_stop > en_stop):
        return "vi", round(min(1.0, 0.5 * min(1.0, diacritic / 0.2) + 0.5 * min(1.0, vi_stop / 0.15)), 3)
    if en_stop >= 0.12 and diacritic < 0.03:
        return "en", round(min(1.0, en_stop / 0.25), 3)
    return "other", round(max(vi_stop, en_stop), 3)


def vi_diacritic_ratio(text: str) -> float:
    """Tỷ lệ chữ cái có dấu đặc trưng tiếng Việt trong mọi chữ cái (số đo chất lượng vi_diacritic, bắt tiếng Việt không dấu)."""
    words = _WORD.findall(text[:MAX_CHARS].lower())
    letters = sum(len(w) for w in words)
    return sum(ch in _VI_CHARS for w in words for ch in w) / letters if letters else 0.0


class HeuristicLid:
    """Bộ nhận diện heuristic, cùng giao diện predict với FastTextLid."""

    spec = "heuristic"

    def predict(self, segments: list[str]) -> list[tuple[str, float]]:
        """(nhãn, độ tin cậy) cho từng đoạn."""
        return [detect_language(s) for s in segments]


class FastTextLid:
    """Bộ nhận diện fastText (lid.176.bin)."""

    def __init__(self, path: str) -> None:
        """Nạp mô hình fastText từ file .bin."""
        import fasttext

        self.spec = f"fasttext:{path}"
        self.model = fasttext.load_model(path)

    def predict(self, segments: list[str]) -> list[tuple[str, float]]:
        """(nhãn ISO, độ tin cậy) cho từng đoạn; fastText cần mỗi đoạn trên một dòng."""
        labels, probs = self.model.predict([" ".join(s.split()) for s in segments], k=1)
        return [(lab[0].replace("__label__", ""), min(1.0, float(p[0]))) for lab, p in zip(labels, probs)]


def model_dir() -> Path:
    """Thư mục cache mô hình: VI_CORPUS_MODEL_DIR hoặc ~/.cache/vi_corpus."""
    return Path(os.environ.get("VI_CORPUS_MODEL_DIR") or Path.home() / ".cache" / "vi_corpus")


def ensure_lid_model(url: str = LID_URL) -> Path:
    """
    Đảm bảo lid.176.bin có đủ trong model_dir(), tải nếu thiếu hoặc tải dở (khoá flock, ghi .part rồi đổi tên nguyên tử,
    an toàn khi nhiều tiến trình / actor cùng gọi).

    Raises:
        RuntimeError: file tải về nhỏ bất thường.
    """
    import requests

    path = model_dir() / "lid.176.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # tự nhả khi đóng file
        if path.exists() and path.stat().st_size >= LID_MIN_BYTES:
            return path
        part = Path(f"{path}.part")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        if part.stat().st_size < LID_MIN_BYTES:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"File mô hình tải từ {url} quá nhỏ (tải dở?)")
        os.replace(part, path)
    return path


@lru_cache(maxsize=4)
def get_lid(spec: str):
    """
    Bộ nhận diện dùng chung trong tiến trình cho một spec ("heuristic", "fasttext:lid.176", "fasttext:<đường dẫn>").

    Raises:
        ValueError: spec không hợp lệ.
    """
    if spec == "heuristic":
        return HeuristicLid()
    if spec.startswith("fasttext:"):
        name = spec[len("fasttext:"):]
        return FastTextLid(str(ensure_lid_model() if name == "lid.176" else Path(name)))
    raise ValueError(f"lang_model không hợp lệ: {spec!r} (dùng 'fasttext:lid.176', 'fasttext:<file.bin>' hoặc 'heuristic')")


def lid_text(row: dict) -> str:
    """Phần văn bản đưa vào nhận diện: với hội thoại (meta.turns) là nội dung các lượt không phải system."""
    meta = row.get("meta")
    if meta:
        try:
            turns = json.loads(meta).get("turns") or []
        except (json.JSONDecodeError, AttributeError):
            turns = []
        if turns:
            return "\n\n".join(t["content"] for t in turns if str(t.get("role", "")).lower() not in _SYSTEM_ROLES)
    return row["text"] or ""


def segments_of(text: str, max_segments: int) -> list[str]:
    """Các đoạn văn có chữ cái (xem bước 1 ở docstring module), rải đều tối đa max_segments đoạn."""
    parts = [p for p in text.split("\n\n") if p.strip()]
    if len(parts) <= 1:
        parts = [p for p in text.split("\n") if p.strip()]
    parts = [p[:MAX_SEGMENT_CHARS] for p in parts if _WORD.search(p)]
    if len(parts) > max_segments:
        step = len(parts) / max_segments
        parts = [parts[int(i * step)] for i in range(max_segments)]
    return parts


def detect_mix(text: str, lid, max_segments: int = 50) -> tuple[str, float, dict[str, float]]:
    """
    Nhận diện theo đoạn (xem docstring module).

    Returns:
        (language, lang_score, lang_mix).
    """
    segs = segments_of(text, max_segments)
    if not segs:
        return "other", 0.0, {}
    weight: dict[str, float] = {}
    conf: dict[str, float] = {}
    total = 0
    for seg, (label, p) in zip(segs, lid.predict(segs)):
        n = len(seg)
        total += n
        weight[label] = weight.get(label, 0) + n
        conf[label] = conf.get(label, 0) + n * p
    lang = max(weight, key=lambda k: (weight[k], k))
    mix = {k: round(v / total, 4) for k, v in sorted(weight.items(), key=lambda kv: -kv[1])}
    return lang, round(conf[lang] / total, 4), mix


def annotate_language(rows: list[dict], lang_model: str = "heuristic", max_segments: int = 50) -> list[dict]:
    """
    Điền language, lang_score, lang_mix (JSON) cho mọi bản ghi (sửa tại chỗ, trả lại chính list đó).

    Hàm thuần theo từng bản ghi nên chạy song song được (vi_corpus.pipeline.curator).
    """
    lid = get_lid(lang_model)
    for row in rows:
        lang, score, mix = detect_mix(lid_text(row), lid, max_segments)
        row["language"], row["lang_score"], row["lang_mix"] = lang, score, json.dumps(mix)
    return rows


def language_stats(rows: list[dict]) -> dict:
    """Thống kê của stage language: số bản ghi theo từng nhãn ngôn ngữ, số bản ghi trộn ngôn ngữ (xem is_mixed)."""
    counts: dict[str, int] = {}
    mixed = 0
    for row in rows:
        counts[row["language"]] = counts.get(row["language"], 0) + 1
        mixed += is_mixed(row["lang_mix"])
    return {"version": LANG_VERSION, "counts": counts, "mixed": mixed}


def is_mixed(lang_mix: str | None) -> bool:
    """True nếu ngôn ngữ thứ hai trong lang_mix (JSON) chiếm >= MIXED_SHARE."""
    shares = sorted(json.loads(lang_mix or "{}").values(), reverse=True)
    return len(shares) > 1 and shares[1] >= MIXED_SHARE
