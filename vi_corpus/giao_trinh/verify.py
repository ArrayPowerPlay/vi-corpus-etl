"""
Kiểm chứng bộ dò chữ hỏng do font trên giáo trình bằng OCR (GT-06): trang bị gắn cờ có thật sự hỏng không?

Bộ dò của VJOL (vi_corpus.common.garbled) chỉnh cho bài báo tiếng Việt nên báo nhầm trên giáo trình (sách tiếng Anh, code, công thức). Thay vì đoán ngưỡng,
công cụ này lấy mẫu phân tầng các trang bị gắn cờ cùng các trang đối chứng không bị gắn cờ, OCR lại bằng chuỗi VLM rồi đo độ trùng từ (Dice trên túi từ,
vi_corpus.vjol.font_canary.word_overlap) giữa lớp chữ và kết quả OCR: trùng cao = lớp chữ đúng (gắn cờ nhầm), trùng thấp = lớp chữ hỏng thật.

- collect_pages: đọc checkpoint của lần trích có bật bộ dò (`flagged_dir`) và của lần trích trước đó (`before_dir`), gán mỗi trang một tầng.
- pick_sample: chọn tối đa N trang mỗi tầng, ổn định theo hạt giống.
- summarize: thống kê độ trùng theo tầng.
Phần chạy OCR nằm ở scripts/giao_trinh/verify_flagged.py.
"""

import hashlib
import statistics
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from vi_corpus.common.state import read_json

# Ký tự chỉ có trong tiếng Việt (nguyên âm có mũ / móc / dấu thanh và đ): mật độ của chúng trên số chữ cái cho biết trang có phải tiếng Việt hay không.
_VI_CHARS = frozenset("ăâêôơưđĂÂÊÔƠƯĐàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
                      "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ")
VI_MIN_DENSITY = 0.01  # từ ngưỡng này trở lên coi là trang tiếng Việt (đo trên mẫu giáo trình: tiếng Anh / code gần 0)
STRATA = ("flag_tcvn3", "flag_text_vi", "flag_text_en", "ctrl_vi", "ctrl_en")
# flag_tcvn3: trang bị gắn cờ mà lần trích trước gán nhãn tcvn3; flag_text_vi / flag_text_en: bị gắn cờ, lần trước là text, mật độ chữ Việt >= / < VI_MIN_DENSITY;
# ctrl_vi / ctrl_en: đối chứng, trang text không bị gắn cờ (cùng chia theo mật độ chữ Việt).


def vi_density(text: str) -> float:
    """Tỷ lệ chữ cái tiếng Việt đặc trưng (có dấu, ă â ê ô ơ ư đ) trên tổng số chữ cái; 0 nếu không có chữ cái."""
    letters = sum(c.isalpha() for c in text)
    return sum(c in _VI_CHARS for c in text) / letters if letters else 0.0


def stratum_of(before_method: str, flagged: bool, text: str) -> str | None:
    """
    Tầng của một trang, None nếu không thuộc tầng nào.

    Args:
        before_method: method của trang ở lần trích trước khi bật bộ dò ("text", "tcvn3", ...).
        flagged:       Trang bị bộ dò gắn cờ (needs_ocr vì lỗi font) ở lần trích có bật bộ dò.
        text:          Lớp chữ của trang.
    """
    vi = vi_density(text) >= VI_MIN_DENSITY
    if flagged:
        if before_method == "tcvn3":
            return "flag_tcvn3"
        return ("flag_text_vi" if vi else "flag_text_en") if before_method == "text" else None
    if before_method == "text":
        return "ctrl_vi" if vi else "ctrl_en"
    return None


def collect_pages(flagged_dir: Path, before_dir: Path) -> Iterator[dict]:
    """
    Duyệt checkpoint PDF của hai lần trích (cùng sha256, cùng số trang) và sinh mỗi trang thuộc một tầng.

    Mỗi phần tử: {"sha256", "path" (đường dẫn file đầu tiên, tương đối so với data_root), "page", "stratum", "text" (lớp chữ),
    "density"}. Trang flagged có text là lớp chữ hỏng giữ lại trong checkpoint (reason garbled_text_layer); trang không bị gắn cờ lấy text của
    lần trích có bật bộ dò. Checkpoint thiếu bản trước, không phải PDF, hoặc lệch số trang bị bỏ.
    """
    for ckpt in sorted(Path(flagged_dir).glob("*.json")):
        now, before = read_json(ckpt), read_json(Path(before_dir) / ckpt.name)
        if not now or not before or now.get("format") != "pdf" or len(now["pages"]) != len(before["pages"]):
            continue
        for i, (pn, pb) in enumerate(zip(now["pages"], before["pages"])):
            flagged = pn["method"] == "needs_ocr" and pn.get("reason") == "garbled_text_layer"
            if not flagged and pn["method"] != "text":
                continue
            stratum = stratum_of(pb["method"], flagged, pn["text"])
            if stratum:
                yield {"sha256": ckpt.stem, "path": now["paths"][0], "page": i, "stratum": stratum, "text": pn["text"],
                       "density": round(vi_density(pn["text"]), 4)}


def _key(seed: int, row: dict) -> str:
    """Khoá sắp xếp ổn định theo hạt giống của một trang (băm), để mẫu không đổi giữa các lần chạy."""
    return hashlib.sha1(f"{seed}:{row['sha256']}:{row['page']}".encode()).hexdigest()


def pick_sample(rows: Iterable[dict], per_stratum: int, seed: int = 0) -> list[dict]:
    """Chọn tối đa `per_stratum` trang mỗi tầng theo băm (seed, sha256, trang); trả về theo thứ tự băm để các trang của một lô rải đều nhiều file."""
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stratum[r["stratum"]].append(r)
    picked = [r for s in STRATA for r in sorted(by_stratum.get(s, []), key=lambda r: _key(seed, r))[:per_stratum]]
    return sorted(picked, key=lambda r: _key(seed, r))


def summarize(results: Iterable[dict]) -> dict:
    """
    Thống kê độ trùng từ giữa lớp chữ và OCR theo tầng.

    Args:
        results: Mỗi phần tử có "stratum", "status" (trạng thái OCR) và "overlap" (None nếu OCR không ra chữ).

    Returns:
        {tầng: {"n", "ocr_ok", "median", "p25", "p75", "share_ge_0.8" (lớp chữ khớp OCR: nhiều khả năng gắn cờ nhầm), "share_le_0.4" (lớp chữ lệch OCR: nhiều
        khả năng hỏng thật)}}; các thống kê chỉ tính trên trang OCR ra chữ.
    """
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_stratum[r["stratum"]].append(r)
    out = {}
    for s in STRATA:
        rows = by_stratum.get(s, [])
        ov = sorted(r["overlap"] for r in rows if r["overlap"] is not None)
        entry = {"n": len(rows), "ocr_ok": len(ov)}
        if ov:
            q = statistics.quantiles(ov, n=4) if len(ov) >= 2 else [ov[0]] * 3
            entry |= {"median": round(statistics.median(ov), 3), "p25": round(q[0], 3), "p75": round(q[2], 3),
                      "share_ge_0.8": round(sum(x >= 0.8 for x in ov) / len(ov), 3), "share_le_0.4": round(sum(x <= 0.4 for x in ov) / len(ov), 3)}
        out[s] = entry
    return out
