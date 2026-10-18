"""
Kiểm kết quả OCR stbook bằng chuỗi VLM sau khi chạy xong (chỉ dùng CPU, không cần GPU hay server vLLM).

Đọc mọi file `interim/stbook_ocr/<slug>/<product_id>.json` do chuỗi VLM ghi và log của lượt chạy, rồi ghi một thư mục nhỏ
`processed/ocr_vlm_check/` để tải về xem:
    summary.json        : số liệu tổng (tính nhất quán, chữ/trang theo engine, tỷ lệ dấu tiếng Việt, tốc độ, cảnh báo trong log,
                          loại chữ viết và ngôn ngữ theo engine, sách ngoại ngữ, trang lạ trong sách tiếng Việt, phân bố tỷ lệ âm tiết thật).
    books.csv           : mỗi cuốn một dòng (số trang, số trang theo engine, chạy lại, giây, trang/giây, chữ/trang, dấu, tỷ lệ trang tiếng Việt).
    suspect_pages.csv   : mọi trang đáng nghi (ít dấu tiếng Việt, lẫn chữ nước ngoài, lặp, ảnh nhúng, quá dài, chữ dính liền, từ bịa dài)
                          kèm mã lý do (hai mã cuối là O-30 (4): chỉ để chọn ngưỡng, chưa dùng lọc trang).
    foreign_books.csv   : sách mà phần lớn trang không phải tiếng Việt (bảng sách ngoại ngữ, O-27 a).
    odd_pages_in_vi_books.csv : trang không phải tiếng Việt nằm trong sách mà các trang còn lại là tiếng Việt (nghi bịa chữ, O-27 b).
    baseline_pages.csv  : mọi trang baseline kèm tỷ lệ âm tiết thật để chọn ngưỡng lọc (O-27 c); cần syllables.json.
    samples/<nhóm>/     : trang mẫu theo engine, theo trang đáng nghi, theo trang lạ trong sách tiếng Việt và theo dải tỷ lệ âm tiết thật
                          của baseline; mỗi trang gồm ảnh .jpg và văn bản .txt để so bằng mắt.
    logs/               : bản sao log của từng GPU (stbook_ocr_vlm_gpu*.out).
Các ngưỡng "đáng nghi" chỉ là gợi ý để chọn trang xem bằng mắt, không phải cổng loại bỏ.
"""

import csv
import json
import logging
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from vi_corpus.common.state import read_json
from vi_corpus.ocr_bakeoff.guard import MAX_CHARS, repeat_stats
from vi_corpus.ocr_bakeoff.syllables import load_syllables, unknown_rate
from vi_corpus.pipeline.language import MAX_SEGMENT_CHARS, get_lid
from vi_corpus.stbook.ocr_books import OCR_DIR
from vi_corpus.stbook.ocr_vlm import ENGINE_TAG

logger = logging.getLogger("vi_corpus")

CHECK_DIR = "processed/ocr_vlm_check"
_LOWER_DIAC = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
_DIAC = re.compile("[" + _LOWER_DIAC + _LOWER_DIAC.upper() + "]")
_LETTER = re.compile(r"[^\W\d_]")
_NON_LATIN = re.compile("[Ѐ-ӿ؀-ۿ฀-๿぀-ヿ㐀-鿿가-힯]")
_LETTER_RUN = re.compile(r"[^\W\d_]+")
_IMAGE_MARK = ("data:image", "base64,")
_LOG_LINE = re.compile(r"\] (\d+) trang, ([\d.]+) trang/s")

LOW_DIAC_MAX = 0.10  # tiếng Việt thường có ~25-30% chữ cái mang dấu; dưới mức này thường là mất dấu hoặc trang không phải tiếng Việt
LOW_DIAC_MIN_LETTERS = 300  # trang quá ít chữ thì tỷ lệ dấu không đáng tin
NON_LATIN_MAX = 0.02  # chữ Hán / Kirin / Thái ... quá 2% số chữ cái là dấu hiệu mô hình bịa chữ
LOOP_COVER_MAX = 0.20  # dãy lặp phủ quá 20% trang
GLUED_MEAN_WORD_LEN = 9.0  # độ dài trung bình một cụm chữ cái (tiếng Việt / Anh ~4-5) vượt mức này là chữ dính liền không dấu cách (O-30 4a)
GLUED_MIN_LETTERS = 200  # trang ít chữ cái hơn mức này không xét chữ dính liền
LONG_WORD_LETTERS = 18  # cụm chữ cái dài từ mức này là từ bịa (vd "CONTRANSITIONALIZATION", 22) hoặc địa chỉ web (O-30 4b)
LOG_MESSAGES_CAP = 200  # tối đa số dòng cảnh báo / lỗi chép vào summary.json
LANG_MIN_LETTERS = 100  # trang ít chữ cái hơn mức này không nhận diện ngôn ngữ (kết quả không đáng tin)
VI_BOOK_MIN_SHARE = 0.8  # sách có từ tỷ lệ này số trang (đủ chữ) là tiếng Việt thì coi là sách tiếng Việt: trang lạ trong đó là nghi bịa
FOREIGN_BOOK_MAX_SHARE = 0.5  # sách có dưới tỷ lệ này số trang là tiếng Việt thì vào bảng sách ngoại ngữ
SYL_MIN_WORDS = 30  # trang ít từ chữ cái hơn mức này không tính tỷ lệ âm tiết thật
SYL_BINS = 20  # số cột của phân bố tỷ lệ âm tiết thật trên [0, 1]
SYL_SAMPLE_EDGES = (0.5, 0.7, 0.85)  # chia trang baseline thành 4 dải tỷ lệ âm tiết thật để lấy trang mẫu xem bằng mắt
DEFAULT_LANG_MODEL = "fasttext:lid.176"
# Chữ viết: mỗi nhóm một lớp ký tự; trang gắn nhãn theo nhóm có nhiều chữ cái nhất.
_SCRIPTS = {
    "latin": re.compile("[A-Za-zÀ-ɏḀ-ỿ]"),
    "han": re.compile("[㐀-鿿]"),
    "cyrillic": re.compile("[Ѐ-ӿ]"),
    "arabic": re.compile("[؀-ۿ]"),
    "thai": re.compile("[฀-๿]"),
    "lao": re.compile("[຀-໿]"),
    "khmer": re.compile("[ក-៿]"),
    "kana_hangul": re.compile("[぀-ヿ가-힯]"),
}


def page_metrics(text: str) -> dict:
    """
    Số đo rẻ của một trang để phát hiện trang đáng nghi.

    Hai mã cuối ("glued_text", "long_word", O-30 4) chỉ xét trang chữ Latin (chữ Thái, Lào không có dấu cách theo cách viết) và chỉ dùng để
    chọn ngưỡng sau, chưa dùng để loại trang.

    Returns:
        {"chars", "letters", "diac_ratio" (chữ cái mang dấu / chữ cái), "non_latin_ratio", "loop_cover" (tỷ lệ phủ của dãy lặp),
        "script" (xem script_of), "mean_word_len" (chữ cái trung bình mỗi cụm chữ cái liền nhau), "long_words" (số cụm dài từ LONG_WORD_LETTERS),
        "reasons": [mã lý do đáng nghi]}.
    """
    letters = len(_LETTER.findall(text))
    diac = len(_DIAC.findall(text))
    non_latin = len(_NON_LATIN.findall(text))
    cover, _ = repeat_stats(text)
    runs = [len(w) for w in _LETTER_RUN.findall(text)]
    m = {"chars": len(text), "letters": letters, "diac_ratio": diac / letters if letters else 0.0,
         "non_latin_ratio": non_latin / letters if letters else 0.0, "loop_cover": cover, "script": script_of(text),
         "mean_word_len": letters / len(runs) if runs else 0.0, "long_words": sum(n >= LONG_WORD_LETTERS for n in runs)}
    reasons = []
    if letters >= LOW_DIAC_MIN_LETTERS and m["diac_ratio"] < LOW_DIAC_MAX:
        reasons.append("low_diacritic")
    if letters and m["non_latin_ratio"] > NON_LATIN_MAX:
        reasons.append("non_latin")
    if cover > LOOP_COVER_MAX:
        reasons.append("loop")
    if any(k in text for k in _IMAGE_MARK):
        reasons.append("embedded_image")
    if len(text) > MAX_CHARS:
        reasons.append("too_long")
    if m["script"] == "latin":
        if letters >= GLUED_MIN_LETTERS and m["mean_word_len"] > GLUED_MEAN_WORD_LEN:
            reasons.append("glued_text")
        if m["long_words"]:
            reasons.append("long_word")
    m["reasons"] = reasons
    return m


def script_of(text: str) -> str:
    """Loại chữ viết chiếm nhiều chữ cái nhất trong trang ("latin", "han", "cyrillic", "arabic", "thai", "lao", "khmer", "kana_hangul"); "none" nếu không có."""
    latin = len(_SCRIPTS["latin"].findall(text))
    if latin * 2 > len(text):  # chữ Latin chiếm quá nửa ký tự: không cần đếm các nhóm còn lại (đa số trang, đỡ tốn thời gian trên 366k trang)
        return "latin"
    counts = {name: len(rx.findall(text)) for name, rx in _SCRIPTS.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] else "none"


def _percentiles(values: list[float]) -> dict | None:
    """Số trang, trung bình và các phân vị p1/p5/p10/p25/p50/p75/p90 của một danh sách số (None nếu rỗng)."""
    if not values:
        return None
    v = sorted(values)
    pick = lambda q: round(v[min(len(v) - 1, int(len(v) * q))], 4)  # noqa: E731
    return {"n": len(v), "mean": round(sum(v) / len(v), 4), **{f"p{int(q * 100)}": pick(q) for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)}}


def _histogram(values: list[float]) -> dict:
    """Đếm số trang theo SYL_BINS cột đều nhau trên [0, 1] (giá trị 1.0 vào cột cuối)."""
    h = Counter(min(int(x * SYL_BINS), SYL_BINS - 1) for x in values)
    return {f"{b / SYL_BINS:.2f}-{(b + 1) / SYL_BINS:.2f}": h.get(b, 0) for b in range(SYL_BINS)}


def detect_page_languages(texts: list[str], lid) -> list[tuple[str, float] | None]:
    """
    Ngôn ngữ từng trang trong một lượt gọi (fastText nhanh khi nhận theo lô); trang ít hơn LANG_MIN_LETTERS chữ cái trả None.

    Dùng chung cho bộ kiểm (`_scan`) và bộ lọc trang (vi_corpus.stbook.ocr_filter) để hai nơi cùng một quy tắc.

    Returns:
        Danh sách cùng độ dài `texts`: (nhãn ngôn ngữ, độ tin cậy) hoặc None.
    """
    idx = [i for i, t in enumerate(texts) if len(_LETTER.findall(t)) >= LANG_MIN_LETTERS]
    out: list[tuple[str, float] | None] = [None] * len(texts)
    if idx:
        for i, res in zip(idx, lid.predict([texts[i][:MAX_SEGMENT_CHARS] for i in idx])):
            out[i] = res
    return out


def _reservoir(store: dict, seen: Counter, key: str, item: tuple, k: int, rng: random.Random) -> None:
    """Reservoir sampling: mỗi phần tử của nhóm `key` có cùng xác suất vào danh sách store[key] (tối đa k phần tử)."""
    seen[key] += 1
    if len(store[key]) < k:
        store[key].append(item)
    elif (j := rng.randrange(seen[key])) < k:
        store[key][j] = item


def _scan(data_root: Path, per_tier: int, n_suspect_samples: int, rng: random.Random, lid, vocab: frozenset[str] | None) -> dict:
    """
    Duyệt mọi file OCR của chuỗi VLM, gom số đo theo cuốn / theo engine và chọn trang mẫu (reservoir sampling).

    Mỗi cuốn được đọc trọn một lần: số đo từng trang, loại chữ viết, ngôn ngữ (lid, theo lô cả cuốn), tỷ lệ âm tiết thật (nếu có
    vocab) rồi mới gộp ở mức cuốn để tìm sách ngoại ngữ và trang lạ nằm trong sách tiếng Việt.

    Returns:
        Từ điển: books, tiers, suspects, problems, samples {nhóm: [(đường dẫn json, chỉ số trang, ghi chú)]}, suspect_total,
        scripts / langs {engine: Counter}, foreign_books, odd_pages, odd_by {tier: Counter}, baseline_pages, syl {engine: [tỷ lệ]}.
    """
    books, suspects, problems, foreign_books, odd_pages, baseline_pages = [], [], [], [], [], []
    tiers: dict[str, Counter] = defaultdict(Counter)
    scripts: dict[str, Counter] = defaultdict(Counter)
    langs: dict[str, Counter] = defaultdict(Counter)
    odd_by: dict[str, Counter] = defaultdict(Counter)
    syl: dict[str, list] = defaultdict(list)
    samples: dict[str, list] = defaultdict(list)
    seen: Counter = Counter()
    n_susp = 0
    for path in sorted((data_root / OCR_DIR).glob("*/*.json")):
        done = read_json(path)
        if not done or (done.get("ocr") or {}).get("engine") != ENGINE_TAG:
            problems.append(f"{path.parent.name}/{path.stem}: không phải kết quả của chuỗi VLM")
            continue
        name = f"{path.parent.name}/{path.stem}"
        pages, info, st = done["pages"], done["page_info"], done["stats"]
        if not (len(pages) == len(info) == st["pages"]):
            problems.append(f"{name}: lệch số trang pages={len(pages)} page_info={len(info)} stats={st['pages']}")
        empty_ok = 0
        tot = Counter()
        rows = []  # số đo từng trang của cuốn này, dùng lại sau khi biết ngôn ngữ cả cuốn
        for i, (text, pi) in enumerate(zip(pages, info)):
            tier = pi.get("tier", "?") if pi.get("status") == "ok" else f"({pi.get('status')})"
            if pi.get("status") == "ok" and not text.strip():
                empty_ok += 1
            m = page_metrics(text)
            m.update(i=i, tier=tier, text=text)
            rows.append(m)
            for k in ("chars", "letters"):
                tot[k] += m[k]
                tiers[tier][k] += m[k]
            tot["diac"] += m["diac_ratio"] * m["letters"]
            tiers[tier]["diac"] += m["diac_ratio"] * m["letters"]
            tiers[tier]["pages"] += 1
            scripts[tier][m["script"]] += 1
            for r in m["reasons"]:
                tiers[tier]["suspect_" + r] += 1
        # Ngôn ngữ cả cuốn trong một lượt gọi (fastText nhanh khi nhận theo lô).
        for r, res in zip(rows, detect_page_languages([r["text"] for r in rows], lid)):
            if res:
                r["lang"], r["lang_conf"] = res[0], round(res[1], 3)
        judged = [r for r in rows if "lang" in r]
        vi_share = sum(r["lang"] == "vi" for r in judged) / len(judged) if judged else None
        book_langs = Counter(r["lang"] for r in judged)
        if vi_share is not None and vi_share < FOREIGN_BOOK_MAX_SHARE:
            book_scripts = Counter(r["script"] for r in rows if r["letters"])
            foreign_books.append({"book": name, "pages": len(rows), "judged_pages": len(judged), "vi_share": round(vi_share, 3),
                                  "top_langs": json.dumps(dict(book_langs.most_common(3)), ensure_ascii=False),
                                  "top_scripts": json.dumps(dict(book_scripts.most_common(3))),
                                  "baseline_share": round(st["ok_by_tier"].get("baseline", 0) / max(st["pages"], 1), 3)})
        for r in rows:
            tier = r["tier"]
            if "lang" in r:
                langs[tier][r["lang"]] += 1
                if vi_share is not None and vi_share >= VI_BOOK_MIN_SHARE and r["lang"] != "vi":
                    odd_by[tier][r["lang"]] += 1
                    odd_pages.append({"book": name, "page": r["i"] + 1, "tier": tier, "lang": r["lang"], "lang_conf": r["lang_conf"],
                                      "script": r["script"], "letters": r["letters"], "book_vi_share": round(vi_share, 3)})
                    _reservoir(samples, seen, f"odd_{tier}", (path, r["i"], f"lang {r['lang']} ({r['lang_conf']}), sách {vi_share:.0%} tiếng Việt"),
                               n_suspect_samples, rng)
            if tier in ("(flagged)", "(blank)"):
                continue
            real = None
            if vocab is not None:
                unk, n_words = unknown_rate(r["text"], vocab)
                if n_words >= SYL_MIN_WORDS:
                    real = 1 - unk / n_words
                    syl[tier].append(real)
            if tier == "baseline":
                baseline_pages.append({"book": name, "page": r["i"] + 1, "real_syllable_rate": None if real is None else round(real, 4),
                                       "diac_ratio": round(r["diac_ratio"], 3), "lang": r.get("lang", ""), "lang_conf": r.get("lang_conf", ""),
                                       "script": r["script"], "letters": r["letters"]})
                if real is not None:
                    band = sum(real >= e for e in SYL_SAMPLE_EDGES)
                    lo = ("0", *map(str, SYL_SAMPLE_EDGES))[band]
                    hi = (*map(str, SYL_SAMPLE_EDGES), "1")[band]
                    _reservoir(samples, seen, f"baseline_syllable_{lo}-{hi}", (path, r["i"], f"tỷ lệ âm tiết thật {real:.3f}"), per_tier, rng)
            _reservoir(samples, seen, f"tier_{tier}", (path, r["i"], ""), per_tier, rng)  # mỗi trang của engine có cùng xác suất được chọn
            if r["reasons"]:
                n_susp += 1
                suspects.append({"book": name, "page": r["i"] + 1, "tier": tier, "reasons": "|".join(r["reasons"]), "chars": r["chars"],
                                 "diac_ratio": round(r["diac_ratio"], 3), "non_latin_ratio": round(r["non_latin_ratio"], 3),
                                 "loop_cover": round(r["loop_cover"], 3), "mean_word_len": round(r["mean_word_len"], 2),
                                 "long_words": r["long_words"], "script": r["script"], "lang": r.get("lang", "")})
                _reservoir(samples, seen, "suspect", (path, r["i"], "|".join(r["reasons"])), n_suspect_samples, rng)
                for reason in ("glued_text", "long_word"):  # O-30 (4): mẫu riêng theo engine để chọn ngưỡng bằng mắt
                    if reason in r["reasons"]:
                        _reservoir(samples, seen, f"{reason}_{tier}", (path, r["i"], f"{reason}, từ trung bình {r['mean_word_len']:.1f} chữ, {r['long_words']} từ dài"),
                                   n_suspect_samples, rng)
        if empty_ok:
            problems.append(f"{name}: {empty_ok} trang status ok nhưng văn bản rỗng")
        n = max(st["pages"], 1)
        books.append({"book": name, "pages": st["pages"], "pages_per_tier": json.dumps(st["ok_by_tier"]), "rerun": st["rerun"],
                      "blank": st["blank"], "flagged": st["flagged"], "seconds": st["seconds"],
                      "pages_per_sec": round(st["pages"] / max(st["seconds"], 0.1), 2), "chars_per_page": round(tot["chars"] / n),
                      "diac_ratio": round(tot["diac"] / tot["letters"], 3) if tot["letters"] else 0.0,
                      "baseline_share": round(st["ok_by_tier"].get("baseline", 0) / n, 3),
                      "vi_share": None if vi_share is None else round(vi_share, 3)})
    return {"books": books, "tiers": tiers, "suspects": suspects, "problems": problems, "samples": samples, "suspect_total": n_susp,
            "scripts": scripts, "langs": langs, "foreign_books": foreign_books, "odd_pages": odd_pages, "odd_by": odd_by,
            "baseline_pages": baseline_pages, "syl": syl}


def _parse_logs(data_root: Path, out_dir: Path) -> dict:
    """
    Chép log từng GPU vào out_dir/logs và rút tốc độ (trang/giây của từng cuốn) cùng các dòng cảnh báo / lỗi.

    Returns:
        {gpu: {"books", "pages", "pages_per_sec_p50", "pages_per_sec_min", "pages_per_sec_max"}} và "messages": các dòng WARNING / ERROR.
    """
    result: dict = {"per_gpu": {}, "messages": []}
    logs = sorted((data_root / "logs").glob("stbook_ocr_vlm_gpu*.out"))
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    for log in logs:
        shutil.copy2(log, out_dir / "logs" / log.name)
        rates, pages = [], 0
        for line in log.read_text(errors="replace").splitlines():
            if (m := _LOG_LINE.search(line)):
                pages += int(m.group(1))
                rates.append(float(m.group(2)))
            elif re.search(r"WARNING|ERROR|Traceback|Lỗi", line) and len(result["messages"]) < LOG_MESSAGES_CAP:
                result["messages"].append(f"{log.name}: {line.strip()[:300]}")
        rates.sort()
        result["per_gpu"][log.stem] = {"books": len(rates), "pages": pages, "pages_per_sec_p50": rates[len(rates) // 2] if rates else None,
                                       "pages_per_sec_min": rates[0] if rates else None, "pages_per_sec_max": rates[-1] if rates else None}
    return result


def _write_samples(data_root: Path, out_dir: Path, scan: dict) -> int:
    """
    Ghi trang mẫu: ảnh .jpg (render từ PDF) và văn bản .txt cạnh nhau. Bỏ qua phần ảnh nếu thiếu pymupdf hoặc PDF.

    Returns:
        Số trang mẫu đã ghi.
    """
    try:
        import pymupdf
    except ImportError:
        pymupdf = None
        logger.warning("Thiếu pymupdf (uv sync --group ocr): chỉ ghi văn bản mẫu, không có ảnh")
    groups = scan["samples"]
    n = 0
    for group, items in groups.items():
        gdir = out_dir / "samples" / group.replace("(", "").replace(")", "")
        gdir.mkdir(parents=True, exist_ok=True)
        for path, i, note in items:
            done = read_json(path)
            stem = f"{path.parent.name}__{path.stem}__p{i + 1}"
            pi = done["page_info"][i]
            head = (f"# {path.parent.name}/{path.stem} trang {i + 1} | engine {pi.get('tier')} | lý do bộ kiểm {pi.get('reasons')}"
                    f"{' | ' + note if note else ''}\n\n")
            (gdir / f"{stem}.txt").write_text(head + done["pages"][i], encoding="utf-8")
            pdf = Path(done["pdf_path"])
            pdf = pdf if pdf.is_absolute() else data_root / pdf
            if pymupdf and pdf.exists():
                with pymupdf.open(pdf) as doc:
                    (gdir / f"{stem}.jpg").write_bytes(doc[i].get_pixmap(dpi=80).tobytes("jpg"))
            n += 1
    return n


def _load_lid(spec: str):
    """
    Nạp bộ nhận diện ngôn ngữ; không nạp được (thiếu fasttext hoặc mạng) thì dùng heuristic và cảnh báo rõ, kết quả ghi lại spec đã dùng.

    Returns:
        (bộ nhận diện, spec thực tế đã dùng).
    """
    try:
        return get_lid(spec), spec
    except Exception as e:  # noqa: BLE001 - thiếu thư viện / mạng đều phải rơi về heuristic, không làm hỏng cả bộ kiểm
        logger.warning("Không nạp được bộ nhận diện ngôn ngữ %s (%s: %s): dùng heuristic, số ngôn ngữ kém chính xác hơn", spec, type(e).__name__, e)
        return get_lid("heuristic"), "heuristic"


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Ghi danh sách dòng ra CSV (bỏ qua khi rỗng)."""
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def check(data_root: Path, per_tier: int = 20, n_suspect_samples: int = 20, seed: int = 0, lang_model: str = DEFAULT_LANG_MODEL,
          syllables: Path | None = None) -> Path:
    """
    Chạy toàn bộ phần kiểm và ghi kết quả vào <data_root>/processed/ocr_vlm_check/.

    Args:
        per_tier:          Số trang mẫu ngẫu nhiên cho mỗi engine (kèm ảnh để so bằng mắt) và cho mỗi dải tỷ lệ âm tiết của baseline.
        n_suspect_samples: Số trang đáng nghi (và trang lạ trong sách tiếng Việt, mỗi engine) lấy mẫu thêm.
        seed:              Hạt giống chọn mẫu (cùng dữ liệu thì cùng mẫu).
        lang_model:        Bộ nhận diện ngôn ngữ theo trang ("fasttext:lid.176", "fasttext:<file.bin>" hoặc "heuristic").
        syllables:         File tập âm tiết (mặc định <data_root>/processed/ocr_bakeoff/syllables.json); thiếu thì bỏ số đo tỷ lệ âm tiết thật.

    Returns:
        Thư mục kết quả.
    """
    out_dir = data_root / CHECK_DIR
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    lid, lid_spec = _load_lid(lang_model)
    syl_path = syllables or data_root / "processed/ocr_bakeoff/syllables.json"
    vocab = load_syllables(syl_path) if syl_path.exists() else None
    if vocab is None:
        logger.warning("Không có %s: bỏ số đo tỷ lệ âm tiết thật (chạy scripts/ocr_bakeoff/build_syllables.py)", syl_path)
    scan = _scan(data_root, per_tier, n_suspect_samples, random.Random(seed), lid, vocab)
    logs = _parse_logs(data_root, out_dir)
    n_samples = _write_samples(data_root, out_dir, scan)

    scan["foreign_books"].sort(key=lambda r: r["vi_share"])
    scan["baseline_pages"].sort(key=lambda r: (r["real_syllable_rate"] is None, r["real_syllable_rate"] or 0))
    for name, rows in (("books.csv", scan["books"]), ("suspect_pages.csv", scan["suspects"]), ("foreign_books.csv", scan["foreign_books"]),
                       ("odd_pages_in_vi_books.csv", scan["odd_pages"]), ("baseline_pages.csv", scan["baseline_pages"])):
        _write_csv(out_dir / name, rows)

    tier_summary = {}
    for t, c in scan["tiers"].items():
        tier_summary[t] = {"pages": c["pages"], "chars_per_page": round(c["chars"] / max(c["pages"], 1)),
                           "diac_ratio": round(c["diac"] / c["letters"], 3) if c["letters"] else None,
                           "suspect": {k[8:]: v for k, v in c.items() if k.startswith("suspect_")}}
    rates = sorted(b["pages_per_sec"] for b in scan["books"])
    odd_books = Counter()
    for r in scan["odd_pages"]:
        odd_books[(r["tier"], r["book"])] += 1
    shape = {}  # O-30 (4): hai chỉ số hình dạng chữ, số trang theo engine và cuốn nhiều trang nhất
    for reason in ("glued_text", "long_word"):
        hits = [r for r in scan["suspects"] if reason in r["reasons"].split("|")]
        shape[reason] = {"total": len(hits), "by_tier": dict(Counter(r["tier"] for r in hits)),
                         "top_books": [[t, b, n] for (t, b), n in Counter((r["tier"], r["book"]) for r in hits).most_common(40)]}
    shape["thresholds"] = {"glued_mean_word_len": GLUED_MEAN_WORD_LEN, "glued_min_letters": GLUED_MIN_LETTERS, "long_word_letters": LONG_WORD_LETTERS}
    summary = {
        "books": len(scan["books"]), "pages": sum(b["pages"] for b in scan["books"]),
        "problems_consistency": scan["problems"][:LOG_MESSAGES_CAP], "n_problems": len(scan["problems"]),
        "by_tier": tier_summary, "suspect_pages_total": scan["suspect_total"],
        "books_pages_per_sec": {"p10": rates[len(rates) // 10], "p50": rates[len(rates) // 2], "p90": rates[len(rates) * 9 // 10],
                                "min": rates[0], "max": rates[-1]} if rates else None,
        "books_with_baseline_share_over_5pct": sorted([(b["book"], b["baseline_share"]) for b in scan["books"] if b["baseline_share"] > 0.05],
                                                      key=lambda x: -x[1])[:30],
        "lang_model": lid_spec,
        "shape_metrics": shape,
        "scripts_by_tier": {t: dict(c.most_common()) for t, c in scan["scripts"].items()},
        "languages_by_tier": {t: dict(c.most_common(15)) for t, c in scan["langs"].items()},
        "foreign_books": {"n": len(scan["foreign_books"]), "vi_share_max": FOREIGN_BOOK_MAX_SHARE,
                          "top": [(r["book"], r["vi_share"], r["top_langs"]) for r in scan["foreign_books"][:30]]},
        "odd_pages_in_vi_books": {"vi_book_min_share": VI_BOOK_MIN_SHARE, "lang_min_letters": LANG_MIN_LETTERS,
                                  "total": len(scan["odd_pages"]),
                                  "by_tier": {t: {"total": sum(c.values()), "by_lang": dict(c.most_common(10))} for t, c in scan["odd_by"].items()},
                                  "top_books": [[t, b, n] for (t, b), n in odd_books.most_common(40)]},
        "syllables": None if vocab is None else {
            "file": str(syl_path), "min_words": SYL_MIN_WORDS, "vocab_size": len(vocab),
            "real_rate_by_tier": {t: _percentiles(v) for t, v in scan["syl"].items()},
            "baseline_histogram": _histogram(scan["syl"].get("baseline", []))},
        "logs": logs, "samples_written": n_samples,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir
