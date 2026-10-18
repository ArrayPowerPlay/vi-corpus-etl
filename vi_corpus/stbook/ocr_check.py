"""
Kiểm kết quả OCR stbook bằng chuỗi VLM sau khi chạy xong (chỉ dùng CPU, không cần GPU hay server vLLM).

Đọc mọi file `interim/stbook_ocr/<slug>/<product_id>.json` do chuỗi VLM ghi và log của lượt chạy, rồi ghi một thư mục nhỏ
`processed/ocr_vlm_check/` để tải về xem:
    summary.json        : số liệu tổng (tính nhất quán, chữ/trang theo engine, tỷ lệ dấu tiếng Việt, tốc độ, cảnh báo trong log).
    books.csv           : mỗi cuốn một dòng (số trang, số trang theo engine, chạy lại, giây, trang/giây, chữ/trang, dấu).
    suspect_pages.csv   : trang đáng nghi (ít dấu tiếng Việt, lẫn chữ nước ngoài, lặp, ảnh nhúng, quá dài) kèm mã lý do.
    samples/<engine>/   : trang lấy ngẫu nhiên mỗi engine và vài trang đáng nghi, mỗi trang gồm ảnh .jpg và văn bản .txt để so bằng mắt.
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
from vi_corpus.stbook.ocr_books import OCR_DIR
from vi_corpus.stbook.ocr_vlm import ENGINE_TAG

logger = logging.getLogger("vi_corpus")

CHECK_DIR = "processed/ocr_vlm_check"
_LOWER_DIAC = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
_DIAC = re.compile("[" + _LOWER_DIAC + _LOWER_DIAC.upper() + "]")
_LETTER = re.compile(r"[^\W\d_]")
_NON_LATIN = re.compile("[Ѐ-ӿ؀-ۿ฀-๿぀-ヿ㐀-鿿가-힯]")
_IMAGE_MARK = ("data:image", "base64,")
_LOG_LINE = re.compile(r"\] (\d+) trang, ([\d.]+) trang/s")

LOW_DIAC_MAX = 0.10  # tiếng Việt thường có ~25-30% chữ cái mang dấu; dưới mức này thường là mất dấu hoặc trang không phải tiếng Việt
LOW_DIAC_MIN_LETTERS = 300  # trang quá ít chữ thì tỷ lệ dấu không đáng tin
NON_LATIN_MAX = 0.02  # chữ Hán / Kirin / Thái ... quá 2% số chữ cái là dấu hiệu mô hình bịa chữ
LOOP_COVER_MAX = 0.20  # dãy lặp phủ quá 20% trang
SUSPECT_CSV_CAP = 5000  # tối đa số dòng ghi ra suspect_pages.csv
LOG_MESSAGES_CAP = 200  # tối đa số dòng cảnh báo / lỗi chép vào summary.json


def page_metrics(text: str) -> dict:
    """
    Số đo rẻ của một trang để phát hiện trang đáng nghi.

    Returns:
        {"chars", "letters", "diac_ratio" (chữ cái mang dấu / chữ cái), "non_latin_ratio", "loop_cover" (tỷ lệ phủ của dãy lặp),
        "reasons": [mã lý do đáng nghi]}.
    """
    letters = len(_LETTER.findall(text))
    diac = len(_DIAC.findall(text))
    non_latin = len(_NON_LATIN.findall(text))
    cover, _ = repeat_stats(text)
    m = {"chars": len(text), "letters": letters, "diac_ratio": diac / letters if letters else 0.0,
         "non_latin_ratio": non_latin / letters if letters else 0.0, "loop_cover": cover}
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
    m["reasons"] = reasons
    return m


def _scan(data_root: Path, per_tier: int, n_suspect_samples: int, rng: random.Random) -> dict:
    """
    Duyệt mọi file OCR của chuỗi VLM, gom số đo theo cuốn / theo engine và chọn trang mẫu (reservoir sampling).

    Returns:
        {"books": [dòng books.csv], "tiers": {engine: Counter}, "suspects": [dòng suspect_pages.csv], "problems": [chuỗi],
        "samples": {engine: [(đường dẫn json, chỉ số trang)]}, "suspect_samples": [(đường dẫn json, chỉ số trang)]}.
    """
    books, suspects, problems = [], [], []
    tiers: dict[str, Counter] = defaultdict(Counter)
    samples: dict[str, list] = defaultdict(list)
    seen: Counter = Counter()
    suspect_samples: list = []
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
        for i, (text, pi) in enumerate(zip(pages, info)):
            tier = pi.get("tier", "?") if pi.get("status") == "ok" else f"({pi.get('status')})"
            if pi.get("status") == "ok" and not text.strip():
                empty_ok += 1
            m = page_metrics(text)
            for k in ("chars", "letters"):
                tot[k] += m[k]
                tiers[tier][k] += m[k]
            tot["diac"] += m["diac_ratio"] * m["letters"]
            tiers[tier]["diac"] += m["diac_ratio"] * m["letters"]
            tiers[tier]["pages"] += 1
            for r in m["reasons"]:
                tiers[tier]["suspect_" + r] += 1
            if tier in ("(flagged)", "(blank)"):
                continue
            seen[tier] += 1  # reservoir sampling: mỗi trang của engine có cùng xác suất được chọn
            if len(samples[tier]) < per_tier:
                samples[tier].append((path, i))
            elif (j := rng.randrange(seen[tier])) < per_tier:
                samples[tier][j] = (path, i)
            if m["reasons"]:
                n_susp += 1
                if len(suspects) < SUSPECT_CSV_CAP:
                    suspects.append({"book": name, "page": i + 1, "tier": tier, "reasons": "|".join(m["reasons"]), "chars": m["chars"],
                                     "diac_ratio": round(m["diac_ratio"], 3), "non_latin_ratio": round(m["non_latin_ratio"], 3),
                                     "loop_cover": round(m["loop_cover"], 3)})
                if len(suspect_samples) < n_suspect_samples:
                    suspect_samples.append((path, i))
                elif (j := rng.randrange(n_susp)) < n_suspect_samples:
                    suspect_samples[j] = (path, i)
        if empty_ok:
            problems.append(f"{name}: {empty_ok} trang status ok nhưng văn bản rỗng")
        n = max(st["pages"], 1)
        books.append({"book": name, "pages": st["pages"], "pages_per_tier": json.dumps(st["ok_by_tier"]), "rerun": st["rerun"],
                      "blank": st["blank"], "flagged": st["flagged"], "seconds": st["seconds"],
                      "pages_per_sec": round(st["pages"] / max(st["seconds"], 0.1), 2), "chars_per_page": round(tot["chars"] / n),
                      "diac_ratio": round(tot["diac"] / tot["letters"], 3) if tot["letters"] else 0.0,
                      "baseline_share": round(st["ok_by_tier"].get("baseline", 0) / n, 3)})
    return {"books": books, "tiers": tiers, "suspects": suspects, "problems": problems, "samples": samples,
            "suspect_samples": suspect_samples, "suspect_total": n_susp}


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
    groups = {**{f"tier_{t}": v for t, v in scan["samples"].items()}, "suspect": scan["suspect_samples"]}
    n = 0
    for group, items in groups.items():
        gdir = out_dir / "samples" / group.replace("(", "").replace(")", "")
        gdir.mkdir(parents=True, exist_ok=True)
        for path, i in items:
            done = read_json(path)
            stem = f"{path.parent.name}__{path.stem}__p{i + 1}"
            pi = done["page_info"][i]
            head = f"# {path.parent.name}/{path.stem} trang {i + 1} | engine {pi.get('tier')} | lý do bộ kiểm {pi.get('reasons')}\n\n"
            (gdir / f"{stem}.txt").write_text(head + done["pages"][i], encoding="utf-8")
            pdf = Path(done["pdf_path"])
            pdf = pdf if pdf.is_absolute() else data_root / pdf
            if pymupdf and pdf.exists():
                with pymupdf.open(pdf) as doc:
                    (gdir / f"{stem}.jpg").write_bytes(doc[i].get_pixmap(dpi=80).tobytes("jpg"))
            n += 1
    return n


def check(data_root: Path, per_tier: int = 20, n_suspect_samples: int = 20, seed: int = 0) -> Path:
    """
    Chạy toàn bộ phần kiểm và ghi kết quả vào <data_root>/processed/ocr_vlm_check/.

    Args:
        per_tier:          Số trang mẫu ngẫu nhiên cho mỗi engine (kèm ảnh để so bằng mắt).
        n_suspect_samples: Số trang đáng nghi lấy mẫu thêm.
        seed:              Hạt giống chọn mẫu (cùng dữ liệu thì cùng mẫu).

    Returns:
        Thư mục kết quả.
    """
    out_dir = data_root / CHECK_DIR
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    scan = _scan(data_root, per_tier, n_suspect_samples, random.Random(seed))
    logs = _parse_logs(data_root, out_dir)
    n_samples = _write_samples(data_root, out_dir, scan)

    for name, rows in (("books.csv", scan["books"]), ("suspect_pages.csv", scan["suspects"])):
        if rows:
            with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    tier_summary = {}
    for t, c in scan["tiers"].items():
        tier_summary[t] = {"pages": c["pages"], "chars_per_page": round(c["chars"] / max(c["pages"], 1)),
                           "diac_ratio": round(c["diac"] / c["letters"], 3) if c["letters"] else None,
                           "suspect": {k[8:]: v for k, v in c.items() if k.startswith("suspect_")}}
    rates = sorted(b["pages_per_sec"] for b in scan["books"])
    summary = {
        "books": len(scan["books"]), "pages": sum(b["pages"] for b in scan["books"]),
        "problems_consistency": scan["problems"][:LOG_MESSAGES_CAP], "n_problems": len(scan["problems"]),
        "by_tier": tier_summary, "suspect_pages_total": scan["suspect_total"],
        "books_pages_per_sec": {"p10": rates[len(rates) // 10], "p50": rates[len(rates) // 2], "p90": rates[len(rates) * 9 // 10],
                                "min": rates[0], "max": rates[-1]} if rates else None,
        "books_with_baseline_share_over_5pct": sorted([(b["book"], b["baseline_share"]) for b in scan["books"] if b["baseline_share"] > 0.05],
                                                      key=lambda x: -x[1])[:30],
        "logs": logs, "samples_written": n_samples,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir
