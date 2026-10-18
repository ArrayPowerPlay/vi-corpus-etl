"""
Đo chữ hỏng do font PDF trên các checkpoint VJOL đã có (J-10 (5)): chỉ dùng CPU, không cần GPU hay server vLLM, để chốt ngưỡng
TRƯỚC khi tốn GPU chạy lại mẫu.

Đọc `interim/vjol_text/*.json` (cả bản cũ chưa dò chữ hỏng lẫn bản mới) rồi ghi `processed/vjol_garbled_check/`:
    summary.json   : số trang / bài bị nghi ở ngưỡng hiện hành, quét ngưỡng (tỷ lệ từ nhiều dấu thanh x ngưỡng cấp bài), phân bố
                     tỷ lệ theo trang, phân bố theo năm, ước số trang phải OCR lại và số phút GPU, và tín hiệu thứ hai (tỷ lệ "từ"
                     một chữ cái, kiểu "đ u") trên trang tiếng Việt, tách theo trang bị nghi / không bị nghi (chỉ đo, chưa quyết).
    docs.csv       : mỗi bài một dòng (năm, số trang chữ, số trang bị nghi, tỷ lệ, bài có bị nâng lên cấp bài không).
    pages.csv      : mọi trang bị nghi kèm tỷ lệ, lý do và đoạn đầu của lớp chữ.
    examples.txt   : vài trang bị nghi nặng nhất và vài trang ngẫu nhiên, để mắt người xem có đúng là chữ hỏng không.
    blind_spots.txt: các trang KHÔNG bị nghi nhưng đáng xem (J-13): sát dưới ngưỡng, nhiều cụm chữ một ký tự nhất, nhiều âm tiết lạ nhất
                     (cần syllables.json của O-27), và trang có lớp chữ nhưng quá ít từ; để biết bộ dò có bỏ sót kiểu chữ hỏng nào không.
Lớp chữ của một trang là text của trang "text" / "tcvn3", hoặc "garbled_text" nếu trang đã được OCR lại vì lỗi font (nên chạy được cả
trước lẫn sau khi chạy lại mẫu). Trang scan, trang trắng không có lớp chữ.
"""

import csv
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from vi_corpus.common.garbled import (
    GARBLED_DOC_SHARE,
    GARBLED_MIN_WORDS,
    GARBLED_RATIO,
    REASON_DOC,
    REASON_PAGE,
    garbled_ratio,
    plan_garbled,
    single_letter_ratio,
)
from vi_corpus.common.state import read_json, write_json_atomic
from vi_corpus.ocr_bakeoff.syllables import load_syllables, unknown_rate
from vi_corpus.pipeline.language import vi_diacritic_ratio
from vi_corpus.vjol.extract import TEXT_DIR

logger = logging.getLogger("vi_corpus")

CHECK_DIR = "processed/vjol_garbled_check"
RATIO_SWEEP = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20)  # ngưỡng tỷ lệ từ nhiều dấu thanh thử
SHARE_SWEEP = (0.10, 0.20, 0.30, 0.50)  # ngưỡng cấp bài thử
HIST_BINS = (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01)  # cột phân bố tỷ lệ theo trang
VI_MIN_DIACRITIC = 0.05  # trang có tỷ lệ chữ có dấu từ mức này coi là tiếng Việt (để đo tín hiệu thứ hai)
PAGES_PER_SEC_PER_GPU = 1.55  # O-22, để ước giờ OCR lại
GPUS = 4
SNIPPET = 160
N_EXAMPLES = 15
SYL_MIN_WORDS = 30  # trang ít từ hơn mức này không tính tỷ lệ âm tiết lạ
SYLLABLES_REL = "processed/ocr_bakeoff/syllables.json"
NEAR_BAND = (0.02, GARBLED_RATIO)  # "sát dưới ngưỡng": tỷ lệ trong khoảng này mà chưa bị nghi


def layer_text(page: dict) -> str | None:
    """Lớp chữ của một mục trang trong checkpoint (xem docstring module); None nếu trang không có lớp chữ để dò."""
    if page.get("garbled_text") is not None:
        return page["garbled_text"]
    return page["text"] if page["method"] in ("text", "tcvn3") else None


def hist_bin(ratio: float) -> str:
    """Nhãn cột của tỷ lệ trong HIST_BINS, dạng "0.04-0.06"."""
    for lo, hi in zip(HIST_BINS, HIST_BINS[1:]):
        if lo <= ratio < hi:
            return f"{lo:.2f}-{min(hi, 1.0):.2f}"
    return f"{HIST_BINS[-2]:.2f}-1.00"


def quantiles(values: list[float]) -> dict:
    """Số mẫu, trung bình và các phân vị 50 / 90 / 99 của một danh sách (rỗng thì chỉ có n = 0)."""
    if not values:
        return {"n": 0}
    v = sorted(values)
    pick = lambda q: round(v[min(int(q * len(v)), len(v) - 1)], 4)  # noqa: E731
    return {"n": len(v), "mean": round(sum(v) / len(v), 4), "p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99)}


def read_docs(data_root: Path) -> list[dict]:
    """
    Đọc checkpoint VJOL không lỗi, không bị bỏ qua.

    Returns:
        [{"sha256", "path", "year", "layers": [lớp chữ từng trang hoặc None]}].
    """
    docs = []
    for ckpt in sorted((data_root / TEXT_DIR).glob("*.json")):
        done = read_json(ckpt)
        if not done or done.get("error") or done.get("skipped"):
            continue
        docs.append({"sha256": done["sha256"], "path": done["paths"][0], "year": done.get("year_dir"),
                     "layers": [layer_text(p) for p in done["pages"]]})
    return docs


def sweep(docs: list[dict]) -> list[dict]:
    """
    Quét ngưỡng: với mỗi cặp (ngưỡng tỷ lệ, ngưỡng cấp bài), số trang phải OCR lại và số bài bị nâng lên cấp bài.

    Returns:
        [{"ratio", "doc_share", "pages_to_ocr", "docs_with_ocr", "docs_whole"}], pages_to_ocr gồm trang bị nghi lẫn trang kéo theo.
    """
    out = []
    for ratio in RATIO_SWEEP:
        for share in SHARE_SWEEP:
            pages = docs_hit = whole = 0
            for d in docs:
                plan = plan_garbled(d["layers"], min_ratio=ratio, doc_share=share)
                n = sum(r is not None for r in plan)
                pages += n
                docs_hit += n > 0
                whole += any(r == REASON_DOC for r in plan)
            out.append({"ratio": ratio, "doc_share": share, "pages_to_ocr": pages, "docs_with_ocr": docs_hit, "docs_whole": whole})
    return out


def check(data_root: Path, seed: int = 0, syllables: Path | None = None) -> Path:
    """
    Chạy toàn bộ phần đo trên checkpoint VJOL hiện có và ghi kết quả vào <data_root>/processed/vjol_garbled_check/.

    Args:
        data_root: Thư mục gốc dữ liệu.
        seed:      Hạt giống chọn trang ngẫu nhiên cho examples.txt.
        syllables: File tập âm tiết (mặc định <data_root>/processed/ocr_bakeoff/syllables.json); thiếu thì bỏ số đo âm tiết lạ.

    Returns:
        Đường dẫn thư mục kết quả.
    """
    out_dir = data_root / CHECK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = read_docs(data_root)
    hist: Counter = Counter()
    years: dict = defaultdict(lambda: Counter())
    pages_rows: list[dict] = []
    doc_rows: list[dict] = []
    single: dict[str, list[float]] = {"suspect": [], "normal": []}
    n_text_pages = 0
    syl_path = syllables or data_root / SYLLABLES_REL
    vocab = load_syllables(syl_path) if syl_path.exists() else None
    normal: list[dict] = []  # trang không bị nghi, cho blind_spots.txt
    short: list[dict] = []
    unknown: list[float] = []
    for d in docs:
        plan = plan_garbled(d["layers"])
        text_pages = [(i, t) for i, t in enumerate(d["layers"]) if t is not None and t.strip()]
        n_text_pages += len(text_pages)
        n_suspect = sum(r == REASON_PAGE for r in plan)
        years[d["year"]]["docs"] += 1
        years[d["year"]]["text_pages"] += len(text_pages)
        years[d["year"]]["suspect_pages"] += n_suspect
        years[d["year"]]["pages_to_ocr"] += sum(r is not None for r in plan)
        years[d["year"]]["docs_whole"] += any(r == REASON_DOC for r in plan)
        years[d["year"]]["docs_with_suspect"] += n_suspect > 0
        doc_rows.append({"sha256": d["sha256"], "path": d["path"], "year": d["year"], "text_pages": len(text_pages),
                         "suspect_pages": n_suspect,
                         "suspect_share": round(n_suspect / len(text_pages), 3) if text_pages else 0.0,
                         "whole_doc": any(r == REASON_DOC for r in plan)})
        for i, t in text_pages:
            ratio, n_words = garbled_ratio(t)
            if n_words >= GARBLED_MIN_WORDS:
                hist[hist_bin(ratio)] += 1
            is_vi = vi_diacritic_ratio(t) >= VI_MIN_DIACRITIC
            if is_vi:
                single["suspect" if plan[i] == REASON_PAGE else "normal"].append(single_letter_ratio(t))
            if plan[i] is None:
                row = {"year": d["year"], "path": d["path"], "page": i + 1, "ratio": ratio, "words": n_words,
                       "single": single_letter_ratio(t) if is_vi else 0.0, "unknown": None,
                       "snippet": " ".join(t.split())[:SNIPPET]}
                if vocab is not None and is_vi:
                    unk, n_w = unknown_rate(t, vocab)
                    if n_w >= SYL_MIN_WORDS:
                        row["unknown"] = unk / n_w
                        unknown.append(row["unknown"])
                (normal if n_words >= GARBLED_MIN_WORDS else short).append(row)
            if plan[i] == REASON_PAGE:
                pages_rows.append({"sha256": d["sha256"], "path": d["path"], "year": d["year"], "page": i + 1,
                                   "ratio": round(ratio, 4), "words": n_words, "reason": plan[i],
                                   "snippet": " ".join(t.split())[:SNIPPET]})
    n_suspect_pages = len(pages_rows)
    docs_with_ocr = sum(r["suspect_pages"] > 0 or r["whole_doc"] for r in doc_rows)
    pages_to_ocr = sum(c["pages_to_ocr"] for c in years.values())
    summary = {
        "thresholds": {"ratio": GARBLED_RATIO, "min_words": GARBLED_MIN_WORDS, "doc_share": GARBLED_DOC_SHARE},
        "docs": len(docs), "text_layer_pages": n_text_pages, "suspect_pages": n_suspect_pages,
        "suspect_page_rate": round(n_suspect_pages / n_text_pages, 4) if n_text_pages else None,
        "docs_with_suspect_pages": sum(r["suspect_pages"] > 0 for r in doc_rows),
        "docs_whole_upgraded": sum(r["whole_doc"] for r in doc_rows), "docs_to_ocr": docs_with_ocr,
        "pages_to_ocr_again": pages_to_ocr,
        "ocr_minutes_estimate": round(pages_to_ocr / PAGES_PER_SEC_PER_GPU / GPUS / 60, 2),
        "ocr_minutes_estimate_note": f"Ước chỉ cho các trang lỗi font (chưa tính trang scan): {PAGES_PER_SEC_PER_GPU} trang/giây/GPU (O-22) x {GPUS} GPU, chưa đo với trang vẽ lại",
        "ratio_histogram_pages": dict(sorted(hist.items())),
        "by_year": {str(y): dict(c) for y, c in sorted(years.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "single_letter_ratio_vi_pages": {k: quantiles(v) for k, v in single.items()},
        "unknown_syllable_rate_vi_pages_not_suspect": quantiles(unknown) if vocab is not None else "thiếu syllables.json",
        "short_text_pages": len(short),
        "sweep": sweep(docs),
    }
    write_json_atomic(out_dir / "summary.json", summary)
    for name, rows in (("docs.csv", doc_rows), ("pages.csv", pages_rows)):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    rng = random.Random(seed)
    worst = sorted(pages_rows, key=lambda r: -r["ratio"])[:N_EXAMPLES]
    worst_ids = {id(r) for r in worst}
    rest = [r for r in pages_rows if id(r) not in worst_ids]
    picked = rng.sample(rest, min(N_EXAMPLES, len(rest)))
    with open(out_dir / "examples.txt", "w", encoding="utf-8") as f:
        for title, rows in (("Nặng nhất", worst), ("Ngẫu nhiên", picked)):
            f.write(f"=== {title}\n")
            for r in rows:
                f.write(f"[{r['year']}] {r['path']} trang {r['page']} tỷ lệ {r['ratio']:.3f}: {r['snippet']}\n")
    with open(out_dir / "blind_spots.txt", "w", encoding="utf-8") as f:
        def dump(title: str, rows: list[dict]) -> None:
            """Ghi một nhóm trang: năm, đường dẫn, trang, các số đo và đoạn đầu."""
            f.write(f"=== {title}\n")
            for r in rows:
                unk = "-" if r["unknown"] is None else f"{r['unknown']:.2f}"
                f.write(f"[{r['year']}] {r['path']} trang {r['page']} garbled {r['ratio']:.3f} một-ký-tự {r['single']:.2f} "
                        f"âm-tiết-lạ {unk} ({r['words']} từ): {r['snippet']}\n")

        near = [r for r in normal if NEAR_BAND[0] <= r["ratio"] < NEAR_BAND[1]]
        dump(f"Sát dưới ngưỡng ({NEAR_BAND[0]}-{NEAR_BAND[1]}): {len(near)} trang, lấy ngẫu nhiên", rng.sample(near, min(N_EXAMPLES, len(near))))
        dump("Nhiều cụm chữ một ký tự nhất (trang tiếng Việt, không bị nghi)", sorted(normal, key=lambda r: -r["single"])[:N_EXAMPLES])
        if vocab is not None:
            dump("Nhiều âm tiết lạ nhất (trang tiếng Việt, không bị nghi)",
                 sorted((r for r in normal if r["unknown"] is not None), key=lambda r: -r["unknown"])[:N_EXAMPLES])
        dump(f"Có lớp chữ nhưng dưới {GARBLED_MIN_WORDS} từ: {len(short)} trang, lấy ngẫu nhiên", rng.sample(short, min(N_EXAMPLES, len(short))))
    logger.info("Đo chữ hỏng: %s", json.dumps({k: summary[k] for k in ("docs", "text_layer_pages", "suspect_pages", "docs_to_ocr",
                                                                         "pages_to_ocr_again")}, ensure_ascii=False))
    return out_dir
