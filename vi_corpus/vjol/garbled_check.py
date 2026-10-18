"""
Đo chữ hỏng do font PDF trên các checkpoint VJOL đã có (J-10 (5), J-16): chỉ dùng CPU, không cần GPU hay server vLLM, để chốt ngưỡng
TRƯỚC khi tốn GPU chạy lại mẫu.

Bộ dò có bốn tín hiệu (vi_corpus.common.garbled): tone (từ nhiều dấu thanh), case (chữ hoa chen giữa từ), foreign (chữ cái Latin-1 lạ
ñ ö ï ø å), unknown (âm tiết lạ so với syllables.json; bỏ nếu thiếu file). Một trang bị nghi nếu có ít nhất một tín hiệu vượt ngưỡng.

Đọc `interim/vjol_text/*.json` (cả bản cũ chưa dò chữ hỏng lẫn bản mới) rồi ghi `processed/vjol_garbled_check/`:
    summary.json   : số trang / bài bị nghi ở ngưỡng hiện hành (tổng, theo từng tín hiệu, và chỉ riêng tín hiệu đó bắt), quét ngưỡng
                     từng tín hiệu, quét tone x ngưỡng cấp bài, phân bố tỷ lệ tone theo trang, phân bố theo năm, ước số phút OCR lại.
    docs.csv       : mỗi bài một dòng (năm, số trang chữ, số trang bị nghi, tỷ lệ, bài có bị nâng lên cấp bài không).
    pages.csv      : mọi trang bị nghi kèm tín hiệu, số đo và đoạn đầu của lớp chữ.
    examples.txt   : vài trang bị nghi nặng nhất (theo tone) và vài trang ngẫu nhiên.
    new_signals.txt: với mỗi tín hiệu bổ sung (case, foreign, unknown): trang CHỈ tín hiệu đó bắt (kiểm độ chính xác bằng mắt) và trang
                     sát dưới ngưỡng (kiểm độ sót).
    blind_spots.txt: trang KHÔNG bị nghi nhưng đáng xem (J-13): nhiều âm tiết lạ nhất, nhiều chữ hoa chen giữa nhất, và trang có lớp chữ
                     nhưng quá ít từ.
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
    DEFAULT_THRESHOLDS,
    GARBLED_DOC_SHARE,
    REASON_DOC,
    REASON_PAGE,
    SIGNALS,
    load_vocab,
    page_measures,
    plan_from_flags,
    signals_from_measures,
)
from vi_corpus.common.state import read_json, write_json_atomic
from vi_corpus.vjol.extract import SYLLABLES_REL, TEXT_DIR

logger = logging.getLogger("vi_corpus")

CHECK_DIR = "processed/vjol_garbled_check"
SWEEPS = {  # ngưỡng thử cho từng tín hiệu
    "tone": (0.03, 0.05, 0.08, 0.10, 0.15, 0.20),
    "case": (0.02, 0.05, 0.10, 0.20),
    "foreign": (0.003, 0.01, 0.02, 0.05),
    "unknown": (0.25, 0.30, 0.40, 0.50, 0.60),
}
SHARE_SWEEP = (0.10, 0.20, 0.30, 0.50)  # ngưỡng cấp bài thử
HIST_BINS = (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01)  # cột phân bố tỷ lệ tone theo trang
PAGES_PER_SEC_PER_GPU = 1.55  # O-22, để ước giờ OCR lại
GPUS = 4
SNIPPET = 160
N_EXAMPLES = 15
N_NEAR = 10


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


def read_docs(data_root: Path, vocab: frozenset[str] | None) -> list[dict]:
    """
    Đọc checkpoint VJOL không lỗi, không bị bỏ qua và đo bốn tín hiệu cho từng trang có lớp chữ (một lần).

    Returns:
        [{"sha256", "path", "year", "pages": [{"text", "m"} hoặc None cho trang không có lớp chữ]}], "m" là kết quả page_measures.
    """
    docs = []
    for ckpt in sorted((data_root / TEXT_DIR).glob("*.json")):
        done = read_json(ckpt)
        if not done or done.get("error") or done.get("skipped"):
            continue
        pages = []
        for p in done["pages"]:
            t = layer_text(p)
            pages.append({"text": t, "m": page_measures(t, vocab)} if t is not None and t.strip() else None)
        docs.append({"sha256": done["sha256"], "path": done["paths"][0], "year": done.get("year_dir"), "pages": pages})
    return docs


def plan_doc(doc: dict, th=DEFAULT_THRESHOLDS, doc_share: float = GARBLED_DOC_SHARE) -> list[str | None]:
    """Kế hoạch OCR lại (xem vi_corpus.common.garbled.plan_garbled) của một bài từ số đo đã tính."""
    has_text = [p is not None for p in doc["pages"]]
    suspect = [p is not None and bool(signals_from_measures(p["m"], th)) for p in doc["pages"]]
    return plan_from_flags(has_text, suspect, doc_share)


def value(m: dict, sig: str) -> float | None:
    """Giá trị tỷ lệ của tín hiệu `sig` trong số đo m (None nếu không đo được, vd thiếu vocab)."""
    return None if m[sig] is None else m[sig][0]


def sweep(docs: list[dict]) -> dict:
    """
    Quét ngưỡng.

    Returns:
        {"by_signal": {tín hiệu: [{"threshold", "pages_flagged", "pages_only_this_signal"}]}, "tone_x_doc_share": [{"tone", "doc_share",
        "pages_to_ocr", "docs_with_ocr", "docs_whole"}]}. Khi quét một tín hiệu, các tín hiệu khác để ngưỡng mặc định; "chỉ riêng" nghĩa
        là không tín hiệu khác nào ở ngưỡng mặc định bắt trang đó.
    """
    pages = [p for d in docs for p in d["pages"] if p is not None]
    defaults = [signals_from_measures(p["m"]) for p in pages]
    by_signal = {}
    for sig, values in SWEEPS.items():
        rows = []
        for t in values:
            th = DEFAULT_THRESHOLDS._replace(**{sig: t})
            hit = [sig in signals_from_measures(p["m"], th) for p in pages]
            rows.append({"threshold": t, "pages_flagged": sum(hit),
                         "pages_only_this_signal": sum(h and not [x for x in d if x != sig] for h, d in zip(hit, defaults))})
        by_signal[sig] = rows
    grid = []
    for t in SWEEPS["tone"]:
        for share in SHARE_SWEEP:
            th = DEFAULT_THRESHOLDS._replace(tone=t)
            n_pages = n_docs = whole = 0
            for d in docs:
                plan = plan_doc(d, th, share)
                n = sum(r is not None for r in plan)
                n_pages += n
                n_docs += n > 0
                whole += any(r == REASON_DOC for r in plan)
            grid.append({"tone": t, "doc_share": share, "pages_to_ocr": n_pages, "docs_with_ocr": n_docs, "docs_whole": whole})
    return {"by_signal": by_signal, "tone_x_doc_share": grid}


def fmt(row: dict) -> str:
    """Một dòng ví dụ: năm, đường dẫn, trang, số đo của bốn tín hiệu và đoạn đầu."""
    ms = " ".join(f"{s}={'-' if row['m'][s] is None else format(row['m'][s][0], '.3f')}" for s in SIGNALS)
    return f"[{row['year']}] {row['path']} trang {row['page']} {ms} ({row['m']['tone'][1]} từ): {row['snippet']}\n"


def check(data_root: Path, seed: int = 0, syllables: Path | None = None) -> Path:
    """
    Chạy toàn bộ phần đo trên checkpoint VJOL hiện có và ghi kết quả vào <data_root>/processed/vjol_garbled_check/.

    Args:
        data_root: Thư mục gốc dữ liệu.
        seed:      Hạt giống chọn trang ngẫu nhiên cho các file ví dụ.
        syllables: File tập âm tiết (mặc định <data_root>/processed/ocr_bakeoff/syllables.json); thiếu thì bỏ tín hiệu "unknown".

    Returns:
        Đường dẫn thư mục kết quả.
    """
    out_dir = data_root / CHECK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = load_vocab(syllables or data_root / SYLLABLES_REL)
    if vocab is None:
        logger.warning("Không thấy tập âm tiết: bỏ tín hiệu 'unknown' trong lần đo này")
    docs = read_docs(data_root, vocab)
    hist: Counter = Counter()
    years: dict = defaultdict(Counter)
    by_signal: Counter = Counter()
    only_signal: Counter = Counter()
    suspect_rows: list[dict] = []
    clear_rows: list[dict] = []  # trang có lớp chữ và không bị nghi
    short_rows: list[dict] = []
    doc_rows: list[dict] = []
    n_text_pages = 0
    for d in docs:
        plan = plan_doc(d)
        text_pages = [(i, p) for i, p in enumerate(d["pages"]) if p is not None]
        n_text_pages += len(text_pages)
        n_suspect = sum(r == REASON_PAGE for r in plan)
        y = years[d["year"]]
        y["docs"] += 1
        y["text_pages"] += len(text_pages)
        y["suspect_pages"] += n_suspect
        y["pages_to_ocr"] += sum(r is not None for r in plan)
        y["docs_whole"] += any(r == REASON_DOC for r in plan)
        y["docs_with_suspect"] += n_suspect > 0
        doc_rows.append({"sha256": d["sha256"], "path": d["path"], "year": d["year"], "text_pages": len(text_pages),
                         "suspect_pages": n_suspect, "suspect_share": round(n_suspect / len(text_pages), 3) if text_pages else 0.0,
                         "whole_doc": any(r == REASON_DOC for r in plan)})
        for i, p in text_pages:
            sigs = signals_from_measures(p["m"])
            row = {"sha256": d["sha256"], "path": d["path"], "year": d["year"], "page": i + 1, "m": p["m"], "signals": sigs,
                   "snippet": " ".join(p["text"].split())[:SNIPPET]}
            if p["m"]["tone"][1] >= DEFAULT_THRESHOLDS.min_words:
                hist[hist_bin(p["m"]["tone"][0])] += 1
            for s in sigs:
                by_signal[s] += 1
            if len(sigs) == 1:
                only_signal[sigs[0]] += 1
            if sigs:
                suspect_rows.append(row)
            else:
                clear_rows.append(row)
                if p["m"]["tone"][1] < DEFAULT_THRESHOLDS.min_words:
                    short_rows.append(row)
    n_suspect_pages = len(suspect_rows)
    pages_to_ocr = sum(c["pages_to_ocr"] for c in years.values())
    summary = {
        "thresholds": DEFAULT_THRESHOLDS._asdict() | {"doc_share": GARBLED_DOC_SHARE}, "vocab": vocab is not None,
        "docs": len(docs), "text_layer_pages": n_text_pages, "suspect_pages": n_suspect_pages,
        "suspect_page_rate": round(n_suspect_pages / n_text_pages, 4) if n_text_pages else None,
        "suspect_pages_by_signal": dict(by_signal), "suspect_pages_only_one_signal": dict(only_signal),
        "docs_with_suspect_pages": sum(r["suspect_pages"] > 0 for r in doc_rows),
        "docs_whole_upgraded": sum(r["whole_doc"] for r in doc_rows),
        "docs_to_ocr": sum(r["suspect_pages"] > 0 or r["whole_doc"] for r in doc_rows),
        "pages_to_ocr_again": pages_to_ocr,
        "ocr_minutes_estimate": round(pages_to_ocr / PAGES_PER_SEC_PER_GPU / GPUS / 60, 2),
        "ocr_minutes_estimate_note": f"Ước chỉ cho các trang lỗi font (chưa tính trang scan): {PAGES_PER_SEC_PER_GPU} trang/giây/GPU (O-22) x {GPUS} GPU, chưa đo với trang vẽ lại",
        "tone_histogram_pages": dict(sorted(hist.items())),
        "short_text_pages": len(short_rows),
        "by_year": {str(y): dict(c) for y, c in sorted(years.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "sweep": sweep(docs),
    }
    write_json_atomic(out_dir / "summary.json", summary)
    for name, rows in (("docs.csv", doc_rows), ("pages.csv", [
            {"sha256": r["sha256"], "path": r["path"], "year": r["year"], "page": r["page"], "signals": "+".join(r["signals"]),
             **{s: "" if r["m"][s] is None else round(r["m"][s][0], 4) for s in SIGNALS}, "words": r["m"]["tone"][1],
             "snippet": r["snippet"]} for r in suspect_rows])):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    rng = random.Random(seed)

    def sample(rows: list[dict], n: int) -> list[dict]:
        """n dòng ngẫu nhiên (hoặc hết nếu ít hơn)."""
        return rng.sample(rows, min(n, len(rows)))

    tone_rows = [r for r in suspect_rows if "tone" in r["signals"]]
    worst = sorted(tone_rows, key=lambda r: -r["m"]["tone"][0])[:N_EXAMPLES]
    with open(out_dir / "examples.txt", "w", encoding="utf-8") as f:
        for title, rows in (("Nặng nhất (tone)", worst), ("Ngẫu nhiên (mọi trang bị nghi)", sample(suspect_rows, N_EXAMPLES))):
            f.write(f"=== {title}\n")
            f.writelines(fmt(r) for r in rows)
    with open(out_dir / "new_signals.txt", "w", encoding="utf-8") as f:
        for sig in SIGNALS[1:]:
            if sig == "unknown" and vocab is None:
                f.write("=== unknown: thiếu syllables.json\n")
                continue
            th = getattr(DEFAULT_THRESHOLDS, sig)
            only = [r for r in suspect_rows if r["signals"] == [sig]]
            f.write(f"=== {sig}: {by_signal[sig]} trang bị bắt (ngưỡng {th}), {len(only)} trang CHỈ tín hiệu này bắt; lấy ngẫu nhiên\n")
            f.writelines(fmt(r) for r in sample(only, N_EXAMPLES))
            near = [r for r in clear_rows if (value(r["m"], sig) or 0.0) >= th / 2 and (value(r["m"], sig) or 0.0) < th]
            f.write(f"=== {sig}: sát dưới ngưỡng ({th / 2:.4g}-{th:.4g}) mà chưa bị nghi: {len(near)} trang; lấy ngẫu nhiên\n")
            f.writelines(fmt(r) for r in sample(near, N_NEAR))
    with open(out_dir / "blind_spots.txt", "w", encoding="utf-8") as f:
        for sig in ("unknown", "case"):
            rows = [r for r in clear_rows if value(r["m"], sig) is not None]
            f.write(f"=== Nhiều nhất theo {sig} trong số trang KHÔNG bị nghi\n")
            f.writelines(fmt(r) for r in sorted(rows, key=lambda r: -value(r["m"], sig))[:N_EXAMPLES])
        f.write(f"=== Có lớp chữ nhưng dưới {DEFAULT_THRESHOLDS.min_words} từ: {len(short_rows)} trang, lấy ngẫu nhiên\n")
        f.writelines(fmt(r) for r in sample(short_rows, N_EXAMPLES))
    logger.info("Đo chữ hỏng: %s", json.dumps({k: summary[k] for k in ("docs", "text_layer_pages", "suspect_pages", "docs_to_ocr",
                                                                         "pages_to_ocr_again")}, ensure_ascii=False))
    return out_dir
