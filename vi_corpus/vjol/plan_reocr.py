"""
Đo CPU để chốt ngưỡng OCR lại cả bài cho VJOL (J-42 mục (2), P1): bao nhiêu bài / trang có lớp chữ lệch dấu, rơi chữ, dính liền, rác, gạch nối thay ư; nếu chuyển cả bài đó sang OCR VLM thì tốn bao nhiêu
trang, bao nhiêu giờ GPU (cho mẫu và nhân ra cả kho). Chỉ đọc checkpoint `interim/vjol_text/` (không cần GPU / vLLM), mẫu `vi_corpus.vjol.garbled_check`.

Mỗi bài: chỉ xét trang còn lớp chữ gốc (method "text" / "tcvn3"; trang đã OCR vì lỗi font hay scan không còn gì để dò). Mỗi trang đo bằng vi_corpus.vjol.reocr.page_defects; bài đi OCR lại khi số trang trúng
tín hiệu >= max(min_pages, share x số trang có chữ) (reocr.pages_needed), cùng quy tắc với extract_pdf nên con số ở đây là số thật của lần trích lại.

Ghi <data-root>/processed/vjol_reocr_plan/:
    summary.json   : số bài / trang trúng từng tín hiệu; với ngưỡng mặc định: số bài, số trang OCR lại, giờ (1,55 trang/giây/GPU, 4 GPU, O-22) cho mẫu và cho cả kho (nhân theo số PDF); quét share x bộ tín hiệu;
                     bảng theo (producer, creator) và theo tạp chí; bài chỉ có gạch nối thay ư (OCR không sửa được, J-32) đếm riêng; cảnh báo nếu trang OCR lại > REOCR_WARN_SHARE số trang của mẫu
    reocr_plan.csv : mỗi bài có ít nhất một tín hiệu một dòng (bài, tạp chí, producer, creator, số trang, số trang trúng từng tín hiệu, có OCR lại không, lý do, số trang sẽ OCR)
(producer, creator) lấy lại từ PDF gốc (checkpoint không lưu), tạp chí từ metadata; thiếu thì để trống.
"""

import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

from vi_corpus.common.garbled import load_vocab
from vi_corpus.common.state import read_json, write_json_atomic
from vi_corpus.vjol.extract import SYLLABLES_REL, TEXT_DIR
from vi_corpus.vjol.reocr import ALL_SIGNALS, REOCR_MIN_PAGES, REOCR_SHARE, REOCR_SIGNALS, page_defects, page_flags, pages_needed, pdf_producer_creator

logger = logging.getLogger("vi_corpus")

PLAN_DIR = "processed/vjol_reocr_plan"
PAGES_PER_SEC_PER_GPU = 1.55  # O-22
GPUS = 4
CORPUS_PDFS = 177_031  # số PDF của cả kho VJOL (J-41), để nhân ra ước lượng cả kho
REOCR_WARN_SHARE = 0.15  # trang OCR lại vượt tỷ lệ này số trang của mẫu thì báo chủ dự án chọn lại ngưỡng trước khi chạy GPU (J-42)
SHARE_SWEEP = (0.05, 0.10, 0.15, 0.25, 0.50)
SIGNAL_SETS = {"mặc định": REOCR_SIGNALS, "+gibberish": REOCR_SIGNALS + ("gibberish",), "+hyphen": REOCR_SIGNALS + ("hyphen",), "+joined_words": REOCR_SIGNALS + ("joined_words",), "tất cả": ALL_SIGNALS}
TOP_GROUPS = 25


def hours(pages: float) -> float:
    """Giờ OCR (đồng hồ) cho `pages` trang với PAGES_PER_SEC_PER_GPU trang/giây/GPU trên GPUS GPU."""
    return pages / PAGES_PER_SEC_PER_GPU / GPUS / 3600


def measure_docs(data_root: Path, vocab: frozenset[str] | None, with_pdf_meta: bool = True, metadata: dict[str, dict] | None = None) -> list[dict]:
    """
    Đo mọi checkpoint VJOL không lỗi.

    Args:
        data_root:     Thư mục gốc dữ liệu.
        vocab:         Tập âm tiết (None thì tone_shift / joined_words bằng 0).
        with_pdf_meta: True thì mở PDF gốc lấy (producer, creator) (chậm hơn một chút, cần raw/); False thì để trống.
        metadata:      {"<article_id>_<galley_id>": {"journal_slug": ...}} (records.load_metadata) để ghi tạp chí; None thì để trống.

    Returns:
        [{"sha256", "path", "article_id", "journal", "producer", "creator", "text_pages", "flags": {tín hiệu: số trang trúng}, "page_flags": [[tín hiệu của trang]] (mỗi trang có chữ một mục)}].
    """
    docs = []
    for ckpt in sorted((data_root / TEXT_DIR).glob("*.json")):
        done = read_json(ckpt)
        if not done or done.get("error") or done.get("skipped"):
            continue
        flags = [page_flags(page_defects(p["text"], vocab)) for p in done["pages"] if p["method"] in ("text", "tcvn3") and p["text"].strip()]
        path = Path(done["paths"][0])
        if not path.is_file():
            path = data_root / done["paths"][0]
        producer, creator = pdf_producer_creator(path) if with_pdf_meta and path.is_file() else ("", "")
        stem = Path(done["paths"][0]).stem
        docs.append({"sha256": done["sha256"], "path": done["paths"][0], "article_id": done.get("article_id"), "journal": ((metadata or {}).get(stem) or {}).get("journal_slug") or "",
                     "producer": producer, "creator": creator, "text_pages": len(flags), "page_flags": flags,
                     "flags": {sig: sum(sig in f for f in flags) for sig in ALL_SIGNALS}})
    return docs


def decide(doc: dict, signals: tuple[str, ...] = REOCR_SIGNALS, min_pages: int = REOCR_MIN_PAGES, share: float = REOCR_SHARE) -> tuple[bool, int]:
    """(bài có OCR lại cả bài không, số trang chữ sẽ OCR) theo quy tắc của reocr.reocr_reason từ số đo đã tính (page_flags)."""
    if not doc["text_pages"]:
        return False, 0
    hit = sum(any(s in signals for s in f) for f in doc["page_flags"])
    return (True, doc["text_pages"]) if hit >= pages_needed(doc["text_pages"], min_pages, share) else (False, 0)


def summarize(docs: list[dict], corpus_pdfs: int = CORPUS_PDFS) -> dict:
    """
    Tổng hợp các số đo (xem docstring module).

    Returns:
        {"docs", "text_pages", "docs_with_signal": {tín hiệu: số bài}, "pages_with_signal": {tín hiệu: số trang}, "default": {"docs", "pages", "hours", "pages_share", "hours_corpus", "pages_corpus"},
         "sweep": [{"signals", "share", "docs", "pages", "pages_share", "hours", "hours_corpus"}], "hyphen_only_docs": số bài chỉ có gạch nối thay ư (không vào mặc định),
         "by_producer_creator": [...], "by_journal": [...], "warning": chuỗi hoặc None}.
    """
    n_docs, n_pages = len(docs), sum(d["text_pages"] for d in docs)
    scale = corpus_pdfs / n_docs if n_docs else 0.0

    def row(signals: tuple[str, ...], share: float) -> dict:
        """Một dòng của bảng quét: số bài / trang OCR lại và giờ cho mẫu và cho cả kho."""
        picked = [decide(d, signals, REOCR_MIN_PAGES, share) for d in docs]
        pages = sum(p for _, p in picked)
        return {"signals": "+".join(signals), "share": share, "docs": sum(f for f, _ in picked), "pages": pages, "pages_share": round(pages / n_pages, 4) if n_pages else 0.0,
                "hours": round(hours(pages), 2), "hours_corpus": round(hours(pages * scale), 1)}

    default = row(REOCR_SIGNALS, REOCR_SHARE)
    hyphen_only = sum(1 for d in docs if d["flags"]["hyphen"] and not any(d["flags"][s] for s in REOCR_SIGNALS))

    def table(key) -> list[dict]:
        """Theo nhóm (producer, creator) hoặc tạp chí: số bài, số bài OCR lại (mặc định), số trang OCR lại; TOP_GROUPS nhóm nhiều trang OCR nhất."""
        acc: dict = defaultdict(Counter)
        for d in docs:
            flag, pages = decide(d)
            g = acc[key(d)]
            g["docs"] += 1
            g["docs_reocr"] += flag
            g["pages_reocr"] += pages
        return [{"key": list(k) if isinstance(k, tuple) else k, **dict(c)} for k, c in sorted(acc.items(), key=lambda kv: -kv[1]["pages_reocr"])[:TOP_GROUPS]]

    warning = (f"CẢNH BÁO: OCR lại {default['pages_share']:.1%} số trang của mẫu (> {REOCR_WARN_SHARE:.0%}); báo chủ dự án chọn lại ngưỡng trước khi chạy GPU (J-42)"
               if default["pages_share"] > REOCR_WARN_SHARE else None)
    return {"docs": n_docs, "text_pages": n_pages,
            "docs_with_signal": {s: sum(d["flags"][s] > 0 for d in docs) for s in ALL_SIGNALS}, "pages_with_signal": {s: sum(d["flags"][s] for d in docs) for s in ALL_SIGNALS},
            "default": {"docs": default["docs"], "pages": default["pages"], "pages_share": default["pages_share"], "hours": default["hours"], "hours_corpus": default["hours_corpus"],
                        "pages_corpus": round(default["pages"] * scale)},
            "sweep": [row(sigs, share) for sigs in SIGNAL_SETS.values() for share in SHARE_SWEEP], "hyphen_only_docs": hyphen_only,
            "by_producer_creator": table(lambda d: (d["producer"], d["creator"])), "by_journal": table(lambda d: d["journal"]),
            "assumptions": {"pages_per_sec_per_gpu": PAGES_PER_SEC_PER_GPU, "gpus": GPUS, "corpus_pdfs": corpus_pdfs, "min_pages": REOCR_MIN_PAGES, "share": REOCR_SHARE, "signals": list(REOCR_SIGNALS)},
            "warning": warning}


def write_plan(docs: list[dict], out_dir: Path) -> int:
    """Ghi reocr_plan.csv (mỗi bài có ít nhất một tín hiệu một dòng); trả về số dòng."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in docs:
        if not any(d["flags"].values()):
            continue
        flag, pages = decide(d)
        rows.append({"sha256": d["sha256"], "path": d["path"], "article_id": d["article_id"], "journal": d["journal"], "producer": d["producer"], "creator": d["creator"],
                     "text_pages": d["text_pages"], **{f"pages_{s}": d["flags"][s] for s in ALL_SIGNALS}, "reocr": int(flag),
                     "reasons": "+".join(s for s in REOCR_SIGNALS if d["flags"][s]) if flag else "", "pages_to_ocr": pages})
    with open(out_dir / "reocr_plan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["sha256"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def plan(data_root: Path, root: Path | None = None, syllables: Path | None = None, with_pdf_meta: bool = True, corpus_pdfs: int = CORPUS_PDFS) -> Path:
    """
    Chạy toàn bộ phép đo trên checkpoint VJOL hiện có và ghi kết quả vào <data_root>/processed/vjol_reocr_plan/.

    Args:
        root:          Thư mục raw/vjol (mặc định <data_root>/raw/vjol) để đọc metadata tạp chí; thiếu metadata thì cột tạp chí để trống.
        syllables:     Tập âm tiết (mặc định <data_root>/processed/ocr_bakeoff/syllables.json); thiếu thì tone_shift / joined_words không đo được (ghi cảnh báo).
        with_pdf_meta: Mở PDF gốc lấy (producer, creator).
        corpus_pdfs:   Số PDF của cả kho để nhân ước lượng.

    Returns:
        Thư mục kết quả.
    """
    from vi_corpus.vjol.records import METADATA_REL, load_metadata

    vocab = load_vocab(syllables or data_root / SYLLABLES_REL)
    if vocab is None:
        logger.warning("Không thấy tập âm tiết: tone_shift và joined_words không đo được, số đo chỉ gồm dropped_letters / gibberish / hyphen")
    meta_path = (root or data_root / "raw" / "vjol") / METADATA_REL
    stems = {Path(d["paths"][0]).stem for d in (read_json(p) or {"paths": [""]} for p in sorted((data_root / TEXT_DIR).glob("*.json")))}
    metadata = load_metadata(meta_path, stems) if meta_path.is_file() else {}
    docs = measure_docs(data_root, vocab, with_pdf_meta, metadata)
    out = data_root / PLAN_DIR
    summary = summarize(docs, corpus_pdfs)
    summary["vocab"] = vocab is not None
    summary["plan_rows"] = write_plan(docs, out)
    write_json_atomic(out / "summary.json", summary)
    logger.info("OCR lại cả bài: %s", {k: summary["default"][k] for k in ("docs", "pages", "hours", "hours_corpus")})
    return out
