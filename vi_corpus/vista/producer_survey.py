"""
Khảo sát producer của PDF VISTA (VS-05), chạy trên CPU: producer nào có lớp chữ mất dấu / hỏng mà luật "OCR cả bài" (OLD_OCR_PRODUCERS) chưa phủ.

Với mỗi PDF trong mẫu chia đều theo năm: đọc metadata (producer, creator), số trang, lớp chữ của vài trang đầu, rồi dùng bộ dò chữ hỏng của VJOL
(vi_corpus.common.garbled.page_signals, cả tín hiệu "nodiac" mất dấu) để biết lớp chữ có đáng nghi không. Gộp theo producer (30 ký tự đầu):
- docs: số PDF; years: khoảng năm; text_share: tỷ lệ PDF có lớp chữ (>= MIN_TEXT_CHARS ký tự ở các trang đầu); suspect_share: trong số PDF có lớp chữ, tỷ lệ bị bộ dò nghi;
- covered: producer đã khớp luật OCR cả bài; uncovered_suspect: số PDF bị nghi mà luật chưa phủ (nơi cần thêm producer vào OLD_OCR_PRODUCERS).
Chỉ nhìn các trang đầu (không đọc cả bài) nên rẻ; PDF tiếng Anh hoặc có dấu đúng không bị nghi.
"""

import csv
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

from vi_corpus.common.garbled import DEFAULT_THRESHOLDS, load_unaccented, load_vocab, page_signals
from vi_corpus.common.pdf_text import _producer_matches
from vi_corpus.vista.layout import OCR_ALL_PRODUCERS
from vi_corpus.vjol.extract import SYLLABLES_REL, UNACCENTED_REL, list_pdfs_by_year, sample_pdfs

logger = logging.getLogger("vi_corpus")

HEAD_PAGES = 3  # số trang đầu lấy lớp chữ
MIN_TEXT_CHARS = 200  # ít hơn thì coi như PDF không có lớp chữ (scan thuần)
OUT_DIR = "processed/vista_producer_survey"

_VOCAB = None
_UNACCENTED = None


def _init_worker(vocab, unaccented) -> None:
    """Đặt tập âm tiết và danh sách từ không dấu cho tiến trình con (nạp một lần, không gửi theo từng PDF)."""
    global _VOCAB, _UNACCENTED
    _VOCAB, _UNACCENTED = vocab, unaccented


def survey_pdf(path: str) -> dict:
    """
    Khảo sát một PDF: {"path", "year", "producer", "creator", "pages", "text_chars", "signals": [tín hiệu của lớp chữ các trang đầu], "error"}.

    Không ném lỗi của file (PDF hỏng -> error). Lớp chữ các trang đầu được nối lại thành một khối để dò (đủ từ cho ngưỡng tối thiểu của bộ dò).
    """
    p = Path(path)
    out = {"path": path, "year": p.parent.name, "producer": "", "creator": "", "pages": 0, "text_chars": 0, "signals": [], "error": None}
    try:
        with pymupdf.open(p) as doc:
            meta = doc.metadata or {}
            out.update(producer=(meta.get("producer") or "").strip(), creator=(meta.get("creator") or "").strip(), pages=doc.page_count)
            text = "\n".join(doc[i].get_text() for i in range(min(HEAD_PAGES, doc.page_count)))
    except Exception as exc:  # noqa: BLE001 — một PDF hỏng không được dừng cả lượt khảo sát
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return out
    out["text_chars"] = len(text.strip())
    if out["text_chars"] >= MIN_TEXT_CHARS:
        out["signals"] = page_signals(text, DEFAULT_THRESHOLDS, _VOCAB, _UNACCENTED)
    return out


def summarize(rows: list[dict]) -> list[dict]:
    """Gộp kết quả survey_pdf theo producer (30 ký tự đầu), xếp theo số PDF bị nghi mà luật chưa phủ giảm dần rồi số PDF giảm dần; các cột xem đầu file."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r["error"]:
            groups[r["producer"][:30]].append(r)
    table = []
    for producer, items in groups.items():
        with_text = [r for r in items if r["text_chars"] >= MIN_TEXT_CHARS]
        suspect = [r for r in with_text if r["signals"]]
        covered = [r for r in items if _producer_matches({"producer": r["producer"], "creator": r["creator"]}, OCR_ALL_PRODUCERS)]
        years = sorted(r["year"] for r in items if r["year"].isdigit())
        table.append({"producer": producer, "docs": len(items), "years": f"{years[0]}-{years[-1]}" if years else "", "text_share": round(len(with_text) / len(items), 3),
                      "suspect_share": round(len(suspect) / len(with_text), 3) if with_text else None, "covered": len(covered) == len(items),
                      "uncovered_suspect": sum(1 for r in suspect if r not in covered), "signals": sorted({s for r in suspect for s in r["signals"]})})
    return sorted(table, key=lambda t: (-t["uncovered_suspect"], -t["docs"]))


def run_survey(root: Path, data_root: Path, sample: int, seed: int = 0, workers: int | None = None) -> dict:
    """
    Khảo sát mẫu `sample` PDF của raw/vista (chia đều theo năm, cùng cách chọn mẫu với trích text), ghi bảng ra processed/vista_producer_survey/.

    Returns:
        {"pdfs", "errors", "no_text", "suspect", "uncovered_suspect", "table": [...]}. Ghi producers.csv và survey.json trong OUT_DIR.
    """
    files = sample_pdfs(list_pdfs_by_year(root), sample, seed)
    vocab = load_vocab(data_root / SYLLABLES_REL)
    unaccented = load_unaccented(data_root / UNACCENTED_REL)
    if vocab is None:
        logger.warning("Không thấy tập âm tiết %s: bộ dò bỏ tín hiệu âm tiết lạ (vẫn có tín hiệu mất dấu)", data_root / SYLLABLES_REL)
    with ProcessPoolExecutor(workers or os.cpu_count() or 1, initializer=_init_worker, initargs=(vocab, unaccented)) as pool:
        rows = list(pool.map(survey_pdf, map(str, files), chunksize=64))
    table = summarize(rows)
    result = {"pdfs": len(rows), "errors": sum(bool(r["error"]) for r in rows), "no_text": sum(1 for r in rows if not r["error"] and r["text_chars"] < MIN_TEXT_CHARS),
              "suspect": sum(1 for r in rows if r["signals"]), "uncovered_suspect": sum(t["uncovered_suspect"] for t in table), "table": table}
    out = data_root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "survey.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "producers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]) if table else ["producer"])
        w.writeheader()
        w.writerows({**t, "signals": " ".join(t["signals"])} for t in table)
    return result
