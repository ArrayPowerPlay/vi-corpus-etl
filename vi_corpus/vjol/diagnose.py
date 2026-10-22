"""
Công cụ chẩn đoán VJOL cho J-22 (1) và (3) A: đo trước, sửa sau (chỉ CPU, đọc checkpoint interim/vjol_text/ và PDF gốc).

- inspect_articles: in từng trang của các bài chỉ định (method, engine OCR, lý do, các tín hiệu chữ hỏng, độ lợi chữ đảo ngược, đoạn đầu của lớp chữ
  và của bản OCR) để xác định chữ đảo ngược (trang xoay 180 độ) đến từ LỚP CHỮ NHÚNG (method "text") hay từ OCR VLM (method "ocr_vlm").
- pdf_producers: đếm trường `producer` / `creator` trong metadata của PDF theo bài, kèm tỷ lệ âm tiết lạ trung bình của lớp chữ và số bài/trang
  trúng tín hiệu chữ hỏng theo từng producer, để quyết định có OCR lại cả nhóm producer sinh ra lớp chữ OCR cũ lỗi nhẹ hay không (J-22 (3) A).
Không sửa checkpoint nào.
"""

import csv
import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import pymupdf

from vi_corpus.common.garbled import SIGNALS, load_vocab, page_measures, reversed_gain, signals_from_measures
from vi_corpus.common.state import read_json, write_json_atomic
from vi_corpus.vjol.extract import SYLLABLES_REL, TEXT_DIR

logger = logging.getLogger("vi_corpus")

OUT_DIR = "processed/vjol_diagnose"
SNIPPET = 200


def load_checkpoints(data_root: Path) -> list[dict]:
    """Mọi checkpoint VJOL không lỗi và không bị bỏ qua (theo tên file)."""
    out = []
    for path in sorted((data_root / TEXT_DIR).glob("*.json")):
        done = read_json(path)
        if done and not done.get("error") and not done.get("skipped"):
            out.append(done)
    return out


def inspect_articles(data_root: Path, ids: list[str], syllables: Path | None = None) -> str:
    """
    Báo cáo từng trang của các bài `ids`, dạng văn bản. Mỗi id là article_id, tiền tố sha256 của PDF, hoặc đoạn băm giữa của doc_id / parent_doc_id
    trong báo cáo pipeline ("vjol:<băm>:0" -> <băm>, là sha1 của sha256 chứ không phải tiền tố sha256).

    Với mỗi trang: method, engine OCR (tier) và status nếu là ocr_vlm, lý do OCR lại (reason), các tín hiệu chữ hỏng vượt ngưỡng của LỚP CHỮ
    (garbled_text nếu trang đã OCR lại, ngược lại text), độ lợi chữ đảo ngược của văn bản hiện tại và đoạn đầu.

    Returns:
        Văn bản báo cáo (một khối cho mỗi bài; bài không thấy được ghi rõ).
    """
    vocab = load_vocab(syllables or data_root / SYLLABLES_REL)
    by_key = {}
    for done in load_checkpoints(data_root):
        by_key[str(done.get("article_id"))] = done
        by_key[done["sha256"][:10]] = done
        by_key[hashlib.sha1(done["sha256"].encode("utf-8")).hexdigest()[:10]] = done  # đoạn băm trong doc_id (make_doc_id)
    lines = [f"tập âm tiết: {'có' if vocab else 'THIẾU (không đo được độ lợi đảo ngược và tín hiệu unknown)'}"]
    for key in ids:
        done = by_key.get(key) or next((d for k, d in by_key.items() if k.startswith(key)), None)
        if done is None:
            lines.append(f"=== {key}: không thấy checkpoint")
            continue
        lines.append(f"=== bài {done.get('article_id')} sha {done['sha256'][:10]} {done['paths'][0]} ({len(done['pages'])} trang)")
        for i, pg in enumerate(done["pages"]):
            layer = pg.get("garbled_text") if pg.get("garbled_text") is not None else pg["text"]
            sig = signals_from_measures(page_measures(layer, vocab)) if pg["method"] in ("text", "tcvn3") or pg.get("garbled_text") else []
            gain = reversed_gain(pg["text"], vocab) if vocab else (0.0, 0.0, 0)
            lines.append(f"  trang {i + 1}: method={pg['method']} tier={pg.get('tier', '-')} status={pg.get('status', '-')} reason={pg.get('reason', '-')} "
                         f"tín hiệu lớp chữ={'+'.join(sig) or '-'} độ lợi đảo={gain[0]:+.2f} (ngược hợp lệ {gain[1]:.2f}, {gain[2]} từ)")
            lines.append(f"      text: {pg['text'][:SNIPPET]!r}")
            if pg.get("garbled_text"):
                lines.append(f"      lớp chữ gốc: {pg['garbled_text'][:SNIPPET]!r}")
    return "\n".join(lines)


def pdf_producers(data_root: Path, syllables: Path | None = None) -> Path:
    """
    Đo phân bố producer / creator của PDF trong checkpoint VJOL (J-22 (3) A) rồi ghi <data-root>/processed/vjol_diagnose/.

    Ghi producers.csv (producer, creator, số bài, số trang có lớp chữ, tỷ lệ âm tiết lạ trung bình của lớp chữ, số trang trúng tín hiệu chữ hỏng)
    và summary.json. Mở PDF gốc chỉ để đọc metadata, không đọc nội dung trang.

    Returns:
        Thư mục kết quả.
    """
    vocab = load_vocab(syllables or data_root / SYLLABLES_REL)
    groups: dict[tuple[str, str], dict] = defaultdict(lambda: {"docs": 0, "pages": 0, "unknown_sum": 0.0, "unknown_n": 0, "suspect": 0})
    missing = 0
    for done in load_checkpoints(data_root):
        path = Path(done["paths"][0])
        path = path if path.is_absolute() else data_root / path
        try:
            with pymupdf.open(path) as doc:
                meta = doc.metadata or {}
        except Exception as exc:  # PDF hỏng hoặc không còn ở đường dẫn cũ: ghi nhận, không dừng
            logger.warning("Không đọc được metadata %s: %s", path, exc)
            missing += 1
            continue
        key = ((meta.get("producer") or "").strip()[:80], (meta.get("creator") or "").strip()[:80])
        g = groups[key]
        g["docs"] += 1
        for pg in done["pages"]:
            if pg["method"] not in ("text", "tcvn3") and pg.get("garbled_text") is None:
                continue
            layer = pg.get("garbled_text") if pg.get("garbled_text") is not None else pg["text"]
            m = page_measures(layer, vocab)
            g["pages"] += 1
            if m["unknown"] is not None and m["unknown"][1] >= 30:
                g["unknown_sum"] += m["unknown"][0]
                g["unknown_n"] += 1
            g["suspect"] += bool(signals_from_measures(m))
    out_dir = data_root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(({"producer": k[0], "creator": k[1], "docs": g["docs"], "text_pages": g["pages"],
                    "mean_unknown_rate": round(g["unknown_sum"] / g["unknown_n"], 4) if g["unknown_n"] else "",
                    "suspect_pages": g["suspect"]} for k, g in groups.items()), key=lambda r: -r["docs"])
    with open(out_dir / "producers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["producer", "creator", "docs", "text_pages", "mean_unknown_rate", "suspect_pages"])
        w.writeheader()
        w.writerows(rows)
    write_json_atomic(out_dir / "summary.json", {"docs": sum(r["docs"] for r in rows), "unreadable_pdfs": missing, "producers": len(rows),
                                                 "signals": list(SIGNALS), "vocab": vocab is not None, "top": rows[:30]})
    return out_dir
