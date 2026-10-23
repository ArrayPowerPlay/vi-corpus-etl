"""
Công cụ chẩn đoán VJOL cho J-22 (1) và (3) A: đo trước, sửa sau (chỉ CPU, đọc checkpoint interim/vjol_text/ và PDF gốc).

- inspect_articles: in từng trang của các bài chỉ định (method, engine OCR, lý do, các tín hiệu chữ hỏng, độ lợi chữ đảo ngược, đoạn đầu của lớp chữ
  và của bản OCR) để xác định chữ đảo ngược (trang xoay 180 độ) đến từ LỚP CHỮ NHÚNG (method "text") hay từ OCR VLM (method "ocr_vlm").
- pdf_producers: đếm trường `producer` / `creator` trong metadata của PDF theo bài, kèm tỷ lệ âm tiết lạ trung bình của lớp chữ và số bài/trang
  trúng tín hiệu chữ hỏng theo từng producer, để quyết định có OCR lại cả nhóm producer sinh ra lớp chữ OCR cũ lỗi nhẹ hay không (J-22 (3) A).
- pdf_producers còn ghi producers_docs.csv (J-26): MỘT DÒNG MỖI BÀI để nối với kết quả pipeline; chunks_by_producer nối bảng đó với 05_dedup.parquet của một lần chạy và đếm đoạn
  xấu còn lọt theo producer (có phải họ OCR cũ / iLovePDF sinh ra chúng không), trước khi đặt quy tắc "lớp chữ do OCR sau tạo ra thì không đáng tin" (J-25 D2).
Không sửa checkpoint nào.
"""

import csv
import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import pymupdf

from vi_corpus.common.garbled import SIGNALS, load_unaccented, load_vocab, page_measures, reversed_gain, signals_from_measures
from vi_corpus.common.state import read_json, write_json_atomic
from vi_corpus.pipeline.io import read_rows
from vi_corpus.vjol.audit import KEPT_BANDS, meta_of, metrics_of, suspicious
from vi_corpus.vjol.extract import SYLLABLES_REL, TEXT_DIR, UNACCENTED_REL

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
    unaccented = load_unaccented(data_root / UNACCENTED_REL)
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
            sig = signals_from_measures(page_measures(layer, vocab, unaccented)) if pg["method"] in ("text", "tcvn3") or pg.get("garbled_text") else []
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
    unaccented = load_unaccented(data_root / UNACCENTED_REL)
    docs_rows: list[dict] = []
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
        d_pages = d_suspect = d_unknown_n = 0
        d_unknown_sum = 0.0
        for pg in done["pages"]:
            if pg["method"] not in ("text", "tcvn3") and pg.get("garbled_text") is None:
                continue
            layer = pg.get("garbled_text") if pg.get("garbled_text") is not None else pg["text"]
            m = page_measures(layer, vocab, unaccented)
            g["pages"] += 1
            d_pages += 1
            if m["unknown"] is not None and m["unknown"][1] >= 30:
                g["unknown_sum"] += m["unknown"][0]
                g["unknown_n"] += 1
                d_unknown_sum += m["unknown"][0]
                d_unknown_n += 1
            flagged = bool(signals_from_measures(m))
            g["suspect"] += flagged
            d_suspect += flagged
        docs_rows.append({"article_id": done.get("article_id"), "sha256": done["sha256"][:10], "doc_hash": hashlib.sha1(done["sha256"].encode("utf-8")).hexdigest()[:10],
                          "year": done.get("year_dir"), "producer": key[0], "creator": key[1], "text_pages": d_pages, "suspect_pages": d_suspect,
                          "mean_unknown_rate": round(d_unknown_sum / d_unknown_n, 4) if d_unknown_n else ""})
    out_dir = data_root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(({"producer": k[0], "creator": k[1], "docs": g["docs"], "text_pages": g["pages"],
                    "mean_unknown_rate": round(g["unknown_sum"] / g["unknown_n"], 4) if g["unknown_n"] else "",
                    "suspect_pages": g["suspect"]} for k, g in groups.items()), key=lambda r: -r["docs"])
    with open(out_dir / "producers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["producer", "creator", "docs", "text_pages", "mean_unknown_rate", "suspect_pages"])
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "producers_docs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["article_id", "sha256", "doc_hash", "year", "producer", "creator", "text_pages", "suspect_pages", "mean_unknown_rate"])
        w.writeheader()
        w.writerows(sorted(docs_rows, key=lambda r: str(r["article_id"])))
    write_json_atomic(out_dir / "summary.json", {"docs": sum(r["docs"] for r in rows), "unreadable_pdfs": missing, "producers": len(rows),
                                                 "signals": list(SIGNALS), "vocab": vocab is not None, "top": rows[:30]})
    return out_dir


def chunks_by_producer(data_root: Path, run_dir: Path) -> Path:
    """
    Nối producers_docs.csv với 05_dedup.parquet của một lần chạy pipeline (qua meta.article_id) và đếm đoạn xấu còn lọt theo producer (J-26).

    Đoạn xấu = đoạn đã giữ (band A / B / C) mà có số đo nghi ngờ (vi_corpus.vjol.audit.suspicious) hoặc tỷ lệ từ lạ >= 0,15. Ghi <data-root>/processed/vjol_diagnose/producers_chunks.csv:
    producer, creator, docs, chunks, bad_chunks, bad_share, nguyên nhân hàng đầu. Cần chạy `producers` trước.

    Returns:
        Đường dẫn producers_chunks.csv.

    Raises:
        FileNotFoundError: chưa có producers_docs.csv.
    """
    docs_path = data_root / OUT_DIR / "producers_docs.csv"
    if not docs_path.is_file():
        raise FileNotFoundError(f"Chưa có {docs_path}: chạy `diagnose.py producers` trước")
    with open(docs_path, encoding="utf-8") as f:
        producer_of = {str(r["article_id"]): (r["producer"], r["creator"]) for r in csv.DictReader(f)}
    docs_in: dict[tuple[str, str], set] = defaultdict(set)
    stats: dict[tuple[str, str], dict] = defaultdict(lambda: {"chunks": 0, "bad": 0, "why": defaultdict(int)})
    unmatched = 0
    for row in read_rows(run_dir / "05_dedup.parquet"):
        if row["source_key"] != "vjol" or row["quality_band"] not in KEPT_BANDS:
            continue
        aid = str(meta_of(row).get("article_id"))
        key = producer_of.get(aid)
        if key is None:
            unmatched += 1
            continue
        docs_in[key].add(aid)
        st = stats[key]
        st["chunks"] += 1
        why = suspicious(row) + (["odd_word_0.15"] if metrics_of(row).get("odd_word", 0.0) >= 0.15 else [])
        if why:
            st["bad"] += 1
            for w in why:
                st["why"][w] += 1
    out = data_root / OUT_DIR / "producers_chunks.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["producer", "creator", "docs", "chunks", "bad_chunks", "bad_share", "top_causes"])
        for key, st in sorted(stats.items(), key=lambda kv: -kv[1]["bad"]):
            top = "; ".join(f"{k}:{n}" for k, n in sorted(st["why"].items(), key=lambda kv: -kv[1])[:3])
            w.writerow([key[0], key[1], len(docs_in[key]), st["chunks"], st["bad"], round(st["bad"] / st["chunks"], 4), top])
    logger.info("producers_chunks: %d đoạn không nối được với bài (thiếu trong producers_docs.csv)", unmatched)
    return out
