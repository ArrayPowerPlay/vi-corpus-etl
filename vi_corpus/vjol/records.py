"""
Bản ghi schema chung (CORPUS_SCHEMA) của VJOL, sinh từ kết quả trích text (interim/vjol_text/) và metadata của tạp chí.

Mỗi bài (một sha256 PDF) một bản ghi. Text là các trang đã làm sạch: bỏ trang bìa / mục lục / tài liệu tham khảo
(vi_corpus.vjol.clean) rồi vi_corpus.common.pdf_text.clean_pages (tiêu đề chạy, số trang, ghép dòng), áp dụng lúc sinh bản ghi
chứ không lưu vào interim. Quyền (rights_status) lấy mặc định của registry ("unknown"): chỉ gắn nhãn, không loại (R-01).

Metadata (J-03): raw/vjol/metadata/vjol_merged.jsonl (hàng trăm MB, mỗi dòng một bài) được đọc theo luồng MỘT lượt, chỉ giữ
dòng của các bài có trong checkpoint, nối qua article_id (khớp tên file <article_id>_<galley_id>.pdf). PDF không có dòng
metadata vẫn thành bản ghi, chỉ thiếu trường. Nhãn ngôn ngữ trong metadata chỉ nằm ở `meta` (khoá journal_language_raw), không thay language id của pipeline.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from vi_corpus.common.pdf_text import clean_pages
from vi_corpus.common.registry import SourceSpec
from vi_corpus.common.schema import make_doc_id, text_sha256
from vi_corpus.common.state import read_json
from vi_corpus.vjol.clean import clean_paper_pages
from vi_corpus.vjol.extract import PDF_STEM, TEXT_DIR

logger = logging.getLogger("vi_corpus")

OUT_PATH = "processed/vjol/vjol.parquet"
METADATA_REL = "metadata/vjol_merged.jsonl"  # tương đối so với thư mục raw/vjol
# Trường metadata giữ lại (bỏ abstract và các trường nặng khác để không phình RAM khi kho lớn).
META_FIELDS = ("journal_slug", "journal_title", "year", "language", "doi", "title", "keywords")
# Trường `language` của metadata là nhãn của tạp chí, gần như luôn "en" cho cả bài tiếng Việt (J-22), nên đổi tên khi ghi vào meta để không ai dùng nhầm.
META_RENAMES = {"language": "journal_language_raw"}
META_OUT = tuple(META_RENAMES.get(k, k) for k in META_FIELDS)


def load_metadata(path: Path, stems: set[str]) -> dict[str, dict]:
    """
    Đọc metadata theo luồng một lượt, giữ dòng của các bài có trong `stems`.

    Args:
        path:  File jsonl (mỗi dòng một bài, có article_id và galleys[].galley_id).
        stems: Tên file PDF không đuôi, dạng "<article_id>_<galley_id>".

    Returns:
        {stem: {META_OUT..., "galley_match": galley_id có trong galleys[] của bài hay không}}. Bài không có dòng metadata
        vắng mặt. Dòng JSON hỏng bị bỏ (đếm và ghi log). File không tồn tại thì trả về {} (ghi log cảnh báo).
    """
    if not path.is_file():
        logger.warning("Không thấy metadata %s: bản ghi VJOL sẽ thiếu tiêu đề / tạp chí / năm", path)
        return {}
    wanted: dict[str, list[str]] = {}
    for stem in stems:
        m = PDF_STEM.match(stem)
        if m:
            wanted.setdefault(m.group(1), []).append(stem)
    rows: dict[str, dict] = {}
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(row, dict) and str(row.get("article_id")) in wanted:
                rows[str(row["article_id"])] = row
    if bad:
        logger.warning("Metadata %s: bỏ %d dòng JSON hỏng", path, bad)
    out = {}
    for aid, row in rows.items():
        galleys = {str(g.get("galley_id")) for g in row.get("galleys") or [] if isinstance(g, dict)}
        for stem in wanted[aid]:
            out[stem] = {**{META_RENAMES.get(k, k): row.get(k) for k in META_FIELDS}, "galley_match": stem.split("_")[1] in galleys}
    return out


def iter_records(spec: SourceSpec, data_root: Path, limit_files: int | None = None, with_meta: bool = False,
                 metadata_path: Path | None = None) -> Iterator[dict]:
    """
    Sinh bản ghi CORPUS_SCHEMA từ interim/vjol_text/, mỗi bài một bản ghi.

    Bỏ qua checkpoint lỗi / bị skip / không có chữ. Trang needs_ocr (chưa OCR) thành trang rỗng vì text của chúng là rác; trang OCR bị loại
    (flagged) có text rỗng nên không vào kho. source_path là đường dẫn PDF (nằm trong checkpoint).

    Args:
        spec:          Nguồn vjol trong registry.
        data_root:     Thư mục gốc dữ liệu.
        limit_files:   Chỉ đọc N checkpoint đầu (chạy thử).
        with_meta:     Thêm khoá "meta" (JSON): article_id, galley_id, journal_slug, journal_title, year, journal_language_raw, doi, title, keywords,
                       có/không metadata, số trang lỗi font phải OCR lại (J-10), số trang / dòng bị bỏ (bìa, mục lục, tài liệu tham khảo, tiêu đề chạy).
        metadata_path: File metadata (mặc định <data-root>/<spec.path>/metadata/vjol_merged.jsonl); chỉ đọc khi with_meta.
    """
    checkpoints = []
    for path in sorted((data_root / TEXT_DIR).glob("*.json"))[:limit_files]:
        done = read_json(path)
        if done and not done.get("error") and not done.get("skipped"):
            checkpoints.append(done)
    meta_rows: dict[str, dict] = {}
    if with_meta:
        stems = {Path(d["paths"][0]).stem for d in checkpoints}
        meta_rows = load_metadata(metadata_path or data_root / spec.path / METADATA_REL, stems)
    for done in checkpoints:
        pages = ["" if p["method"] == "needs_ocr" else p["text"] for p in done["pages"]]  # giữ chỉ số trang cho heuristic bìa
        md = meta_rows.get(Path(done["paths"][0]).stem)
        pages, clean_stats = clean_paper_pages(pages, (md or {}).get("journal_title"))
        page_stats: dict = {}
        text = clean_pages(pages, stats=page_stats)
        if not text:
            continue
        rec = {
            "doc_id": make_doc_id(spec.key, done["sha256"], 0),
            "source_key": spec.key,
            "source_path": done["paths"][0],
            "source_sha256": text_sha256(text),
            "text": text,
            "language": None,
            "domain": spec.domain,
            "license": spec.license,
            "owner": spec.owner,
            "rights_status": spec.rights_status,
            "quality_band": None,
            "reason_codes": [],
            "token_count": None,
            "dedup_family_id": None,
        }
        if with_meta:
            year = (md or {}).get("year") or done.get("year_dir")
            rec["meta"] = json.dumps({
                "article_id": done.get("article_id"), "galley_id": done.get("galley_id"),
                **({k: md[k] for k in META_OUT if k != "year"} if md else {k: None for k in META_OUT if k != "year"}),
                "year": year, "year_dir": done.get("year_dir"), "metadata_found": md is not None,
                "galley_match": bool(md and md["galley_match"]), "pages": len(done["pages"]),
                "garbled_pages": sum(bool(p.get("reason")) for p in done["pages"]),
                "page_lines_removed": page_stats.get("lines_removed", 0), **clean_stats,
            }, ensure_ascii=False)
        yield rec
