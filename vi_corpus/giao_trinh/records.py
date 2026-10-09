"""
Bản ghi schema chung (CORPUS_SCHEMA) của giáo trình, sinh từ kết quả trích text, và xuất Parquet.

Mỗi tài liệu (một sha256 file) một bản ghi. Text là các trang đã làm sạch (vi_corpus.common.pdf_text.clean_pages,
áp dụng lúc sinh bản ghi, không lưu vào interim) nối thành một văn bản. Chưa phân loại sách/slide, ngôn ngữ,
chất lượng (các trường language, quality_band... để trống cho các stage sau).
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vi_corpus.common.pdf_text import clean_pages
from vi_corpus.common.registry import SourceSpec
from vi_corpus.common.schema import CORPUS_SCHEMA, make_doc_id, text_sha256
from vi_corpus.common.state import read_json
from vi_corpus.giao_trinh.extract import TEXT_DIR

OUT_PATH = "processed/giao_trinh/giao_trinh.parquet"
# Định dạng mà mỗi dòng đã là một đoạn (slide, đoạn văn Word, khối HTML / EPUB): clean_pages không bỏ tiêu đề chạy
# và không ghép dòng (xem clean_pages, slides=True). PDF / DJVU thì làm sạch theo trang như sách.
LINE_PARAGRAPH_FORMATS = frozenset({"pptx", "ppt", "docx", "doc", "html", "epub"})
BATCH = 64  # số bản ghi ghi mỗi lần, để không giữ cả kho trong RAM


def iter_records(spec: SourceSpec, data_root: Path, limit_files: int | None = None,
                 with_meta: bool = False) -> Iterator[dict]:
    """
    Sinh bản ghi CORPUS_SCHEMA từ interim/giao_trinh_text/, mỗi tài liệu một bản ghi.

    Bỏ qua checkpoint lỗi / bị skip / không có chữ. Trang needs_ocr (chưa OCR) bị bỏ vì text của chúng là rác.
    source_path là đường dẫn đầu tiên của file (các đường dẫn trùng nằm trong checkpoint). Slide, Word, HTML, EPUB
    (LINE_PARAGRAPH_FORMATS) mỗi dòng một đoạn và giữ mọi dòng; PDF / DJVU thì bỏ tiêu đề chạy và ghép dòng thành đoạn.

    Args:
        spec:        Nguồn giao_trinh trong registry.
        data_root:   Thư mục gốc dữ liệu.
        limit_files: Chỉ đọc N checkpoint đầu (chạy thử).
        with_meta:   Thêm khoá "meta" (JSON: định dạng, ngành, môn, số dòng tiêu đề / số trang bị clean_pages bỏ).
    """
    for path in sorted((data_root / TEXT_DIR).glob("*.json"))[:limit_files]:
        done = read_json(path)
        if not done or done.get("error") or done.get("skipped"):
            continue
        pages = [p["text"] for p in done["pages"] if p["method"] != "needs_ocr"]
        page_stats: dict = {}
        text = clean_pages(pages, slides=done["format"] in LINE_PARAGRAPH_FORMATS, stats=page_stats)
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
            rec["meta"] = json.dumps({"format": done["format"], "nganh": done.get("nganh"), "mon": done.get("mon"),
                                      "page_lines_removed": page_stats.get("lines_removed", 0)}, ensure_ascii=False)
        yield rec


def export_parquet(records: Iterable[dict], out_path: Path) -> int:
    """
    Ghi các bản ghi ra Parquet (CORPUS_SCHEMA), từng lô BATCH bản ghi, nguyên tử (file tạm rồi os.replace).

    Returns:
        Số bản ghi đã ghi.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    total, batch = 0, []
    with pq.ParquetWriter(tmp, CORPUS_SCHEMA) as writer:
        for rec in records:
            batch.append(rec)
            if len(batch) == BATCH:
                writer.write_table(pa.Table.from_pylist(batch, schema=CORPUS_SCHEMA))
                total, batch = total + len(batch), []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=CORPUS_SCHEMA))
            total += len(batch)
    tmp.replace(out_path)
    return total
