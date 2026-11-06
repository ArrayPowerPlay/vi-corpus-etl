"""
Bản ghi schema chung (CORPUS_SCHEMA) của VISTA, sinh từ checkpoint trích text (interim/vista_text/) và vista_detail.jsonl.

Vòng sinh bản ghi và làm sạch (bìa, mục lục, tài liệu tham khảo, tiêu đề chạy, sửa chữ) là của VJOL, dùng lại nguyên (vi_corpus.vjol.records.iter_paper_records);
file này chỉ nói cách nối metadata của VISTA.

Một PDF duy nhất (một sha256) một bản ghi (VS-03). Nhiều bài VISTA có thể cùng một nội dung PDF (9.708 nhóm, nhóm lớn nhất 15 bài): mã của mọi bài đó lấy từ
`paths` của checkpoint (mỗi file raw đặt tên <publication_id>.pdf) và ghi vào meta.publication_ids; meta.shared_pdf = True khi có từ 2 bài. Trường metadata
(tiêu đề, tạp chí, năm ...) lấy từ bài có mã nhỏ nhất trong nhóm có dòng trong vista_detail.jsonl (meta.publication_id); PDF không có dòng nào (kể cả
unknown/<sha256>.pdf) vẫn thành bản ghi, chỉ thiếu trường (metadata_found = False).

vista_detail.jsonl (~640 MB, hơn 320 nghìn dòng) được đọc theo luồng MỘT lượt, chỉ giữ các trường nhẹ (không giữ tóm tắt) của bài được chọn cho từng PDF.
Tên trường khớp với VJOL để phần dùng chung đọc được: `journal_title` (= source_name của VISTA, dùng cắt tiêu đề chạy), `year` (= issue_year), `title`.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from vi_corpus.common.registry import SourceSpec
from vi_corpus.vista.layout import METADATA_REL, TEXT_DIR, publication_id_of
from vi_corpus.vjol.records import RecordAdapter, iter_paper_records

logger = logging.getLogger("vi_corpus")

# Trường của dòng vista_detail.jsonl giữ lại -> tên trong meta. `page_range` (khoảng trang in trên tạp chí) khác `pages` (số trang PDF) của phần dùng chung.
DETAIL_FIELDS = {"title_vi": "title", "title_en": "title_en", "authors": "authors", "keywords": "keywords", "source_name": "journal_title", "issn": "issn",
                 "issue_year": "year", "issue_number": "issue_number", "pages": "page_range", "research_field_code": "research_field_code",
                 "research_field_name": "research_field_name", "document_type": "document_type"}
META_OUT = tuple(v for v in DETAIL_FIELDS.values() if v != "year")


def publication_ids(done: dict) -> list[str]:
    """Mã bài (không trùng, xếp theo số) của một checkpoint, lấy từ tên mọi file trong `paths`; rỗng với PDF không đặt tên theo mã bài."""
    ids = {pid for pid in map(publication_id_of, done.get("paths") or []) if pid}
    return sorted(ids, key=int)


def load_metadata(path: Path, checkpoints: list[dict]) -> dict[str, dict]:
    """
    Đọc vista_detail.jsonl theo luồng một lượt và chọn dòng metadata cho từng checkpoint.

    Args:
        path:        File vista_detail.jsonl (mỗi dòng một bài, khoá publication_id).
        checkpoints: Các checkpoint trích text (cần `sha256` và `paths`).

    Returns:
        {sha256: {DETAIL_FIELDS đã đổi tên, "publication_id": mã bài được chọn}}: bài có mã nhỏ nhất trong nhóm có dòng metadata. Checkpoint không bài nào có dòng thì vắng mặt.
        Dòng JSON hỏng bị bỏ (đếm và ghi log). File không tồn tại thì trả về {} (ghi log cảnh báo).
    """
    if not path.is_file():
        logger.warning("Không thấy metadata %s: bản ghi VISTA sẽ thiếu tiêu đề / tạp chí / năm", path)
        return {}
    wanted: dict[str, list[str]] = {}  # mã bài -> các sha256 chứa nó
    for done in checkpoints:
        for pid in publication_ids(done):
            wanted.setdefault(pid, []).append(done["sha256"])
    chosen: dict[str, dict] = {}
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
            pid = str(row.get("publication_id")) if isinstance(row, dict) else None
            if pid not in wanted:
                continue
            md = {"publication_id": pid, **{new: row.get(old) for old, new in DETAIL_FIELDS.items()}}
            for sha in wanted[pid]:
                if sha not in chosen or int(pid) < int(chosen[sha]["publication_id"]):
                    chosen[sha] = md
    if bad:
        logger.warning("Metadata %s: bỏ %d dòng JSON hỏng", path, bad)
    return chosen


def meta_head(done: dict, md: dict | None, year: int | None) -> dict:
    """Phần đầu `meta` của bản ghi VISTA: mã bài được chọn, mọi mã bài dùng chung PDF, cờ shared_pdf / metadata_found, các trường metadata, năm."""
    ids = publication_ids(done)
    return {
        "publication_id": md["publication_id"] if md else (ids[0] if ids else None), "publication_ids": ids, "shared_pdf": len(ids) > 1,
        "metadata_found": md is not None,
        **({k: md[k] for k in META_OUT} if md else {k: None for k in META_OUT}),
        "year": year, "year_dir": done.get("year_dir"),
    }


VISTA_ADAPTER = RecordAdapter(label="VISTA", text_dir=TEXT_DIR, metadata_rel=METADATA_REL, load_metadata=load_metadata, meta_head=meta_head)


def iter_records(spec: SourceSpec, data_root: Path, limit_files: int | None = None, with_meta: bool = False,
                 metadata_path: Path | None = None, hyphen_chooser=None) -> Iterator[dict]:
    """Bản ghi VISTA: iter_paper_records với VISTA_ADAPTER, mỗi PDF duy nhất một bản ghi (xem vi_corpus.vjol.records.iter_paper_records về các tham số)."""
    yield from iter_paper_records(spec, data_root, VISTA_ADAPTER, limit_files, with_meta, metadata_path, hyphen_chooser)
