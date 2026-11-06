"""
Bố cục dữ liệu VISTA (bài báo khoa học sti.vista.gov.vn, PDF) cho bộ xử lý bài báo dùng chung với VJOL (VS-02, VS-03).

Dữ liệu chép từ MinIO `datacrawl/sti.vista.gov.vn/` về `<data-root>/raw/vista/` (chữ thường như `raw/vjol`):
    <năm>/<publication_id>.pdf      mã bài của VISTA; nhiều bài có thể cùng một nội dung PDF (checkpoint theo sha256 nên chỉ trích một lần,
                                    `paths` của checkpoint liệt kê mọi file nên mã bài suy ra được từ đó, xem vi_corpus.vista.records)
    unknown/<sha256>.pdf            19 file thử cũ không có trong metadata (VS-03: giữ, bản ghi thiếu metadata)
    metadata/vista_detail.jsonl     một dòng một bài (khoá publication_id), cùng vista_listing.jsonl, pdf_download_log.jsonl, vista_partitions.json

Khác VJOL: tên file là mã bài (không có galley_id); thư mục checkpoint, khoá, nhóm font xấu riêng; có thêm bộ lọc producer OCR cũ (VS-05): PDF do phần mềm OCR
đời cũ tạo ra (lớp chữ mất dấu, sai chữ) thì cả bài đi OCR bằng chuỗi VLM thay vì tin lớp chữ. Danh sách producer xem OLD_OCR_PRODUCERS.
"""

import re
from pathlib import Path

from vi_corpus.vjol.extract import OCR_ALL_PRODUCERS as VJOL_OCR_ALL_PRODUCERS
from vi_corpus.vjol.extract import YEAR_DIR
from vi_corpus.vjol.layout import PaperLayout

TEXT_DIR = "interim/vista_text"
CLAIM_DIR = "state/vista_extract/claims"
OUT_PATH = "processed/vista/vista.parquet"
METADATA_REL = "metadata/vista_detail.jsonl"  # tương đối so với thư mục raw/vista
FONT_GROUPS_REL = "processed/vista_font_canary/font_groups_bad.json"  # mỗi nguồn học nhóm font xấu riêng (J-42 (4))
# VS-05: chủ dự án bật OCR cả bài cho PDF do phần mềm OCR đời cũ tạo, không chờ số đo V2. Cặp (đầu của producer, chuỗi con của creator; "" khớp mọi creator). Ba producer thấy ở mẫu
# 85 PDF (2009-2019): "OCR Manager", "ORPALIS PDF Reducer", "GdPicture Managed PDF Plugin". Đã có thêm cặp ScanGate của VJOL. Thêm producer khác thì sửa danh sách này
# (extract_version đổi theo băm danh sách nên checkpoint cũ tự bị làm lại); scripts/vista/producer_survey.py liệt kê producer nào còn lớp chữ mất dấu mà chưa nằm trong danh sách.
OLD_OCR_PRODUCERS: tuple[tuple[str, str], ...] = (("OCR Manager", ""), ("ORPALIS", ""), ("GdPicture", ""))
OCR_ALL_PRODUCERS: tuple[tuple[str, str], ...] = VJOL_OCR_ALL_PRODUCERS + OLD_OCR_PRODUCERS
PUBLICATION_STEM = re.compile(r"^\d+$")  # tên file <publication_id>.pdf


def publication_id_of(path: str | Path) -> str | None:
    """Mã bài VISTA từ tên file PDF (<publication_id>.pdf); None với file không đặt tên theo mã bài (vd unknown/<sha256>.pdf)."""
    stem = Path(path).stem
    return stem if PUBLICATION_STEM.match(stem) else None


def path_meta(path: Path) -> dict:
    """Metadata suy ra từ đường dẫn: năm (thư mục cha, None với unknown/) và mã bài (từ tên file)."""
    year = path.parent.name
    return {"year_dir": int(year) if YEAR_DIR.match(year) else None, "publication_id": publication_id_of(path)}


VISTA_LAYOUT = PaperLayout(key="vista", label="VISTA", text_dir=TEXT_DIR, claim_dir=CLAIM_DIR, out_path=OUT_PATH, metadata_rel=METADATA_REL,
                           font_groups_rel=FONT_GROUPS_REL, path_meta=path_meta, ocr_all_producers=OCR_ALL_PRODUCERS)
