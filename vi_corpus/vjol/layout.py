"""
Bố cục dữ liệu của một nguồn bài báo khoa học dạng PDF (VJOL, VISTA): các đường dẫn và quy tắc riêng của nguồn.

Bộ xử lý bài báo (vi_corpus.vjol: trích text 2 lượt, dò chữ hỏng, OCR bằng chuỗi VLM, làm sạch, sinh bản ghi) dùng chung cho VJOL và VISTA (VS-02). Chỉ những gì
khác nhau giữa hai nguồn nằm ở PaperLayout: thư mục checkpoint, tên file PDF, nơi để metadata, bộ lọc producer OCR cũ. Mọi nguồn dùng chung tập âm tiết,
danh sách từ không dấu và bộ nhớ đệm OCR (khoá theo sha256 nội dung nên không lẫn nguồn).

Module này không import module nào khác của vi_corpus.vjol để cả extract.py lẫn records.py dùng được mà không vòng import.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperLayout:
    """
    Mô tả một nguồn bài báo PDF cho bộ xử lý dùng chung.

    Attributes:
        key:         Mã nguồn trong registry ("vjol", "vista"); cũng là tiền tố nhãn log và tên tiến trình.
        label:       Tên hiển thị trong log ("VJOL", "VISTA").
        text_dir:    Thư mục checkpoint trích text, tương đối so với data_root (vd "interim/vjol_text").
        claim_dir:   Thư mục khoá theo file cho nhiều tiến trình (mỗi GPU một tiến trình), tương đối so với data_root.
        out_path:    File Parquet xuất ra để xem, tương đối so với data_root.
        metadata_rel: File metadata chính, tương đối so với thư mục raw của nguồn.
        font_groups_rel: File nhóm font xấu của nguồn, tương đối so với data_root (mỗi nguồn học riêng, J-42 (4)).
        path_meta:   Hàm đường dẫn PDF -> dict metadata suy ra từ đường dẫn (trộn vào checkpoint).
        ocr_all_producers: Các cặp (đầu của producer, chuỗi con của creator) mà PDF khớp thì mọi trang có chữ đều đi OCR (J-29 D2).
    """

    key: str
    label: str
    text_dir: str
    claim_dir: str
    out_path: str
    metadata_rel: str
    font_groups_rel: str
    path_meta: Callable[[Path], dict]
    ocr_all_producers: tuple[tuple[str, str], ...] = ()
