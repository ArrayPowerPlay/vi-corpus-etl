"""
Xử lý thô giáo trình (Google Drive "Tổng hợp giáo trình Đại học"): kiểm kê file và trích text từng trang.

Đọc <data-root>/raw/giao_trinh/<ngành>/<môn>/<file> (bản rclone, không sửa), ghi
interim/giao_trinh_text/<sha256>.json cho mỗi file duy nhất (định dạng checkpoint và vòng lặp chạy lại
nằm ở vi_corpus.common.batch_extract, dùng chung với VJOL):

    {"sha256", "size", "paths": [đường dẫn, cùng nội dung], "nganh", "mon", "format",
     "pages": [{"text", "method"}], "error", "skipped", "parser_version", "created_at"}

Phần riêng của giáo trình ở file này (chiến lược ở docs/sources/GIAO_TRINH.md, mục 2-3):
- extract_file: nhận định dạng theo nội dung đầu file và trích text bằng vi_corpus.common.parsers (D-05): PDF (trang
  scan thì OCR, mô hình nạp lười), docx, pptx, doc / ppt (LibreOffice), djvu (djvulibre), html, epub; định dạng khác
  ghi skipped="unsupported_format", thiếu công cụ ngoài ghi skipped="needs_..." (chạy lại với --retry-skipped).
- ngành = thư mục cấp 1, môn = thư mục cấp 2 dưới thư mục giáo trình.
- OCR trang scan bằng chuỗi engine VLM (vi_corpus.common.ocr_cascade.CascadeOcrGetter, O-20) như stbook / VJOL; trang đã OCR được nhớ ở
  interim/giao_trinh_ocr_cache/ (vi_corpus.common.ocr_cache) nên một cuốn scan dài bị ngắt giữa chừng không phải OCR lại từ đầu. Nhiều tiến trình
  (mỗi GPU một tiến trình) chia file bằng khoá flock ở state/giao_trinh_claims/. Chuỗi OCR hỏng (ServerDown, EngineUnavailable) dừng cả lượt.
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path

from vi_corpus.common.batch_extract import (  # noqa: F401 — inventory, file_sha256, make_ocr_getter: tên cũ vẫn import được từ đây
    file_sha256,
    inventory,
    make_ocr_getter,
    run_batch,
)
from vi_corpus.common.batch_extract import checkpoint_path as _checkpoint_path
from vi_corpus.common.batch_extract import status as _status
from vi_corpus.common.ocr_cache import OcrCache
from vi_corpus.common.parsers import detect_format, parse_file

TEXT_DIR = "interim/giao_trinh_text"
OCR_CACHE_DIR = "interim/giao_trinh_ocr_cache"
CLAIM_DIR = "state/giao_trinh_claims"


def checkpoint_path(data_root: Path, sha256: str) -> Path:
    """Đường dẫn checkpoint (file kết quả trích text) của một file giáo trình."""
    return _checkpoint_path(data_root, TEXT_DIR, sha256)


def extract_file(path: Path, get_ocr: Callable[[], object | None], cache_root: Path | None = None) -> dict:
    """
    Trích text một file theo định dạng nhận từ nội dung đầu file (vi_corpus.common.parsers.parse_file); không ném lỗi của file.

    Args:
        path:       File giáo trình.
        get_ocr:    Hàm lấy bộ OCR (xem make_ocr_getter / CascadeOcrGetter).
        cache_root: Thư mục gốc dữ liệu để đặt bộ nhớ đệm OCR theo trang (OCR_CACHE_DIR); chỉ dùng cho PDF và khi `get_ocr` có `signature()`
                    (bộ OCR chuỗi engine). None = không dùng cache.

    Returns:
        {"format", "pages", "error", "skipped", "parser_version"}: định dạng không hỗ trợ thì skipped="unsupported_format",
        thiếu LibreOffice / djvulibre thì skipped="needs_libreoffice" / "needs_djvulibre"; file hỏng / có mật khẩu
        thì error là thông báo lỗi.

    Raises:
        ServerDown, EngineUnavailable: chuỗi OCR hỏng (lỗi hệ thống, không phải lỗi của file) — dừng cả lượt.
    """
    signature = getattr(get_ocr, "signature", None)
    cache = None
    if cache_root is not None and callable(signature) and detect_format(path) == "pdf":
        cache = OcrCache.for_pdf(cache_root, file_sha256(path), signature(), OCR_CACHE_DIR)
    return parse_file(path, get_ocr, ocr_cache=cache)


def run_extract(root: Path, data_root: Path, get_ocr: Callable[[], object | None],
                ocr_enabled: bool = True, limit_files: int | None = None, retry_skipped: bool = False,
                multi_process: bool = False) -> int:
    """
    Kiểm kê rồi trích text mọi file giáo trình chưa làm (xem vi_corpus.common.batch_extract.run_batch).
    retry_skipped=True thì làm lại cả file từng bị bỏ qua (vd sau khi cài LibreOffice / djvulibre).
    multi_process=True khi nhiều tiến trình (mỗi GPU một tiến trình) chạy cùng lúc: các file được chia nhau bằng khoá flock (CLAIM_DIR).

    Ngành = thư mục cấp 1, môn = thư mục cấp 2 dưới `root`.

    Returns:
        Số file bị lỗi trong lần chạy này.
    """
    def path_meta(path: Path) -> dict:
        """Ngành và môn từ đường dẫn tương đối so với `root`."""
        parts = path.relative_to(root).parts
        return {"nganh": parts[0] if len(parts) > 1 else "", "mon": parts[1] if len(parts) > 2 else ""}

    return run_batch(root, data_root, TEXT_DIR, partial(extract_file, cache_root=data_root), path_meta, get_ocr,
                     ocr_enabled=ocr_enabled, limit_files=limit_files, desc="Trích text giáo trình",
                     retry_skipped=retry_skipped, claim_dir=CLAIM_DIR if multi_process else None)


def status(root: Path, data_root: Path) -> dict:
    """Tóm tắt tiến độ giáo trình (xem vi_corpus.common.batch_extract.status)."""
    return _status(root, data_root, TEXT_DIR)
