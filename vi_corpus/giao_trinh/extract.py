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

import logging
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
from vi_corpus.common.garbled import DEFAULT_THRESHOLDS, Thresholds, load_unaccented, load_vocab, unaccented_tag
from vi_corpus.common.ocr_cache import OcrCache
from vi_corpus.common.parsers import detect_format, parse_file

logger = logging.getLogger("vi_corpus")

TEXT_DIR = "interim/giao_trinh_text"
OCR_CACHE_DIR = "interim/giao_trinh_ocr_cache"
CLAIM_DIR = "state/giao_trinh_claims"
# Phiên bản cách trích text giáo trình, ghi vào checkpoint (extract_version); đổi số này thì lần chạy sau làm lại mọi checkpoint cũ hơn.
# Checkpoint không có trường này = trước GT-06 (chưa dò chữ hỏng do font). g1 = GT-06: PDF có trang lỗi font (lớp chữ sai bảng mã, bộ dò chung với
# VJOL ở vi_corpus.common.garbled) cũng đi OCR lại thay vì thành văn bản rác bị stage quality loại.
# g2 = GT-07: ngưỡng bộ dò riêng của giáo trình (GIAO_TRINH_THRESHOLDS), vì ngưỡng của VJOL gắn cờ nhầm ~60% trang đúng của giáo trình.
EXTRACT_VERSION = "g2"
VOCAB_TAG = "+vocab"  # hậu tố extract_version khi bộ dò có tín hiệu âm tiết lạ (họ sai dấu kiểu Windows-1258)
UNACC_TAG = "+unacc"  # hậu tố (kèm băm nội dung) khi bộ dò dùng danh sách từ không dấu sinh từ dữ liệu
SYLLABLES_REL = "processed/ocr_bakeoff/syllables.json"
UNACCENTED_REL = "processed/ocr_bakeoff/unaccented.json"


# GT-07: ngưỡng bộ dò chữ hỏng cho giáo trình, chỉnh bằng lượt kiểm chứng OCR 750 trang (GT-06, processed/giao_trinh_verify_flagged/): với ngưỡng của VJOL,
# trong các trang bị gắn cờ do tín hiệu `case` thì ~58% OCR khớp lớp chữ >= 0,8 (gắn nhầm; sách tiếng Anh, mã nguồn, ligature) còn 11% hỏng thật; tín hiệu `unknown`
# đứng một mình thì 90% gắn nhầm, 0% hỏng thật (sách kỹ thuật tiếng Anh có 30-50% từ ngoài tập âm tiết). Ngưỡng mới là điểm khởi đầu từ 448 trang gắn cờ, chưa kiểm chứng
# trên mẫu độc lập (chia đôi theo file cho kết quả chọn ngưỡng khác nhau ở `case` / `nodiac`):
#   tone 0,05 -> 0,10 (trang đúng median 0,076, hỏng median 0,34); case 0,05 -> 0,30 (đúng median 0,094); foreign 0,003 -> 0,04 (đúng median 0,022, hỏng 0,096);
#   unknown 0,30 -> 0,50 (đúng max 0,50, hỏng median 0,69); nodiac 0,02 -> 0,03 (không phân biệt được đúng / hỏng, chỉ hạ số báo nhầm);
#   script (mới) 0,01: chữ Kirin / Armenia do font cũ ánh xạ sai, 12 trang hỏng thật, 1 trang đúng, 0 trong 300 trang đối chứng.
# `control` và `hyphen` giữ nguyên (control: 85% hỏng thật).
GIAO_TRINH_THRESHOLDS = DEFAULT_THRESHOLDS._replace(tone=0.10, case=0.30, foreign=0.04, unknown=0.50, nodiac=0.03, script=0.01)


def version_for(vocab: frozenset[str] | None, unaccented: frozenset[str] | None = None) -> str:
    """extract_version ghi vào checkpoint: EXTRACT_VERSION, thêm "+unacc<băm>" nếu có danh sách từ không dấu, rồi "+vocab" nếu bộ dò có tập âm tiết."""
    tag = unaccented_tag(unaccented)
    return EXTRACT_VERSION + (f"{UNACC_TAG}{tag}" if tag else "") + (VOCAB_TAG if vocab is not None else "")


def accepted_versions(vocab: frozenset[str] | None, unaccented: frozenset[str] | None = None) -> frozenset[str]:
    """
    Các extract_version mà lần chạy này coi là đã xong: có tập âm tiết thì chỉ "+vocab" (checkpoint dò thiếu tín hiệu bị làm lại); không có thì
    chấp nhận cả bản có và không có "+vocab" (không bắt làm lại chỉ vì lần này thiếu file).
    """
    return frozenset({version_for(vocab, unaccented)} | ({version_for(frozenset(), unaccented)} if vocab is None else set()))


def checkpoint_path(data_root: Path, sha256: str) -> Path:
    """Đường dẫn checkpoint (file kết quả trích text) của một file giáo trình."""
    return _checkpoint_path(data_root, TEXT_DIR, sha256)


def extract_file(path: Path, get_ocr: Callable[[], object | None], cache_root: Path | None = None,
                 vocab: frozenset[str] | None = None, th: Thresholds = GIAO_TRINH_THRESHOLDS,
                 unaccented: frozenset[str] | None = None) -> dict:
    """
    Trích text một file theo định dạng nhận từ nội dung đầu file (vi_corpus.common.parsers.parse_file); không ném lỗi của file.

    Args:
        path:       File giáo trình.
        get_ocr:    Hàm lấy bộ OCR (xem make_ocr_getter / CascadeOcrGetter).
        cache_root: Thư mục gốc dữ liệu để đặt bộ nhớ đệm OCR theo trang (OCR_CACHE_DIR); chỉ dùng cho PDF và khi `get_ocr` có `signature()`
                    (bộ OCR chuỗi engine). None = không dùng cache.
        vocab:      Tập âm tiết cho tín hiệu "unknown" của bộ dò chữ hỏng do font (None = bỏ tín hiệu này; checkpoint ghi extract_version không có "+vocab").
        th:         Ngưỡng của bộ dò chữ hỏng.
        unaccented: Danh sách từ không dấu cho tín hiệu "nodiac" (None = danh sách tối thiểu trong vi_corpus.common.garbled).

    Returns:
        {"format", "pages", "error", "skipped", "parser_version", "extract_version"}: định dạng không hỗ trợ thì skipped="unsupported_format",
        thiếu LibreOffice / djvulibre thì skipped="needs_libreoffice" / "needs_djvulibre"; file hỏng / có mật khẩu
        thì error là thông báo lỗi.

    Raises:
        ServerDown, EngineUnavailable: chuỗi OCR hỏng (lỗi hệ thống, không phải lỗi của file) — dừng cả lượt.
    """
    signature = getattr(get_ocr, "signature", None)
    cache = None
    if cache_root is not None and callable(signature) and detect_format(path) == "pdf":
        cache = OcrCache.for_pdf(cache_root, file_sha256(path), signature(), OCR_CACHE_DIR)
    pdf_kwargs = {"detect_garbled": True, "garbled_vocab": vocab, "garbled_th": th, "garbled_unaccented": unaccented}
    out = parse_file(path, get_ocr, ocr_cache=cache, pdf_kwargs=pdf_kwargs)
    out["extract_version"] = version_for(vocab, unaccented)
    return out


def run_extract(root: Path, data_root: Path, get_ocr: Callable[[], object | None],
                ocr_enabled: bool = True, limit_files: int | None = None, retry_skipped: bool = False,
                multi_process: bool = False, syllables: Path | None = None, unaccented_path: Path | None = None,
                th: Thresholds = GIAO_TRINH_THRESHOLDS) -> int:
    """
    Kiểm kê rồi trích text mọi file giáo trình chưa làm (xem vi_corpus.common.batch_extract.run_batch).
    retry_skipped=True thì làm lại cả file từng bị bỏ qua (vd sau khi cài LibreOffice / djvulibre).
    syllables / unaccented_path: file tập âm tiết và danh sách từ không dấu của bộ dò chữ hỏng (mặc định <data_root>/processed/ocr_bakeoff/{syllables,unaccented}.json,
    như VJOL); thiếu file thì bỏ tín hiệu tương ứng (ghi cảnh báo). Checkpoint cũ (không có extract_version khớp) bị làm lại.
    multi_process=True khi nhiều tiến trình (mỗi GPU một tiến trình) chạy cùng lúc: các file được chia nhau bằng khoá flock (CLAIM_DIR).

    Ngành = thư mục cấp 1, môn = thư mục cấp 2 dưới `root`.

    Returns:
        Số file bị lỗi trong lần chạy này.
    """
    vocab = load_vocab(syllables or data_root / SYLLABLES_REL)
    if vocab is None:
        logger.warning("Không thấy tập âm tiết %s: bộ dò chữ hỏng bỏ tín hiệu âm tiết lạ (họ chữ sai dấu kiểu Windows-1258 có thể lọt)",
                       syllables or data_root / SYLLABLES_REL)
    unaccented = load_unaccented(unaccented_path or data_root / UNACCENTED_REL)
    if unaccented is None:
        logger.warning("Không thấy danh sách từ không dấu %s: dùng danh sách tối thiểu", unaccented_path or data_root / UNACCENTED_REL)

    def path_meta(path: Path) -> dict:
        """Ngành và môn từ đường dẫn tương đối so với `root`."""
        parts = path.relative_to(root).parts
        return {"nganh": parts[0] if len(parts) > 1 else "", "mon": parts[1] if len(parts) > 2 else ""}

    return run_batch(root, data_root, TEXT_DIR, partial(extract_file, cache_root=data_root, vocab=vocab, th=th, unaccented=unaccented), path_meta, get_ocr,
                     ocr_enabled=ocr_enabled, limit_files=limit_files, desc="Trích text giáo trình",
                     retry_skipped=retry_skipped, claim_dir=CLAIM_DIR if multi_process else None,
                     extract_versions=accepted_versions(vocab, unaccented))


def status(root: Path, data_root: Path) -> dict:
    """Tóm tắt tiến độ giáo trình (xem vi_corpus.common.batch_extract.status)."""
    return _status(root, data_root, TEXT_DIR)
