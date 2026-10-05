"""
OCR sách stbook song song trên NHIỀU GPU bằng NeMo Curator + Ray (cùng khung với vi_corpus.pipeline.curator).

Cùng checkpoint và cùng định dạng kết quả với run_ocr (interim/stbook_ocr/<slug>/<product_id>.json, ghi nguyên tử từng cuốn),
nên có thể chạy tuần tự rồi chuyển sang song song (hoặc ngược lại) giữa chừng. Mỗi actor Ray giữ một bộ PageOcr trên một GPU
(nạp trong setup, một lần), nhận từng cuốn một: 4 GPU = 4 cuốn OCR cùng lúc. Cuốn lỗi chỉ được ghi log, không dừng cả lượt.
Cần `uv sync --group ocr --group curator`.
"""

import logging
from pathlib import Path

from nemo_curator.stages.resources import Resources

from vi_corpus.pipeline.curator import CuratorBackend
from vi_corpus.stbook.ocr_books import books_to_ocr, ocr_one_book

logger = logging.getLogger("vi_corpus")


class _BookOcr:
    """Hàm `rows -> rows` của một actor: nạp PageOcr một lần rồi OCR từng cuốn, trả trạng thái từng cuốn."""

    def __init__(self, data_root: str, device: str) -> None:
        """Nạp bộ OCR (trên GPU được Ray cấp cho actor nếu device = "cuda")."""
        from vi_corpus.common.ocr import PageOcr

        self.data_root = Path(data_root)
        self.ocr = PageOcr(device)

    def __call__(self, rows: list[dict]) -> list[dict]:
        """OCR các cuốn trong rows; mỗi dòng trả về thêm khoá "error" (None nếu thành công)."""
        out = []
        for r in rows:
            try:
                ocr_one_book(self.ocr, r["slug"], r["book"], Path(r["pdf"]), self.data_root)
                out.append({**r, "error": None})
            except Exception as exc:  # noqa: BLE001 — một cuốn lỗi không được dừng cả lượt
                logger.warning("Lỗi OCR %s: %s", r["pdf"], exc)
                out.append({**r, "error": str(exc)})
        return out


def _make_book_ocr(data_root: str, device: str) -> _BookOcr:
    """Factory cấp module (pickle được) cho RowsStage."""
    return _BookOcr(data_root, device)


def run_ocr_parallel(stbook_root: Path, data_root: Path, backend: CuratorBackend, device: str = "cuda",
                     limit_books: int | None = None) -> int:
    """
    OCR mọi sách chưa làm, mỗi GPU một cuốn cùng lúc (backend.gpus_per_worker < 1 thì nhiều actor / GPU).

    Args:
        stbook_root: Thư mục data/ của stbook-crawler.
        data_root:   Thư mục gốc dữ liệu (kết quả ở interim/stbook_ocr/).
        backend:     CuratorBackend đã vào context (Ray đã chạy).
        device:      "cuda" hoặc "cpu" (cpu: mỗi actor chiếm CPU, không cần GPU).
        limit_books: Chỉ OCR N cuốn (chạy thử).

    Returns:
        Số cuốn bị lỗi.
    """
    import functools

    todo = books_to_ocr(stbook_root, data_root, limit_books)
    logger.info("Cần OCR %d cuốn (song song, executor=%s)", len(todo), backend.executor_name)
    if not todo:
        return 0
    rows = [{"doc_id": f"{slug}/{book['product_id']}", "slug": slug, "book": book, "pdf": str(pdf)} for slug, book, pdf in todo]
    res = Resources(cpus=2.0, gpus=backend.gpus_per_worker if device == "cuda" else 0.0)
    old = backend.rows_per_task
    backend.rows_per_task = 1  # mỗi phân vùng một cuốn: cân tải tốt nhất vì thời gian mỗi cuốn rất khác nhau
    try:
        out = backend.run("ocr_stbook", rows, functools.partial(_make_book_ocr, str(data_root), device), res)
    finally:
        backend.rows_per_task = old
    failed = sum(1 for r in out if r["error"])
    logger.info("Xong OCR: %d cuốn thành công, %d lỗi", len(out) - failed, failed)
    return failed
