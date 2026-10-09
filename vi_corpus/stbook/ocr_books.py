"""
Nguồn sách stbook.vn (NXB Chính trị quốc gia Sự thật), dữ liệu do repo stbook-crawler tải về.

Cấu trúc thư mục gốc do stbook-crawler tạo (chép nguyên thư mục data/ của nó vào
<data-root>/raw/stbook/, không sửa gì):

    raw/stbook/<danh-mục>/books.json            # metadata mọi sách của danh mục
    raw/stbook/<danh-mục>/content/<product_id>.pdf   # PDF dạng ảnh, chỉ sách miễn phí
    raw/stbook/<danh-mục>/content/<product_id>_pages/ # sách đang tải dở, bỏ qua

Vì PDF không có lớp chữ, xử lý qua 2 bước:
1. run_ocr: OCR từng cuốn, ghi interim/stbook_ocr/<danh-mục>/<product_id>.json (text từng
   trang + metadata sách). Checkpoint theo cuốn: file đã có và khớp kích thước PDF thì bỏ qua.
2. iter_records: đọc kết quả OCR, sinh bản ghi theo schema chung (mỗi cuốn một bản ghi).
"""

import json
import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from vi_corpus.common.pdf_text import clean_pages
from vi_corpus.common.state import read_json, write_json_atomic
from vi_corpus.common.schema import make_doc_id, text_sha256
from vi_corpus.common.registry import SourceSpec

logger = logging.getLogger("vi_corpus")

OCR_DIR = "interim/stbook_ocr"


def find_books(stbook_root: Path) -> Iterator[tuple[str, dict, Path]]:
    """
    Sinh (slug danh mục, metadata sách, đường dẫn PDF) cho mọi sách đã có PDF hoàn chỉnh.

    Sách chưa tải PDF (có giá, hoặc đang tải dở) bị bỏ qua. books.json hỏng thì ghi cảnh báo
    và bỏ qua cả danh mục đó.
    """
    for books_json in sorted(stbook_root.glob("*/books.json")):
        payload = read_json(books_json)
        if payload is None:
            logger.warning("Bỏ qua %s: không đọc được", books_json)
            continue
        for book in payload.get("books", []):
            pdf = books_json.parent / "content" / f"{book['product_id']}.pdf"
            if pdf.exists():
                yield books_json.parent.name, book, pdf


def _ocr_path(data_root: Path, slug: str, product_id: str) -> Path:
    """Đường dẫn file kết quả OCR của một cuốn."""
    return data_root / OCR_DIR / slug / f"{product_id}.json"


def ocr_status(stbook_root: Path, data_root: Path) -> tuple[int, int]:
    """Trả về (số cuốn đã OCR xong, tổng số cuốn có PDF)."""
    books = list(find_books(stbook_root))
    done = sum(_ocr_path(data_root, slug, b["product_id"]).exists() for slug, b, _ in books)
    return done, len(books)


def books_to_ocr(stbook_root: Path, data_root: Path, limit_books: int | None = None) -> list[tuple[str, dict, Path]]:
    """
    Các cuốn cần OCR: chưa có kết quả, hoặc kích thước PDF đã đổi so với lúc OCR (PDF tải lại thì OCR lại).

    Returns:
        Danh sách (slug danh mục, metadata sách, đường dẫn PDF), tối đa limit_books cuốn.
    """
    todo = []
    for slug, book, pdf in find_books(stbook_root):
        done = read_json(_ocr_path(data_root, slug, book["product_id"]))
        if not (done and done.get("pdf_size") == pdf.stat().st_size):
            todo.append((slug, book, pdf))
    return todo[:limit_books]


def ocr_one_book(ocr, slug: str, book: dict, pdf: Path, data_root: Path) -> None:
    """
    OCR một cuốn bằng bộ OCR `ocr` (vi_corpus.common.ocr.PageOcr) và ghi kết quả nguyên tử ngay khi xong.

    Dùng chung cho chạy tuần tự (run_ocr) và chạy song song nhiều GPU (vi_corpus.stbook.ocr_parallel).

    Raises:
        ValueError: PDF hỏng / bị cắt cụt (xem pdf_page_images). Các lỗi OCR khác cũng được ném lên cho nơi gọi xử lý.
    """
    from vi_corpus.common.ocr import pdf_page_images

    pages = [ocr.page_text(img) for img in pdf_page_images(pdf)]
    write_json_atomic(_ocr_path(data_root, slug, book["product_id"]), {
        "category_slug": slug,
        "book": book,
        "pdf_path": str(pdf.relative_to(data_root)) if pdf.is_relative_to(data_root) else str(pdf),
        "pdf_size": pdf.stat().st_size,
        "ocr": {"det": "PP-OCRv5_mobile_det", "rec": "vietocr/vgg_transformer"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
    })


def run_ocr(stbook_root: Path, data_root: Path, device: str, limit_books: int | None = None) -> int:
    """
    OCR mọi sách chưa làm trên MỘT thiết bị, ghi kết quả từng cuốn ngay khi xong (chạy lại để làm tiếp).

    Lỗi ở một cuốn (PDF hỏng do crawler bị kill giữa lúc ghi, ...) chỉ ghi log rồi làm cuốn khác.
    Muốn dùng nhiều GPU thì dùng vi_corpus.stbook.ocr_parallel.run_ocr_parallel (cùng checkpoint, cùng kết quả).
    ponytail: checkpoint theo cuốn, crash mất tối đa 1 cuốn (~15 phút với sách 900 trang);
    chuyển sang checkpoint theo trang nếu thấy lãng phí.

    Returns:
        Số cuốn bị lỗi.
    """
    from vi_corpus.common.ocr import PageOcr

    todo = books_to_ocr(stbook_root, data_root, limit_books)
    logger.info("Cần OCR %d cuốn", len(todo))
    if not todo:
        return 0

    ocr = PageOcr(device)
    failed = 0
    for slug, book, pdf in tqdm(todo, desc="OCR stbook", unit="cuốn"):
        try:
            ocr_one_book(ocr, slug, book, pdf, data_root)
        except Exception as exc:  # noqa: BLE001 — một cuốn lỗi không được dừng cả lượt
            failed += 1
            logger.warning("Lỗi OCR %s: %s", pdf, exc)
    logger.info("Xong OCR: %d cuốn thành công, %d lỗi", len(todo) - failed, failed)
    return failed


def iter_records(spec: SourceSpec, data_root: Path, limit_files: int | None = None,
                 clean: bool = False, with_meta: bool = False) -> Iterator[dict]:
    """
    Sinh bản ghi schema chung (vi_corpus.common.schema.CORPUS_SCHEMA) từ kết quả OCR, mỗi cuốn một bản ghi.

    Mặc định text là các trang nối bằng dòng trống, giữ nguyên kết quả OCR. Với clean=True thì làm sạch theo
    trang bằng vi_corpus.common.pdf_text.clean_pages (bỏ tiêu đề/chân trang chạy, số trang lẻ, ghép dòng thành đoạn).
    source_path trỏ về PDF gốc. Chỉ đọc sách đã OCR xong; chạy run_ocr trước.

    Args:
        spec:        Nguồn stbook trong registry.
        data_root:   Thư mục gốc dữ liệu.
        limit_files: Chỉ đọc N cuốn đầu (chạy thử).
        clean:       Làm sạch theo trang (xem trên).
        with_meta:   Thêm khoá "meta" (JSON: tên sách, danh mục, số trang, số dòng tiêu đề / số trang bị clean_pages bỏ).
    """
    for path in sorted((data_root / OCR_DIR).glob("*/*.json"))[:limit_files]:
        done = read_json(path)
        if done is None:
            continue
        page_stats: dict = {}
        text = clean_pages(done["pages"], stats=page_stats) if clean else "\n\n".join(done["pages"])
        rec = {
            "doc_id": make_doc_id(spec.key, done["pdf_path"], 0),
            "source_key": spec.key,
            "source_path": done["pdf_path"],
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
            book = done.get("book") or {}
            rec["meta"] = json.dumps({"title": book.get("title"), "category": done.get("category_slug"),
                                      "pages": len(done["pages"]),
                                      "page_lines_removed": page_stats.get("lines_removed", 0)}, ensure_ascii=False)
        yield rec
