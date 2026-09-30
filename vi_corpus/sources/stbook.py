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

import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from vi_corpus.download.checkpoint import read_json, write_json_atomic
from vi_corpus.schema import make_doc_id, text_sha256
from vi_corpus.sources.registry import SourceSpec

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


def run_ocr(stbook_root: Path, data_root: Path, device: str, limit_books: int | None = None) -> int:
    """
    OCR mọi sách chưa làm, ghi kết quả từng cuốn ngay khi xong (chạy lại để làm tiếp).

    Cuốn đã có kết quả được bỏ qua nếu kích thước PDF không đổi (PDF tải lại thì OCR lại).
    Lỗi ở một cuốn (PDF hỏng do crawler bị kill giữa lúc ghi, ...) chỉ ghi log rồi làm cuốn khác.
    ponytail: checkpoint theo cuốn, crash mất tối đa 1 cuốn (~15 phút với sách 900 trang);
    chuyển sang checkpoint theo trang nếu thấy lãng phí.

    Returns:
        Số cuốn bị lỗi.
    """
    from vi_corpus.ocr import PageOcr, pdf_page_images

    todo = []
    for slug, book, pdf in find_books(stbook_root):
        done = read_json(_ocr_path(data_root, slug, book["product_id"]))
        if not (done and done.get("pdf_size") == pdf.stat().st_size):
            todo.append((slug, book, pdf))
    todo = todo[:limit_books]
    logger.info("Cần OCR %d cuốn", len(todo))
    if not todo:
        return 0

    ocr = PageOcr(device)
    failed = 0
    for slug, book, pdf in tqdm(todo, desc="OCR stbook", unit="cuốn"):
        try:
            pages = [ocr.page_text(img) for img in pdf_page_images(pdf)]
        except Exception as exc:  # noqa: BLE001 — một cuốn lỗi không được dừng cả lượt
            failed += 1
            logger.warning("Lỗi OCR %s: %s", pdf, exc)
            continue
        write_json_atomic(_ocr_path(data_root, slug, book["product_id"]), {
            "category_slug": slug,
            "book": book,
            "pdf_path": str(pdf.relative_to(data_root)) if pdf.is_relative_to(data_root) else str(pdf),
            "pdf_size": pdf.stat().st_size,
            "ocr": {"det": "PP-OCRv5_mobile_det", "rec": "vietocr/vgg_transformer"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pages": pages,
        })
    logger.info("Xong OCR: %d cuốn thành công, %d lỗi", len(todo) - failed, failed)
    return failed


def iter_records(spec: SourceSpec, data_root: Path, limit_files: int | None = None) -> Iterator[dict]:
    """
    Sinh bản ghi schema chung (vi_corpus.schema.CORPUS_SCHEMA) từ kết quả OCR, mỗi cuốn một bản ghi.

    Text là các trang nối bằng dòng trống, giữ nguyên kết quả OCR (làm sạch ở stage normalize).
    source_path trỏ về PDF gốc. Chỉ đọc sách đã OCR xong; chạy run_ocr trước.

    Args:
        spec:        Nguồn stbook trong registry.
        data_root:   Thư mục gốc dữ liệu.
        limit_files: Chỉ đọc N cuốn đầu (chạy thử).
    """
    for path in sorted((data_root / OCR_DIR).glob("*/*.json"))[:limit_files]:
        done = read_json(path)
        if done is None:
            continue
        text = "\n\n".join(done["pages"])
        yield {
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
