"""Điểm vào (entrypoint) chính của crawler stbook.vn.

Code chép nguyên từ repo stbook-crawler (gói `stbook_crawler`), chỉ đổi thư mục ghi mặc định.

Cách dùng cơ bản (crawl toàn bộ site, chỉ lấy metadata):

    uv run python scripts/stbook/crawl.py

Các tuỳ chọn thường dùng:

    --category <slug>        chỉ crawl 1 danh mục (có thể lặp lại nhiều lần)
    --skip-detail             không gọi trang chi tiết (nhanh hơn, ít dữ liệu hơn;
                               không dùng được cùng --download-pdf)
    --download-covers         tải ảnh bìa về data/<slug>/covers/
    --download-pdf              tải TOÀN BỘ nội dung các sách "Miễn phí" đọc được
                                 online, ghép thành file PDF tại data/<slug>/content/<id>.pdf
    --max-pages N                giới hạn số trang tải mỗi sách khi --download-pdf
                                 (mặc định: không giới hạn, tải hết sách)
    --workers N                   số request tải nội dung sách chạy song song
                                 (mặc định 8 — đo thực tế cho thấy server không
                                 phản hồi nhanh hơn dù mở nhiều kết nối hơn)
    --keep-page-images           giữ lại ảnh từng trang sau khi đã ghép PDF (mặc định xoá)
    --delay SECONDS               độ trễ giữa các request lấy metadata (mặc định 0.8s)
    --out DIR                     thư mục ghi dữ liệu (mặc định <SEA_DATA_ROOT hoặc ./data>/raw/stbook)

VỀ RESUME: nếu tiến trình bị ngắt giữa chừng (mất mạng, Ctrl+C, lỗi
ngoài dự kiến, terminal bị đóng...), chỉ cần chạy lại ĐÚNG lệnh cũ.
Crawler tự đọc lại `books.json` đã ghi của từng danh mục để biết sách
nào đã có đủ dữ liệu, bỏ qua việc gọi lại trang chi tiết cho các sách
đó; ảnh bìa và PDF đã tải cũng tự động được bỏ qua (kiểm tra file đã
tồn tại trên đĩa). `books.json` được ghi lại ngay sau mỗi sách xử lý
xong, nên nhiều nhất chỉ mất tiến trình của 1 cuốn đang xử lý dở khi bị
ngắt. Xem thêm log chi tiết tại `<out>/crawl.log`.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from .categories import CATEGORIES, Category
from .client import RateLimitedSession, build_bulk_session
from .content import DEFAULT_WORKERS, assemble_pdf, download_book_pages
from .parse_category import BookListItem, fetch_category_books
from .parse_detail import BookDetail, fetch_book_detail
from .storage import book_record, category_dir_for, load_category_books, write_all_books, write_category_books

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stbook_crawler")

_PAGE_COUNT_RE = re.compile(r"\d+")


def _add_file_logging(out_dir: Path) -> None:
    """Ghi thêm log ra file `<out_dir>/crawl.log`, bên cạnh log ra console.

    Log console có thể mất khi đóng terminal hoặc rớt kết nối SSH; ghi
    thêm ra file giúp biết chính xác crawl đã dừng ở danh mục/sách nào
    khi tiến trình bị ngắt đột ngột.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "crawl.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl sách trên stbook.vn theo danh mục.")
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        metavar="SLUG",
        help="Chỉ crawl danh mục có slug này (lặp lại cờ để chọn nhiều danh mục).",
    )
    parser.add_argument(
        "--skip-detail",
        action="store_true",
        help="Bỏ qua việc gọi trang chi tiết từng sách (nhanh hơn, thiếu mô tả/tác giả đầy đủ).",
    )
    parser.add_argument(
        "--download-covers",
        action="store_true",
        help="Tải ảnh bìa sách về data/<slug>/covers/.",
    )
    parser.add_argument(
        "--download-pdf",
        action="store_true",
        help="Tải toàn bộ nội dung các sách 'Miễn phí' đọc được online, ghép thành PDF.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Giới hạn số trang tải mỗi sách khi dùng --download-pdf (mặc định: không giới hạn).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Số request tải nội dung sách chạy song song (mặc định {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--keep-page-images",
        action="store_true",
        help="Giữ lại ảnh JPEG từng trang sau khi đã ghép thành PDF (mặc định xoá để tiết kiệm ổ đĩa).",
    )
    parser.add_argument("--delay", type=float, default=0.8, help="Độ trễ (giây) giữa các request lấy metadata.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.getenv("SEA_DATA_ROOT", "data")) / "raw" / "stbook",
        help="Thư mục ghi dữ liệu (mặc định <SEA_DATA_ROOT hoặc ./data>/raw/stbook).",
    )
    args = parser.parse_args(argv)

    if args.download_pdf and args.skip_detail:
        parser.error(
            "--download-pdf cần trang chi tiết để biết số trang mỗi sách, "
            "không dùng được cùng --skip-detail."
        )

    return args


def select_categories(slugs: list[str] | None) -> list[Category]:
    if not slugs:
        return CATEGORIES
    wanted = set(slugs)
    selected = [c for c in CATEGORIES if c.slug in wanted]
    missing = wanted - {c.slug for c in selected}
    if missing:
        log.warning("Không tìm thấy danh mục với slug: %s", ", ".join(sorted(missing)))
    return selected


def crawl(args: argparse.Namespace) -> None:
    """Crawl toàn bộ danh mục đã chọn; có thể resume nếu bị ngắt giữa chừng.

    Cơ chế resume: trước khi crawl một danh mục, đọc lại `books.json` cũ
    của chính danh mục đó (nếu có, xem `storage.load_category_books`).
    Sách nào đã có sẵn dữ liệu chi tiết phù hợp với yêu cầu hiện tại thì
    dùng lại thay vì gọi lại trang chi tiết. `books.json` được ghi lại
    ngay sau mỗi sách xử lý xong (không đợi hết cả danh mục), nên chạy
    lại đúng lệnh cũ sau khi bị ngắt sẽ tiếp tục đúng chỗ dừng thay vì
    crawl lại từ đầu. Lỗi khi lấy danh sách sách của 1 danh mục (mạng
    hỏng nặng, server lỗi...) chỉ bỏ qua danh mục đó, không làm hỏng cả
    tiến trình của các danh mục khác.
    """
    args.out.mkdir(parents=True, exist_ok=True)
    session = RateLimitedSession(delay_seconds=args.delay)
    bulk_session = build_bulk_session(pool_size=args.workers) if args.download_pdf else None
    categories = select_categories(args.categories)

    all_books: list[dict] = []

    for category in categories:
        log.info("=== Danh mục: %s (p_id=%s) ===", category.name, category.p_id)
        existing = load_category_books(args.out, category)
        if existing:
            log.info("  Có %d sách từ lần chạy trước, sẽ tiếp tục crawl phần còn thiếu.", len(existing))

        try:
            list_items = fetch_category_books(session, category)
        except Exception as exc:  # noqa: BLE001 - lỗi 1 danh mục không nên làm hỏng cả job
            log.error("  Lỗi lấy danh sách sách của danh mục %s, bỏ qua danh mục này: %s", category.slug, exc)
            all_books.extend(existing.values())
            continue
        log.info("  Tìm thấy %d sách.", len(list_items))

        category_books: list[dict] = []
        for item in tqdm(list_items, desc=category.slug, unit="sách"):
            record = _process_book(session, bulk_session, args, category, item, existing.get(item.product_id))
            category_books.append(record)
            write_category_books(args.out, category, category_books)

        all_books.extend(category_books)

    write_all_books(args.out, all_books)
    log.info("Hoàn tất. Tổng số sách: %d", len(all_books))


def _process_book(
    session: RateLimitedSession,
    bulk_session,
    args: argparse.Namespace,
    category: Category,
    item: BookListItem,
    prior: dict | None,
) -> dict:
    """Xử lý 1 cuốn sách: lấy chi tiết (nếu cần), tải bìa, tải PDF.

    `prior` là bản ghi của sách này trong `books.json` từ lần chạy
    trước, nếu có. Nếu bản ghi cũ đã có trường "info" (nghĩa là lần
    trước đã gọi trang chi tiết thành công) và lần chạy này không dùng
    `--skip-detail`, ta dùng lại dữ liệu cũ thay vì gọi lại trang chi
    tiết, phục vụ resume. Việc tải bìa/PDF vẫn luôn được thử lại (các
    hàm `_download_cover`/`_download_pdf` tự bỏ qua nếu file đã tồn
    tại), vì bản ghi metadata có sẵn không đảm bảo file trên đĩa đã tải
    xong ở lần chạy trước.
    """
    detail: BookDetail | None = None
    info: dict[str, str] | None = None
    can_read_online = True

    reuse_detail = prior is not None and "info" in prior and not args.skip_detail
    if reuse_detail:
        info = prior.get("info")
        can_read_online = bool(prior.get("can_read_online", True))
        record = dict(prior)
        record.update(
            {
                "product_code": item.product_code,
                "cover_url": item.cover_url,
                "price_paper": item.price_paper,
                "price_ebook": item.price_ebook,
            }
        )
    else:
        if not args.skip_detail:
            try:
                detail = fetch_book_detail(session, item.product_id)
                info = detail.info
                can_read_online = detail.can_read_online
            except Exception as exc:  # noqa: BLE001 - crawler phải bền vững, không dừng cả job
                log.warning("  Lỗi lấy chi tiết sách %s: %s", item.product_id, exc)
        record = book_record(item, detail)

    if args.download_covers and item.cover_url:
        _download_cover(session, args.out, category, item)

    if args.download_pdf and item.product_code and _is_free_readable(item, can_read_online):
        _download_pdf(
            bulk_session, args.out, category, item, info, args.max_pages, args.workers, args.keep_page_images
        )

    return record


def _is_free_readable(item: BookListItem, can_read_online: bool) -> bool:
    """Sách có phải bản 'Miễn phí' và đọc được online hay không."""
    is_free = (item.price_ebook or "").strip().lower() == "miễn phí"
    return is_free and can_read_online


def _download_cover(session: RateLimitedSession, out_dir: Path, category: Category, item: BookListItem) -> None:
    covers_dir = category_dir_for(out_dir, category) / "covers"
    covers_dir.mkdir(parents=True, exist_ok=True)
    dest = covers_dir / f"{item.product_id}.png"
    if dest.exists():
        return
    try:
        response = session.get(item.cover_url)
        dest.write_bytes(response.content)
    except Exception as exc:  # noqa: BLE001
        log.warning("  Lỗi tải ảnh bìa sách %s: %s", item.product_id, exc)


def _parse_page_count(info: dict[str, str] | None) -> int | None:
    """Đọc số trang từ field "Số trang" ở trang chi tiết sách (ví dụ "992 trang")."""
    if not info:
        return None
    raw = info.get("Số trang")
    if not raw:
        return None
    match = _PAGE_COUNT_RE.search(raw)
    return int(match.group()) if match else None


def _download_pdf(
    bulk_session,
    out_dir: Path,
    category: Category,
    item: BookListItem,
    info: dict[str, str] | None,
    max_pages: int | None,
    workers: int,
    keep_page_images: bool,
) -> None:
    content_dir = category_dir_for(out_dir, category) / "content"
    pdf_path = content_dir / f"{item.product_id}.pdf"
    if pdf_path.exists():
        return  # đã tải trước đó, không tải lại

    num_pages = _parse_page_count(info)
    if num_pages is None:
        log.warning("  Bỏ qua sách %s: không xác định được số trang.", item.product_id)
        return
    if max_pages is not None:
        num_pages = min(num_pages, max_pages)

    pages_dir = content_dir / f"{item.product_id}_pages"
    try:
        pages = download_book_pages(bulk_session, item.product_code, pages_dir, num_pages, max_workers=workers)
        assemble_pdf(pages, pdf_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("  Lỗi tải nội dung sách %s: %s", item.product_id, exc)
    finally:
        if not keep_page_images and pages_dir.exists():
            shutil.rmtree(pages_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _add_file_logging(args.out)
    try:
        crawl(args)
    except KeyboardInterrupt:
        log.warning(
            "Đã dừng theo yêu cầu người dùng (Ctrl+C). Dữ liệu đã crawl được vẫn được giữ trong '%s'; "
            "chạy lại đúng lệnh này để crawl tiếp phần còn thiếu.",
            args.out,
        )
        return 130
    except Exception:  # noqa: BLE001 - không để lỗi ngoài dự kiến làm mất log/dữ liệu đã ghi
        log.exception(
            "Crawler dừng do lỗi ngoài dự kiến. Dữ liệu đã crawl được vẫn được giữ trong '%s'; "
            "chạy lại đúng lệnh này để crawl tiếp phần còn thiếu. Xem chi tiết tại '%s/crawl.log'.",
            args.out,
            args.out,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
