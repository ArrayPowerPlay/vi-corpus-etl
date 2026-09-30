"""Ghi kết quả crawl ra đĩa, tổ chức theo thư mục từng danh mục.

Cấu trúc thư mục đầu ra:

    data/
      <category-slug>/
        books.json         # toàn bộ sách của riêng danh mục này
        covers/             # ảnh bìa (nếu bật --download-covers)
        content/<product_id>/page_0001.jpg ...  # trang xem thử (nếu bật --fetch-content)
      all_books.json        # gộp toàn bộ sách của tất cả danh mục
"""

from __future__ import annotations

import json
from pathlib import Path

from .categories import Category
from .parse_category import BookListItem
from .parse_detail import BookDetail


def book_record(item: BookListItem, detail: BookDetail | None) -> dict:
    """Gộp dữ liệu từ trang danh mục + trang chi tiết thành 1 bản ghi sách."""
    record = {
        "product_id": item.product_id,
        "title": item.title,
        "author": item.author,
        "product_code": item.product_code,
        "cover_url": item.cover_url,
        "price_paper": item.price_paper,
        "price_ebook": item.price_ebook,
        "category_p_id": item.category_p_id,
        "category_name": item.category_name,
        "detail_url": f"https://stbook.vn/store_detail/x/{item.product_id}",
    }
    if detail is not None:
        record.update(
            {
                "authors": detail.authors,
                "description": detail.description,
                "info": detail.info,
                "can_read_online": detail.can_read_online,
            }
        )
    return record


def load_category_books(data_dir: Path, category: Category) -> dict[str, dict]:
    """Đọc `books.json` cũ của một danh mục (nếu có), phục vụ resume.

    Trả về dict ánh xạ `product_id -> bản ghi sách` đã lưu ở lần chạy
    trước. Trả về dict rỗng nếu chưa từng crawl danh mục này, hoặc nếu
    file bị hỏng/không đọc được (khi đó coi như crawl lại từ đầu thay vì
    làm crash chương trình).
    """
    path = data_dir / category.slug / "books.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {book["product_id"]: book for book in payload.get("books", []) if "product_id" in book}


def write_category_books(data_dir: Path, category: Category, books: list[dict]) -> Path:
    category_dir = data_dir / category.slug
    category_dir.mkdir(parents=True, exist_ok=True)
    out_path = category_dir / "books.json"
    payload = {
        "category": {"p_id": category.p_id, "slug": category.slug, "name": category.name},
        "book_count": len(books),
        "books": books,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def write_all_books(data_dir: Path, all_books: list[dict]) -> Path:
    out_path = data_dir / "all_books.json"
    payload = {"book_count": len(all_books), "books": all_books}
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def category_dir_for(data_dir: Path, category: Category) -> Path:
    return data_dir / category.slug
