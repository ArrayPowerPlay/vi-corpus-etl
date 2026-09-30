"""Phân tích (parse) trang danh sách sách theo danh mục.

URL của một trang danh mục có dạng:
    https://stbook.vn/category/<slug>/<p_id>            (trang 1)
    https://stbook.vn/category/<slug>/<p_id>/<offset>    (offset = 0, 1, 2, ...
                                                           ứng với trang 1, 2, 3, ...)

Mỗi trang liệt kê tối đa 15 cuốn sách. Module này chỉ đọc HTML tĩnh
(không cần chạy JavaScript) vì server đã render sẵn danh sách sách.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from .client import RateLimitedSession
from .categories import Category

_COVER_CODE_RE = re.compile(r"/static/covers/([^/]+)/thumb\.png")


@dataclass
class BookListItem:
    """Thông tin tóm tắt của một cuốn sách, lấy từ trang danh mục."""

    product_id: str
    title: str
    author: str | None
    product_code: str | None
    cover_url: str | None
    price_paper: str | None
    price_ebook: str | None
    category_p_id: int
    category_name: str


def fetch_category_books(session: RateLimitedSession, category: Category) -> list[BookListItem]:
    """Lấy toàn bộ sách của một danh mục, tự động duyệt qua mọi trang.

    Dừng lại khi gặp một trang không còn cuốn sách nào (thay vì dựa vào
    thanh phân trang, vốn có thể chỉ hiển thị một cửa sổ số trang giới
    hạn) — cách này đảm bảo lấy hết dữ liệu kể cả khi danh mục có rất
    nhiều trang.
    """
    books: list[BookListItem] = []
    offset = 0
    while True:
        path = f"/category/{category.slug}/{category.p_id}/{offset}"
        response = session.get(path)
        page_books = _parse_listing_html(response.text, category)
        if not page_books:
            break
        books.extend(page_books)
        offset += 1
    return books


def _parse_listing_html(html: str, category: Category) -> list[BookListItem]:
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one(".booklist.booklist-cat .booklist-wrap > ul")
    if container is None:
        return []

    items: list[BookListItem] = []
    for li in container.find_all("li", recursive=False):
        title_link = li.select_one("h2.blw-title a")
        if title_link is None:
            continue
        product_id = title_link.get("p_id", "").strip()
        title = title_link.get("title", title_link.get_text(strip=True)).strip()
        if not product_id or not title:
            continue

        author_tag = li.select_one("h3.blw-writer")
        author = author_tag.get_text(strip=True) if author_tag else None

        img_tag = li.select_one(".blw-thumb img")
        cover_url = img_tag.get("src") if img_tag else None
        product_code = None
        if cover_url:
            match = _COVER_CODE_RE.search(cover_url)
            if match:
                product_code = match.group(1)

        price_paper = None
        price_ebook = None
        for version_p in li.select("p.blw-version"):
            label = version_p.get_text(" ", strip=True)
            value_span = version_p.select_one("span")
            value = value_span.get_text(strip=True) if value_span else None
            if label.startswith("Bản giấy"):
                price_paper = value or None
            elif label.startswith("Bản điện tử"):
                price_ebook = value or None

        items.append(
            BookListItem(
                product_id=product_id,
                title=title,
                author=author,
                product_code=product_code,
                cover_url=cover_url,
                price_paper=price_paper,
                price_ebook=price_ebook,
                category_p_id=category.p_id,
                category_name=category.name,
            )
        )
    return items
