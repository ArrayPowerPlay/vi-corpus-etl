"""Phân tích trang chi tiết một cuốn sách (/store_detail/<slug>/<p_id>).

Ghi chú: phần `<slug>` trong URL chỉ mang tính hiển thị (SEO), server chỉ
dựa vào `p_id` để trả về đúng sách — vì vậy có thể dùng slug bất kỳ, ví
dụ `/store_detail/x/<p_id>`, khi cần gọi trang chi tiết.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from .client import RateLimitedSession

_PRODUCT_CODE_RE = re.compile(r"/read_book/([^/]+)/")


@dataclass
class BookDetail:
    product_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    description: str | None = None
    product_code: str | None = None
    info: dict[str, str] = field(default_factory=dict)  # "Năm xuất bản", "Số trang", ...
    can_read_online: bool = False


def fetch_book_detail(session: RateLimitedSession, product_id: str) -> BookDetail:
    response = session.get(f"/store_detail/x/{product_id}")
    return _parse_detail_html(response.text, product_id)


def _parse_detail_html(html: str, product_id: str) -> BookDetail:
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.select_one("h1.detail-title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    authors = [a.get_text(strip=True) for a in soup.select(".detail-writer a")]

    desc_tag = soup.select_one(".detail-desc")
    description = desc_tag.get_text("\n", strip=True) if desc_tag else None

    info: dict[str, str] = {}
    for p in soup.select(".detail-info p"):
        text = p.get_text(" ", strip=True)
        if ":" not in text:
            continue
        label, _, value = text.partition(":")
        info[label.strip()] = value.strip()

    product_code = None
    read_link = soup.select_one("#whatchNow")
    can_read_online = read_link is not None
    if read_link is not None:
        match = _PRODUCT_CODE_RE.search(read_link.get("href", ""))
        if match:
            product_code = match.group(1)

    return BookDetail(
        product_id=product_id,
        title=title,
        authors=authors,
        description=description,
        product_code=product_code,
        info=info,
        can_read_online=can_read_online,
    )
