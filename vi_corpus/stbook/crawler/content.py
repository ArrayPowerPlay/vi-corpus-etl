"""Tải nội dung đầy đủ của các sách điện tử "Miễn phí" và đóng gói thành PDF.

Mỗi trang sách trên trình đọc của stbook.vn được ghép từ 4 ảnh PNG, tải
qua `/cbs20/download_preview/img.json/<product_code>/img_short_<1-4><trang>/_READ`.
Module này:

1. Tải song song tất cả các ảnh góc cần thiết (`DEFAULT_WORKERS` request
   cùng lúc), ghép mỗi 4 mảnh thành 1 ảnh trang hoàn chỉnh.
2. Thu nhỏ ảnh về `max_dimension` (mặc định 2000px cạnh dài) và nén JPEG
   để dung lượng hợp lý mà vẫn đọc rõ chữ.
3. Ghép toàn bộ ảnh trang thành 1 file PDF duy nhất cho mỗi cuốn sách.

VỀ SỐ LUỒNG SONG SONG: đo thực tế trên server stbook.vn cho thấy tốc độ
phản hồi đạt trần ở khoảng ~8 request/giây bất kể mở bao nhiêu kết nối
cùng lúc (đã thử 4/8/16/24/32 luồng, từ 8 luồng trở lên không còn nhanh
hơn, có lúc còn kém hơn do dao động mạng) — vì vậy `DEFAULT_WORKERS = 8`
là điểm cân bằng tốt nhất giữa tốc độ và số kết nối mở ra server.

LƯU Ý VỀ QUY MÔ: một số đầu sách (ví dụ các bộ "Toàn tập") có tới hàng
nghìn trang. Nếu tải nhiều cuốn/toàn bộ danh mục, hãy ước lượng dung
lượng ổ đĩa và thời gian trước khi chạy — đây vẫn là nội dung có bản
quyền của NXB, chỉ nên dùng cho mục đích cá nhân/nghiên cứu hợp lý.
"""

from __future__ import annotations

import io
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

from .client import BASE_URL

DEFAULT_MAX_DIMENSION = 2000
DEFAULT_JPEG_QUALITY = 82
DEFAULT_WORKERS = 8
MAX_RETRY_BACKOFF_SECONDS = 30.0
MAX_TILE_RETRIES = 5  # số lần thử LẠI tối đa (không tính lần gọi đầu tiên) cho mỗi ảnh góc

log = logging.getLogger("stbook_crawler")


def download_book_pages(
    session: requests.Session,
    product_code: str,
    out_dir: Path,
    num_pages: int,
    max_workers: int = DEFAULT_WORKERS,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> list[Path]:
    """Tải song song `num_pages` trang đầu của một cuốn sách.

    Cần biết trước số trang cần tải (`num_pages` — lấy từ field "Số
    trang" ở trang chi tiết sách): khi tải song song, các trang hoàn
    thành không theo đúng thứ tự nên không thể "dò tìm trang cuối" như
    cách tải tuần tự.

    Tất cả 4*`num_pages` request ảnh góc được xếp vào một hàng đợi chung
    và xử lý bởi `max_workers` thread cùng lúc — `session` truyền vào
    nên được tạo bằng `client.build_bulk_session(pool_size>=max_workers)`
    để tránh nghẽn cổ chai ở tầng connection pool. Lỗi mạng/timeout khi
    tải một ảnh góc sẽ được thử lại tối đa `MAX_TILE_RETRIES` lần (backoff
    tăng dần, tối đa `MAX_RETRY_BACKOFF_SECONDS` giữa các lần). Lỗi vĩnh
    viễn của server (HTTP 4xx trừ 408/429, ví dụ 404 do trang vượt quá
    số trang thật của sách) KHÔNG được thử lại vì thử bao nhiêu lần cũng
    không khỏi. Trang nào vẫn thiếu ảnh góc (hết lượt thử, lỗi vĩnh viễn,
    hoặc server trả nội dung không phải ảnh do sách "đang cập nhật") sẽ
    bị bỏ qua và ghi log cảnh báo, không làm hỏng các trang khác.

    Trả về danh sách đường dẫn ảnh trang đã lưu, sắp đúng thứ tự trang
    (có thể ngắn hơn `num_pages` nếu có trang lỗi).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def fetch_tile(page: int, quadrant: int) -> Image.Image:
        image_name = f"img_short_{quadrant}{page}"
        url = f"{BASE_URL}/cbs20/download_preview/img.json/{product_code}/{image_name}/_READ"
        attempt = 0
        while True:
            attempt += 1
            try:
                response = session.get(url, timeout=30)
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                if _is_permanent_error(exc) or attempt > MAX_TILE_RETRIES:
                    raise
                wait = min(2.0 * attempt, MAX_RETRY_BACKOFF_SECONDS)
                log.warning(
                    "  Lỗi mạng khi tải trang %d góc %d (lần thử lại %d/%d), thử lại sau %.0fs: %s",
                    page,
                    quadrant,
                    attempt,
                    MAX_TILE_RETRIES,
                    wait,
                    exc,
                )
                time.sleep(wait)
                continue
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                raise ValueError(f"Không phải ảnh (page={page}, quadrant={quadrant}): {content_type}")
            return Image.open(io.BytesIO(response.content))

    tiles_by_page: dict[int, dict[int, Image.Image]] = defaultdict(dict)
    jobs = [(page, quadrant) for page in range(1, num_pages + 1) for quadrant in range(1, 5)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(fetch_tile, page, quadrant): (page, quadrant) for page, quadrant in jobs}
        for future in as_completed(future_to_job):
            page, quadrant = future_to_job[future]
            try:
                tiles_by_page[page][quadrant] = future.result()
            except Exception as exc:  # noqa: BLE001 - một mảnh lỗi không nên làm hỏng cả sách
                log.warning("  Lỗi tải trang %d góc %d: %s", page, quadrant, exc)

    saved: list[Path] = []
    for page in range(1, num_pages + 1):
        tiles = tiles_by_page.get(page, {})
        if len(tiles) != 4:
            log.warning("  Bỏ qua trang %d: chỉ tải được %d/4 ảnh góc", page, len(tiles))
            continue
        ordered_tiles = [tiles[q] for q in range(1, 5)]
        page_image = _stitch_quadrants(ordered_tiles)
        page_image = _downscale(page_image, max_dimension)
        out_path = out_dir / f"page_{page:04d}.jpg"
        page_image.convert("RGB").save(out_path, "JPEG", quality=jpeg_quality)
        saved.append(out_path)

    return saved


def assemble_pdf(page_image_paths: list[Path], pdf_path: Path) -> Path | None:
    """Ghép danh sách ảnh trang (theo đúng thứ tự) thành 1 file PDF.

    Trả về đường dẫn PDF, hoặc None nếu danh sách ảnh rỗng (không có gì
    để ghép).
    """
    if not page_image_paths:
        return None
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(p).convert("RGB") for p in page_image_paths]
    first, rest = images[0], images[1:]
    first.save(pdf_path, "PDF", save_all=True, append_images=rest)
    return pdf_path


def _is_permanent_error(exc: requests.exceptions.RequestException) -> bool:
    """True nếu lỗi là HTTP 4xx vĩnh viễn (thử lại vô ích), trừ 408/429 là lỗi tạm thời."""
    response = getattr(exc, "response", None)
    if response is None:
        return False
    return 400 <= response.status_code < 500 and response.status_code not in (408, 429)


def _downscale(image: Image.Image, max_dimension: int) -> Image.Image:
    longest = max(image.size)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    new_size = (round(image.width * scale), round(image.height * scale))
    return image.resize(new_size, Image.LANCZOS)


def _stitch_quadrants(tiles: list[Image.Image]) -> Image.Image:
    """Ghép 4 ảnh góc thành một ảnh trang hoàn chỉnh.

    Thứ tự quadrant 1-4 suy ra từ CSS `.img_1`..`.img_4` của trình đọc:
    1 = trên-trái, 2 = trên-phải, 3 = dưới-trái, 4 = dưới-phải (lưới 2x2).
    """
    top_left, top_right, bottom_left, bottom_right = tiles
    w, h = top_left.size
    full = Image.new("RGB", (w * 2, h * 2))
    full.paste(top_left, (0, 0))
    full.paste(top_right, (w, 0))
    full.paste(bottom_left, (0, h))
    full.paste(bottom_right, (w, h))
    return full
