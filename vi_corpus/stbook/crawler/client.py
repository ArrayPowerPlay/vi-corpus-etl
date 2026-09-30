"""HTTP client dùng chung cho toàn bộ crawler.

Cung cấp một `requests.Session` đã được cấu hình sẵn:
- User-Agent giống trình duyệt thật (trang stbook.vn có vẻ dựa vào header
  này để trả về đúng nội dung HTML).
- Tự động thử lại khi gặp lỗi mạng/HTTP tạm thời (5xx, timeout).
- Giới hạn tốc độ (rate limit) giữa các request để không gây tải nặng lên
  server của NXB Chính trị Quốc gia Sự thật.
"""

from __future__ import annotations

import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://stbook.vn"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def build_bulk_session(pool_size: int = 8) -> requests.Session:
    """Tạo một `requests.Session` KHÔNG rate-limit theo delay tuần tự.

    Dùng cho tải nội dung sách song song (nhiều thread) — tốc độ được
    khống chế bằng số lượng worker (`max_workers` của `ThreadPoolExecutor`)
    thay vì độ trễ giữa từng request. `pool_size` cần >= số worker đồng
    thời, nếu không connection pool sẽ nghẽn (thread phải đợi có kết nối
    rảnh). Mặc định 8 vì đo thực tế cho thấy server stbook.vn không phản
    hồi nhanh hơn dù mở nhiều kết nối hơn mức này (xem `content.py`).
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    )
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class RateLimitedSession:
    """Wrapper quanh `requests.Session`, tự chèn độ trễ giữa các request.

    Args:
        delay_seconds: khoảng nghỉ tối thiểu (giây) giữa hai request liên
            tiếp, nhằm crawl một cách lịch sự (polite crawling).
    """

    def __init__(self, delay_seconds: float = 0.8) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )
        retry = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self.delay_seconds = delay_seconds
        self._last_request_ts = 0.0

    def get(self, path_or_url: str, **kwargs) -> requests.Response:
        """GET một URL (tuyệt đối hoặc tương đối so với BASE_URL)."""
        url = path_or_url if path_or_url.startswith("http") else BASE_URL + path_or_url

        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)

        response = self._session.get(url, timeout=30, **kwargs)
        self._last_request_ts = time.monotonic()
        response.raise_for_status()
        return response
