"""Danh sách các danh mục (mục/topic) sách cố định trên stbook.vn.

Danh sách này được lấy trực tiếp từ menu "Danh mục sách" ở trang chủ
(https://stbook.vn/) — mỗi mục có một `p_id` (category id nội bộ của
website) và một `slug` do JavaScript của trang tự sinh ra từ tiêu đề
(hàm `friendly()` trong static/js: bỏ dấu tiếng Việt, thay khoảng trắng
bằng dấu gạch ngang). URL danh mục có dạng:

    /category/<slug>/<p_id>            -> trang 1
    /category/<slug>/<p_id>/<offset>   -> trang offset+1 (offset bắt đầu từ 0)

Nếu website bổ sung/đổi tên danh mục trong tương lai, cập nhật danh sách
này cho khớp (mở trang chủ, xem lại khối `<div id="md-list">`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    p_id: int
    slug: str
    name: str


CATEGORIES: list[Category] = [
    Category(13, "kinh-dien", "Kinh điển"),
    Category(14, "chu-tich-ho-chi-minh", "Chủ tịch Hồ Chí Minh"),
    Category(15, "lanh-dao-dang-nha-nuoc", "Lãnh đạo Đảng, Nhà nước"),
    Category(16, "van-kien-dang", "Văn kiện Đảng"),
    Category(17, "xay-dung-dang-nha-nuoc", "Xây dựng Đảng, Nhà nước"),
    Category(18, "kinh-te", "Kinh tế"),
    Category(19, "van-hoa-xa-hoi", "Văn hóa, xã hội"),
    Category(20, "phap-luat", "Pháp luật"),
    Category(21, "nhung-van-de-quoc-te", "Những vấn đề quốc tế"),
    Category(22, "quoc-phong-an-ninh-doi-ngoai", "Quốc phòng, an ninh, đối ngoại"),
    Category(26, "sach-truyen-noi-audio", "Sách truyện nói (Audio)"),
    Category(30, "sach-da-phuong-tien-multimedia", "Sách đa phương tiện (multimedia)"),
    Category(
        35,
        "van-kien-dai-hoi-xiv-cac-hoi-nghi-trung-uong-khoa-xiv-va-tai-lieu-tham-khao",
        "Văn kiện Đại hội XIV, các hội nghị Trung ương khóa XIV và tài liệu tham khảo",
    ),
    Category(24, "cac-an-pham-khac", "Các ấn phẩm khác"),
    Category(32, "tai-lieu", "Tài liệu"),
]
