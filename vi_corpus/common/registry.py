"""
Sổ đăng ký nguồn (source registry): owner, license, domain, định dạng, đường dẫn.

Đáp ứng KPI W1 "mỗi nguồn có owner, provenance/license, domain". Đường dẫn của
VISTA/VJOL là tạm; ghi đè bằng file JSON cấu hình (configs/sources.json)
thay vì sửa code khi có dữ liệu thật.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class SourceSpec:
    """
    Mô tả một nguồn dữ liệu.

    Attributes:
        key:      Mã nguồn (vd "sea_pile_v2").
        fmt:      Định dạng file: "parquet", "jsonl.gz", "pdf" hoặc "stbook_pdf" (thư mục của stbook-crawler).
        path:     Thư mục chứa dữ liệu, tương đối so với data_root (hoặc tuyệt đối).
        owner:    Chủ sở hữu dữ liệu.
        license:  Giấy phép / điều khoản sử dụng.
        domain:   Lĩnh vực nội dung.
        text_col: Tên cột chứa text (bỏ trống với nguồn PDF hoặc dạng hội thoại).
        rights_status: Trạng thái quyền; mặc định "unknown" (sẽ bị quarantine).
    """

    key: str
    fmt: str
    path: str
    owner: str
    license: str
    domain: str
    text_col: str = "text"
    rights_status: str = "unknown"


SOURCES: dict[str, SourceSpec] = {
    s.key: s
    for s in (
        SourceSpec("sea_instruct_2602", "parquet", "raw/sea_vi/sea_instruct_2602/Vietnamese",
                   "AI Singapore", "ODC-By 1.0", "instruction", text_col="conversations"),
        SourceSpec("sea_pile_v2", "parquet", "raw/sea_vi/sea_pile_v2/vi",
                   "AI Singapore", "ODC-By 1.0 + CommonCrawl ToU", "web"),
        SourceSpec("sea_lion_pile_v1", "jsonl.gz", "raw/sea_vi/sea_lion_pile_v1/sea-pile-mc4/vi",
                   "AI Singapore", "ODC-By 1.0 + CommonCrawl ToU", "web"),
        SourceSpec("stbook", "stbook_pdf", "raw/stbook", "NXB Chính trị quốc gia Sự thật",
                   "bản quyền NXB (đọc miễn phí online, chưa rõ quyền tái sử dụng)",
                   "sách chính trị - xã hội", text_col=""),
        SourceSpec("giao_trinh", "giao_trinh", "raw/giao_trinh", "nhiều tác giả / trường / NXB (tổng hợp từ Google Drive)",
                   "chưa rõ (có sách từ PDFDrive, z-lib)", "giáo trình đại học đa ngành", text_col=""),
        # vista: đường dẫn tạm. vjol: raw/VJOL, cấu trúc thư mục bên trong sẽ bổ sung khi có dữ liệu.
        # Cập nhật trong configs/sources.json khi có dữ liệu thật.
        SourceSpec("vista", "pdf", "external/vista", "chưa rõ", "chưa rõ", "scientific"),
        SourceSpec("vjol", "pdf", "raw/VJOL", "chưa rõ", "chưa rõ", "scientific"),
    )
}


def load_sources(config_path: Path | None = None) -> dict[str, SourceSpec]:
    """
    Trả về SOURCES, có ghi đè từ file JSON nếu tồn tại.

    File JSON có dạng {"vista": {"path": "/mnt/vista", "license": "..."}, ...};
    một nguồn chưa có trong SOURCES (vd "drive_books") sẽ được thêm mới và phải đủ
    các trường bắt buộc của SourceSpec.

    Raises:
        TypeError: nếu nguồn mới thiếu trường bắt buộc.
    """
    sources = dict(SOURCES)
    if config_path is None or not config_path.exists():
        return sources
    overrides = json.loads(config_path.read_text(encoding="utf-8"))
    for key, fields in overrides.items():
        sources[key] = replace(sources[key], **fields) if key in sources else SourceSpec(key=key, **fields)
    return sources
