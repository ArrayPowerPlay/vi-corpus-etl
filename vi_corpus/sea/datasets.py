"""
Sổ đăng ký các bộ dữ liệu cần tải.

Mỗi bộ dữ liệu được mô tả bằng một DatasetSpec: repo trên Hugging Face và thư mục
chứa phần tiếng Việt. Muốn thêm bộ mới thì chỉ cần thêm một dòng vào DATASETS.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """
    Mô tả một bộ dữ liệu cần tải.

    Attributes:
        key:         Mã ngắn, dùng làm tên thư mục lưu (vd "sea_pile_v2").
        repo_id:     Tên repo dataset trên Hugging Face.
        vi_dir:      Thư mục trong repo chứa phần tiếng Việt (chỉ tải thư mục này).
        description: Mô tả ngắn để in ra log.
    """

    key: str
    repo_id: str
    vi_dir: str
    description: str


# Thứ tự trong dict cũng là thứ tự download_all.py chạy:
# bộ nhỏ trước để có kết quả sớm, bộ lớn sau.
DATASETS: dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in (
        DatasetSpec(
            key="sea_instruct_2602",
            repo_id="aisingapore/SEA-Instruct-2602",
            vi_dir="Vietnamese",
            description="SEA-Instruct-2602 (hội thoại instruct, ~2.7 GB, 12 file parquet)",
        ),
        DatasetSpec(
            key="sea_pile_v2",
            repo_id="aisingapore/SEA-PILE-v2",
            vi_dir="vi",
            description="SEA-PILE-v2 (văn bản web, ~132 GB, 253 file parquet)",
        ),
        DatasetSpec(
            key="sea_lion_pile_v1",
            repo_id="aisingapore/SEA-PILE-v1",
            vi_dir="sea-pile-mc4/vi",
            description="SEA-LION-Pile v1 / mC4 (văn bản web, ~107 GB, 329 file jsonl.gz)",
        ),
    )
}


def get_spec(key: str) -> DatasetSpec:
    """
    Trả về DatasetSpec theo mã.

    Raises:
        KeyError: nếu mã không có trong DATASETS (kèm danh sách mã hợp lệ).
    """
    if key not in DATASETS:
        raise KeyError(f"Không có bộ dữ liệu '{key}'. Các mã hợp lệ: {', '.join(DATASETS)}")
    return DATASETS[key]
