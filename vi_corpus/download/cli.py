"""
Tham số dòng lệnh dùng chung cho mọi script trong thư mục scripts/.

Mỗi script chỉ cần gọi main([...các mã bộ dữ liệu...], "mô tả").
"""

import argparse
import os
from pathlib import Path

from .datasets import get_spec
from .downloader import print_status, run_dataset


def build_parser(description: str) -> argparse.ArgumentParser:
    """Tạo parser với các tham số chung."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.getenv("SEA_DATA_ROOT", "data")),
        help="Thư mục gốc lưu dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).",
    )
    parser.add_argument("--workers", type=int, default=8, help="Số file tải cùng lúc (mặc định 8).")
    parser.add_argument(
        "--limit-files", type=int, default=None, help="Chỉ tải N file đầu tiên (để chạy thử)."
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Tính sha256 sau khi tải để chắc chắn file không hỏng (chậm hơn một chút).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=5, help="Số lần thử lại mỗi file khi lỗi (mặc định 5)."
    )
    parser.add_argument(
        "--status", action="store_true", help="Chỉ in tiến độ đã tải rồi thoát (không tải gì)."
    )
    return parser


def main(keys: list[str], description: str) -> int:
    """
    Chạy tải (hoặc in tiến độ) cho lần lượt các bộ dữ liệu trong `keys`.

    Returns:
        Mã thoát: 0 nếu mọi file đều xong, 1 nếu còn file lỗi.
    """
    for key in keys:
        get_spec(key)  # báo lỗi sớm nếu mã sai
    args = build_parser(description).parse_args()

    if args.status:
        for key in keys:
            print_status(key, args.data_root)
        return 0

    failed = 0
    for key in keys:
        result = run_dataset(
            key,
            data_root=args.data_root,
            workers=args.workers,
            limit_files=args.limit_files,
            verify_sha256=args.verify_sha256,
            max_retries=args.max_retries,
        )
        failed += result.failed
    return 1 if failed else 0
