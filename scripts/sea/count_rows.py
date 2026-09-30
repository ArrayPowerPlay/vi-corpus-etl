"""
Đếm số dòng (row) của toàn bộ dữ liệu SEA đã tải trong <data-root>/raw/sea_vi/.

- File .parquet: đọc số dòng từ metadata ở cuối file, gần như tức thì.
- File .jsonl.gz: phải giải nén để đếm ký tự xuống dòng, nên chậm (~107 GB với SEA-PILE-v1);
  vì vậy các file được đếm song song bằng nhiều tiến trình.
Bỏ qua thư mục ẩn .cache/ mà Hugging Face tạo ra (chứa file *.incomplete đang tải dở).

Ví dụ (chạy trên server, trong Jupyter Terminal):
    uv run python scripts/sea/count_rows.py --data-root /duong/dan/data --workers 16
"""

import argparse
import gzip
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

SUFFIXES = (".parquet", ".jsonl.gz")


def count_rows(path: Path) -> int:
    """
    Trả về số dòng dữ liệu của một file .parquet hoặc .jsonl.gz.

    Với jsonl.gz, đếm số ký tự xuống dòng; nếu dòng cuối không kết thúc bằng "\\n"
    thì vẫn tính là một dòng.
    """
    if path.name.endswith(".parquet"):
        return pq.ParquetFile(path).metadata.num_rows
    n, last = 0, b"\n"
    with gzip.open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    return n + (last != b"\n")


def find_files(raw_root: Path) -> list[Path]:
    """Liệt kê mọi file dữ liệu trong raw_root, bỏ qua các thư mục ẩn như .cache/."""
    return sorted(
        p
        for p in raw_root.rglob("*")
        if p.is_file()
        and p.name.endswith(SUFFIXES)
        and not any(part.startswith(".") for part in p.relative_to(raw_root).parts)
    )


def main() -> None:
    """Đếm song song mọi file, rồi in tổng số dòng theo từng bộ dữ liệu và tổng chung."""
    parser = argparse.ArgumentParser(description="Đếm số dòng của dữ liệu SEA đã tải trong raw/sea_vi/.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.getenv("SEA_DATA_ROOT", "data")),
        help="Thư mục gốc dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Số tiến trình đếm song song."
    )
    args = parser.parse_args()

    raw_root = args.data_root / "raw" / "sea_vi"
    files = find_files(raw_root)
    if not files:
        raise SystemExit(f"Không tìm thấy file dữ liệu nào trong {raw_root}")

    totals: dict[str, list[int]] = {}  # bộ dữ liệu -> [số file, số dòng]
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (path, rows) in enumerate(zip(files, pool.map(count_rows, files)), 1):
            key = path.relative_to(raw_root).parts[0]
            t = totals.setdefault(key, [0, 0])
            t[0] += 1
            t[1] += rows
            print(f"[{i}/{len(files)}] {path.relative_to(raw_root)}: {rows:,}", flush=True)

    print(f"\n{'Bộ dữ liệu':<24}{'Số file':>10}{'Số dòng':>20}")
    for key, (n_files, rows) in totals.items():
        print(f"{key:<24}{n_files:>10,}{rows:>20,}")
    print(f"{'TỔNG':<24}{len(files):>10,}{sum(r for _, r in totals.values()):>20,}")


if __name__ == "__main__":
    main()
