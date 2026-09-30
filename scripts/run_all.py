"""
Chạy lần lượt các bước xử lý của từng nguồn dữ liệu, mỗi bước là một tiến trình con.

Mỗi phần (part) có đúng một script riêng trong scripts/<phần>/; file này chỉ gọi chúng,
truyền --data-root (và --status nếu có). Một phần lỗi không làm dừng các phần sau:
cuối cùng in tóm tắt và thoát với mã khác 0 nếu có phần lỗi.

Phần xử lý (stbook, giao_trinh) bị bỏ qua nếu thư mục raw của nó chưa có. Riêng "sea" là bước
tải nên tự tạo raw/sea_vi, không bao giờ bị bỏ qua.

Ví dụ:
    uv run python scripts/run_all.py --data-root /duong/dan/data
    uv run python scripts/run_all.py --data-root /duong/dan/data --only sea,giao_trinh
    uv run python scripts/run_all.py --data-root /duong/dan/data --status
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# phần -> (thư mục raw phải có để chạy, None nếu là bước tải; script). Thứ tự dict là thứ tự chạy.
PARTS: dict[str, tuple[str | None, str]] = {
    "sea": (None, "scripts/sea/download_all.py"),
    "stbook": ("raw/stbook", "scripts/stbook/ocr.py"),
    "giao_trinh": ("raw/giao_trinh", "scripts/giao_trinh/extract.py"),
}


def run_parts(names: list[str], data_root: Path, status: bool = False, parts: dict = PARTS) -> dict[str, str]:
    """
    Chạy từng phần trong `names`, trả về kết quả từng phần: "ok", "bỏ qua" hoặc "lỗi (mã N)".

    Args:
        names:     Các phần cần chạy, theo thứ tự.
        data_root: Thư mục gốc dữ liệu, truyền nguyên cho script con.
        status:    True thì chỉ truyền --status (xem tiến độ), không xử lý gì.
        parts:     Bảng phần -> (thư mục raw, script); mặc định là PARTS (tham số để test).
    """
    results = {}
    for name in names:
        raw, script = parts[name]
        if raw and not (data_root / raw).is_dir():
            print(f"[run_all] Bỏ qua {name}: chưa có thư mục {data_root / raw}", flush=True)
            results[name] = "bỏ qua"
            continue
        cmd = [sys.executable, str(REPO / script), "--data-root", str(data_root)] + (["--status"] if status else [])
        print(f"[run_all] === {name}: {' '.join(cmd)}", flush=True)
        code = subprocess.run(cmd).returncode
        results[name] = "ok" if code == 0 else f"lỗi (mã {code})"
    return results


def main() -> int:
    """Đọc tham số, chạy các phần đã chọn và in tóm tắt. Trả về 1 nếu có phần lỗi."""
    parser = argparse.ArgumentParser(description="Chạy các bước xử lý của mọi nguồn dữ liệu.")
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")),
                        help="Thư mục gốc dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).")
    parser.add_argument("--only", default=",".join(PARTS),
                        help=f"Các phần cần chạy, cách nhau bằng dấu phẩy (mặc định: {','.join(PARTS)}).")
    parser.add_argument("--status", action="store_true", help="Chỉ xem tiến độ của từng phần.")
    args = parser.parse_args()
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = [n for n in names if n not in PARTS]
    if unknown:
        parser.error(f"Phần không hợp lệ: {', '.join(unknown)}. Các phần có: {', '.join(PARTS)}")

    results = run_parts(names, args.data_root, args.status)
    print("[run_all] Tóm tắt: " + " | ".join(f"{n}: {r}" for n, r in results.items()), flush=True)
    return 1 if any(r.startswith("lỗi") for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
