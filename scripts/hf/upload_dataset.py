"""
Đẩy thư mục dataset đã xuất (scripts/hf/export_dataset.py) lên Hugging Face, mặc định repo PRIVATE.

Ví dụ:
    uv run --no-sync python scripts/hf/upload_dataset.py --data-root data --repo-id arraypowerplay/vi-scientific-corpus
Token đọc từ HF_TOKEN trong .env (hoặc biến môi trường). Đẩy lại thì chỉ tải file đổi (upload_large_folder có checkpoint, chạy lại
được sau khi đứt). Muốn công khai: đổi trong Settings của repo, hoặc chạy lại với --public (cần suy nghĩ về bản quyền trước).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402


def main() -> int:
    """Tạo repo dataset (nếu chưa có), đẩy cả thư mục, in link; trả về 0."""
    p = argparse.ArgumentParser(description="Đẩy dataset VJOL + VISTA lên Hugging Face.")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    p.add_argument("--folder", type=Path, default=None, help="Mặc định <data-root>/processed/hf_export.")
    p.add_argument("--repo-id", default="arraypowerplay/vi-scientific-corpus")
    p.add_argument("--public", action="store_true", help="Tạo repo công khai (mặc định private).")
    args = p.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    folder = args.folder or args.data_root / "processed" / "hf_export"
    if not (folder / "README.md").exists():
        raise SystemExit(f"Chưa có {folder}/README.md: chạy export_dataset.py trước.")
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=folder, private=not args.public,
                            ignore_patterns=["export_stats.json", "*.part"])
    print(f"Xong: https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
