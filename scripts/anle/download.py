"""
Tải bộ án lệ / bản án `tmquan/anle-toaan-gov-vn` (ViLA đã crawl anle.toaan.gov.vn và công bố, NS-03).

Không crawl lại cổng anle.toaan.gov.vn: lấy bản đã công bố trên Hugging Face, chỉ tải 2 file cần cho văn bản
(`documents-*.parquet` có toàn văn markdown, và `README.md` để giữ nguồn gốc). Bỏ qua embedding (57 MB), ảnh, sankey.
Ghim một revision cho cả lượt, kiểm cỡ file, ghi `state/anle/download.json` (revision, file, cỡ) bằng ghi nguyên tử.
Chạy lại thì bỏ qua file đã đủ cỡ. Không cần HF_TOKEN (repo công khai).

Ví dụ:
    uv run python scripts/anle/download.py --data-root /duong/dan/data
    uv run python scripts/anle/download.py --data-root /duong/dan/data --status
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # để import được vi_corpus

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from vi_corpus.common.state import read_json, setup_logging, write_json_atomic, logger  # noqa: E402

REPO_ID = "tmquan/anle-toaan-gov-vn"
KEY = "anle"
# Chỉ các file cần: văn bản (parquet) và thẻ dataset (nguồn gốc, giấy phép, cách gán nhãn).
WANTED = ("documents-00000-of-00001.parquet", "README.md")


def main() -> int:
    """
    Tải 2 file của dataset về `<data-root>/raw/anle/` rồi ghi checkpoint.

    Trả về 0 nếu mọi file đủ cỡ so với Hugging Face, 1 nếu có file lỗi. Với `--status` chỉ in checkpoint
    của lượt tải gần nhất, không gọi mạng.
    """
    p = argparse.ArgumentParser(description="Tải dataset án lệ của ViLA (tmquan/anle-toaan-gov-vn).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục gốc dữ liệu (mặc định SEA_DATA_ROOT hoặc ./data).")
    p.add_argument("--status", action="store_true", help="Chỉ in kết quả lượt tải gần nhất, không gọi mạng.")
    args = p.parse_args()

    raw_dir = args.data_root / "raw" / KEY
    state_file = args.data_root / "state" / KEY / "download.json"
    if args.status:
        print(read_json(state_file) or "Chưa tải lần nào.")
        return 0

    setup_logging(args.data_root / "logs", KEY)
    api = HfApi()
    info = api.dataset_info(REPO_ID, files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings if s.rfilename in WANTED}
    missing = [w for w in WANTED if w not in sizes]
    if missing:
        logger.error("Repo %s không còn file: %s", REPO_ID, missing)
        return 1
    logger.info("Revision %s, tải %d file vào %s", info.sha, len(sizes), raw_dir)

    files, ok = {}, True
    for name, size in sizes.items():
        dest = raw_dir / name
        if not (dest.exists() and dest.stat().st_size == size):
            hf_hub_download(REPO_ID, name, repo_type="dataset", revision=info.sha, local_dir=raw_dir)
        got = dest.stat().st_size if dest.exists() else -1
        files[name] = {"size": size, "ok": got == size}
        ok &= got == size
        logger.info("%s: %s (%d byte)", name, "đủ" if got == size else f"LỆCH cỡ (có {got})", size)

    write_json_atomic(state_file, {"repo_id": REPO_ID, "revision": info.sha, "last_modified": str(info.last_modified), "files": files})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
