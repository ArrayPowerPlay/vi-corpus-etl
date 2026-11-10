"""
Tìm các PDF VJOL có lớp chữ chứa surrogate lẻ (U+D800–U+DFFF) mà J-62 đã sửa, để gửi cho Claude tái hiện lỗi.

Chỉ CPU, không OCR, không GPU: đọc lớp chữ bằng PyMuPDF (cùng hàm page_text của extract_pdf) và tìm ký tự surrogate lẻ. Chỉ quét PDF CHƯA có checkpoint,
theo đúng thứ tự mà run_extract_vlm.sh xử lý. Lần chạy full chết ở những file đầu tiên chưa có checkpoint nên thường tìm thấy sau vài trăm file; quét dừng
khi đủ --max-hits file hỏng. Mượn cache sha256 `state/inventory_vjol_text.json` của lần chạy trước nên không băm lại cả kho.

Ghi <data-root>/processed/vjol_surrogate_pdfs/: bản chép các PDF tìm thấy + report.json (đường dẫn, trang, số ký tự, mã điểm).
Ví dụ:
    uv run --no-sync python scripts/vjol/find_surrogate_pdfs.py --data-root data
    tar czf vjol_surrogate_pdfs.tar.gz -C data/processed vjol_surrogate_pdfs
"""

import argparse
import json
import os
import re
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

import pymupdf  # noqa: E402

from vi_corpus.common.batch_extract import checkpoint_path, inventory  # noqa: E402
from vi_corpus.common.pdf_text import page_text  # noqa: E402
from vi_corpus.common.registry import SOURCES  # noqa: E402
from vi_corpus.vjol.extract import TEXT_DIR, select_files  # noqa: E402

SURROGATE = re.compile("[\ud800-\udfff]")


def scan_pdf(path: Path) -> dict:
    """
    Quét lớp chữ của một PDF tìm surrogate lẻ.

    Returns:
        {"path", "pages": [số trang (từ 1) có surrogate], "chars": tổng ký tự, "codes": các mã điểm khác nhau (dạng "U+D835"), "error": thông báo hoặc None}.
    """
    out = {"path": str(path), "pages": [], "chars": 0, "codes": [], "error": None}
    try:
        with pymupdf.open(path) as doc:
            for i, page in enumerate(doc):
                found = SURROGATE.findall(page_text(page))
                if found:
                    out["pages"].append(i + 1)
                    out["chars"] += len(found)
                    out["codes"] = sorted({*out["codes"], *(f"U+{ord(c):04X}" for c in found)})
    except Exception as exc:  # noqa: BLE001 — PDF hỏng thì ghi lại rồi quét file kế
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def main() -> int:
    """Chọn PDF chưa có checkpoint, quét song song theo thứ tự, dừng khi đủ file hỏng, chép file tìm thấy và ghi report.json."""
    p = argparse.ArgumentParser(description="Tìm PDF VJOL có surrogate lẻ trong lớp chữ (CPU).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục dữ liệu gốc.")
    p.add_argument("--max-hits", type=int, default=4, help="Dừng khi tìm đủ N file hỏng (mặc định 4).")
    p.add_argument("--max-scan", type=int, default=None, help="Chỉ quét tối đa N file chưa có checkpoint (mặc định: đến khi đủ --max-hits hoặc hết).")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4), help="Số tiến trình quét song song.")
    args = p.parse_args()

    root = args.data_root / SOURCES["vjol"].path
    groups = inventory(root, files=select_files(root), cache_path=args.data_root / "state" / "inventory_vjol_text.json")
    todo = [paths[0] for sha, paths in groups.items() if not checkpoint_path(args.data_root, TEXT_DIR, sha).exists()]
    todo = todo[:args.max_scan]
    print(f"{len(groups)} nội dung duy nhất, {len(todo)} chưa có checkpoint; quét theo thứ tự bằng {args.workers} tiến trình...", flush=True)

    hits: list[dict] = []
    scanned = 0
    with Pool(args.workers) as pool:
        for res in pool.imap(scan_pdf, todo, chunksize=4):
            scanned += 1
            if res["pages"]:
                hits.append(res)
                print(f"TÌM THẤY {res['path']}: trang {res['pages'][:10]}, {res['chars']} ký tự, mã {res['codes']}", flush=True)
                if len(hits) >= args.max_hits:
                    break
            elif scanned % 500 == 0:
                print(f"  đã quét {scanned}/{len(todo)} file, {len(hits)} file hỏng", flush=True)
        pool.terminate()

    out_dir = args.data_root / "processed" / "vjol_surrogate_pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for res in hits:
        shutil.copy2(res["path"], out_dir / Path(res["path"]).name)
    (out_dir / "report.json").write_text(json.dumps({"scanned": scanned, "hits": hits}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Xong: quét {scanned} file, tìm {len(hits)} file hỏng. Kết quả ở {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
