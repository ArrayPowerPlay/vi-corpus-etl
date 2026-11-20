"""
Gom mọi PDF "2 lớp" (có lớp chữ dùng được, không cần OCR) của VJOL và VISTA về MỘT thư mục để đẩy lên MinIO. Chỉ CPU, chỉ đọc checkpoint, không mở PDF.

PDF 2 lớp ở đây = checkpoint của lượt trích `--no-ocr` (interim/<nguồn>_text/<sha256>.json) thoả cả bốn điều kiện:
  1. không lỗi, không bị bỏ qua, và là bản trích hiện hành (`extract_version`);
  2. KHÔNG có trang `needs_ocr` (trang scan, lỗi font, PDF do phần mềm OCR đời cũ tạo: những bài này phải OCR nên không phải 2 lớp);
  3. KHÔNG có trang `ocr_vlm` (đã OCR bằng GPU = chữ không lấy từ lớp chữ gốc);
  4. có ít nhất một trang `text` hoặc `tcvn3` (trang `empty` = trang trắng thì chấp nhận).
Trang `tcvn3` (font TCVN3 cũ, đã giải mã) tính là có lớp chữ; thêm `--no-tcvn3` nếu muốn loại.

Thư mục đầu ra giữ nguyên bố cục như trên MinIO / trong raw, nên đẩy lên bằng một lệnh `rclone copy`:
    <out>/vjol/<năm>/<article_id>_<galley_id>.pdf
    <out>/vista/<năm>/<publication_id>.pdf        (và unknown/<sha256>.pdf)
    <out>/manifest.jsonl                           mỗi dòng một PDF: nguồn, đường dẫn tương đối, sha256, số trang theo method
Mặc định tạo hard link (không tốn thêm đĩa, nhanh); khác ổ đĩa thì tự chép. `--mode symlink|copy` để đổi. Chạy lại được: file đã có đúng kích thước thì bỏ qua.

Ví dụ:
    uv run --no-sync python scripts/collect_text_layer_pdfs.py --data-root $SEA_DATA_ROOT --dry-run
    uv run --no-sync python scripts/collect_text_layer_pdfs.py --data-root $SEA_DATA_ROOT --out $SEA_DATA_ROOT/processed/pdf_text_layer
"""

import argparse
import errno
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common.registry import SOURCES  # noqa: E402
from vi_corpus.vjol.ocr_count import CACHE_DIR, LAYOUTS, load_entries  # noqa: E402

TEXT_METHODS = ("text", "tcvn3")  # trang lấy thẳng từ lớp chữ của PDF
NEUTRAL_METHODS = ("empty",)  # trang trắng: không làm PDF mất tư cách 2 lớp
REASON_OK = "ok"


def classify(entry: dict, allow_tcvn3: bool) -> str:
    """
    Xếp một checkpoint (mục rút gọn của `load_entries`) vào "ok" hoặc lý do bị loại.

    Args:
        entry:       Mục rút gọn {"pages", "methods", "error", "skipped", "current", ...}.
        allow_tcvn3: False thì trang `tcvn3` không được coi là lớp chữ (bài có trang tcvn3 bị loại).

    Returns:
        "ok" nếu là PDF 2 lớp; ngược lại một trong "error", "skipped", "old_version", "needs_ocr", "ocr_vlm", "tcvn3", "no_text".
    """
    methods = entry["methods"]
    if entry["error"]:
        return "error"
    if entry["skipped"]:
        return "skipped"
    if not entry["current"]:
        return "old_version"
    if methods.get("needs_ocr"):
        return "needs_ocr"
    if methods.get("ocr_vlm"):
        return "ocr_vlm"
    if not allow_tcvn3 and methods.get("tcvn3"):
        return "tcvn3"
    if not any(methods.get(m) for m in TEXT_METHODS):
        return "no_text"
    other = set(methods) - set(TEXT_METHODS) - set(NEUTRAL_METHODS)
    return "no_text" if other else REASON_OK  # method lạ (chưa biết): không đoán, loại cho an toàn


def dest_rel(path_str: str, key: str, data_root: Path) -> Path:
    """
    Đường dẫn đích của một PDF, tương đối so với thư mục đầu ra: <nguồn>/<đường dẫn tương đối trong raw/<nguồn>>.

    Args:
        path_str:  Một phần tử của `paths` trong checkpoint (tương đối so với data_root, hoặc tuyệt đối).
        key:       Mã nguồn ("vjol", "vista").
        data_root: Thư mục dữ liệu gốc.

    Returns:
        Path như vjol/2015/123_456.pdf; PDF nằm ngoài raw/<nguồn> thì chỉ giữ tên file.
    """
    src = data_root / path_str
    raw = data_root / SOURCES[key].path
    return Path(key) / (src.relative_to(raw) if src.is_relative_to(raw) else Path(src.name))


def place(src: Path, dst: Path, mode: str) -> str:
    """
    Đưa một PDF vào thư mục đầu ra theo `mode`; bỏ qua nếu đích đã có đúng kích thước.

    Returns:
        "exists" (đã có), "hardlink", "symlink", "copy" (cách đã dùng; hardlink khác ổ đĩa thì tự chuyển sang "copy"), hoặc "missing" nếu PDF nguồn không còn.
    """
    try:
        size = src.stat().st_size
    except OSError:
        return "missing"
    if dst.exists() and dst.stat().st_size == size:
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)  # đích hỏng / lệch kích thước thì làm lại
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return "symlink"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError as e:
            if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
                raise
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)  # ghi nguyên tử: lệnh rclone chạy song song không thấy file dở
    return "copy"


def main() -> int:
    """Đọc tham số, chọn PDF 2 lớp của từng nguồn, đưa vào thư mục đầu ra, ghi manifest, in tóm tắt. Trả về 1 nếu có PDF nguồn biến mất, ngược lại 0."""
    p = argparse.ArgumentParser(description="Gom PDF 2 lớp (không cần OCR) của VJOL + VISTA về một thư mục.")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục dữ liệu gốc.")
    p.add_argument("--out", type=Path, default=None, help="Thư mục đầu ra (mặc định <data-root>/processed/pdf_text_layer).")
    p.add_argument("--sources", default="vjol,vista", help="Các nguồn, cách nhau dấu phẩy (mặc định %(default)s).")
    p.add_argument("--mode", choices=["hardlink", "symlink", "copy"], default="hardlink", help="Cách đưa file vào thư mục đầu ra (mặc định %(default)s).")
    p.add_argument("--no-tcvn3", action="store_true", help="Loại bài có trang font TCVN3 (mặc định coi là có lớp chữ).")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4), help="Số luồng/tiến trình (mặc định %(default)s).")
    p.add_argument("--dry-run", action="store_true", help="Chỉ đếm và in tóm tắt, không tạo file nào.")
    args = p.parse_args()
    keys = [k.strip() for k in args.sources.split(",") if k.strip()]
    unknown = [k for k in keys if k not in LAYOUTS]
    if unknown:
        p.error(f"Nguồn không biết: {unknown}; chọn trong {sorted(LAYOUTS)}")
    data_root = args.data_root
    out = args.out or data_root / "processed" / "pdf_text_layer"

    jobs: list[tuple[Path, Path]] = []  # (PDF nguồn, đích)
    manifest: list[dict] = []
    for key in keys:
        entries, _, unreadable = load_entries(data_root / LAYOUTS[key].text_dir, data_root / CACHE_DIR / f"{key}.json", args.workers)
        reasons: Counter = Counter()
        for name, e in sorted(entries.items()):
            reason = classify(e, not args.no_tcvn3)
            reasons[reason] += 1
            if reason != REASON_OK:
                continue
            for path_str in e["paths"]:
                rel = dest_rel(path_str, key, data_root)
                jobs.append((data_root / path_str, out / rel))
                manifest.append({"source": key, "path": rel.as_posix(), "sha256": name.removesuffix(".json"), "pages": e["pages"], "methods": e["methods"]})
        print(f"[{key}] checkpoint {len(entries):,} (không đọc được {unreadable}): " + ", ".join(f"{k} {v:,}" for k, v in reasons.most_common()), flush=True)
    print(f"PDF 2 lớp: {len(jobs):,} file", flush=True)
    if args.dry_run or not jobs:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        results = Counter(pool.map(lambda j: place(j[0], j[1], args.mode), jobs))
    tmp = out / "manifest.jsonl.part"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out / "manifest.jsonl")
    print(f"Đã đưa vào {out}: " + ", ".join(f"{k} {v:,}" for k, v in results.most_common()), flush=True)
    if results["missing"]:
        print(f"CẢNH BÁO: {results['missing']:,} PDF nguồn không còn trong raw (checkpoint trỏ tới file đã xoá/di chuyển).", flush=True)
    return 1 if results["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
