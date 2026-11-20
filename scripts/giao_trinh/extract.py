"""
Xử lý thô giáo trình: kiểm kê file, trích text từng trang (PDF, docx, pptx, doc / ppt, djvu, html, epub; trang scan
thì OCR bằng chuỗi engine VLM, O-20), rồi xuất Parquet.

Đọc <data-root>/raw/giao_trinh (hoặc --root), ghi:
    interim/giao_trinh_text/<sha256>.json          text từng trang + method (checkpoint theo file)
    interim/giao_trinh_ocr_cache/                  kết quả OCR theo trang (cuốn scan dài bị ngắt thì không OCR lại từ đầu)
    processed/giao_trinh/giao_trinh.parquet        bản ghi CORPUS_SCHEMA, mỗi tài liệu một dòng
Có checkpoint theo file: bị ngắt thì chạy lại đúng lệnh cũ. Xuất Parquet chạy tự động ở cuối mỗi lần chạy (trừ --no-export).
Chỉ OCR trang scan; server vLLM chỉ được dựng khi gặp trang scan đầu tiên (--serve). Nhiều GPU: scripts/giao_trinh/run_extract_vlm.sh
(một server + một tiến trình mỗi GPU, file chia nhau bằng khoá; các tiến trình phải cùng thêm --multi-process, script đó đã thêm).
Chuỗi OCR hỏng (server chết, engine dự phòng không dựng được) dừng cả lượt, mã thoát 1; chạy lại đúng lệnh cũ để làm tiếp.
Ví dụ:
    uv run python scripts/giao_trinh/extract.py --data-root /duong/dan/data --limit-files 5 --no-ocr
    uv run --no-sync python scripts/giao_trinh/extract.py --data-root /duong/dan/data --gpu 0 --serve --vllm-bin $(which vllm)   # một GPU
    uv run python scripts/giao_trinh/extract.py --data-root /duong/dan/data --status
    uv run python scripts/giao_trinh/extract.py --data-root /duong/dan/data --export-only
    uv run python scripts/giao_trinh/extract.py --data-root /duong/dan/data --no-ocr --retry-skipped   # sau khi cài LibreOffice / djvulibre
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Thay thư mục script (sys.path[0]) bằng gốc repo: profile.py cạnh đây sẽ che module `profile` của stdlib
# (paddle/cProfile import nó khi nạp OCR) và làm hỏng việc nạp mô hình.
sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.common.ocr_cli import add_cascade_args, build_getter  # noqa: E402
from vi_corpus.common.registry import SOURCES  # noqa: E402
from vi_corpus.common.state import acquire_run_lock, setup_logging  # noqa: E402
from vi_corpus.giao_trinh.extract import run_extract, status  # noqa: E402
from vi_corpus.giao_trinh.records import OUT_PATH, export_parquet, iter_records  # noqa: E402
from vi_corpus.ocr_bakeoff.cascade import EngineUnavailable, ServerDown  # noqa: E402

log = logging.getLogger("vi_corpus")


def export(spec, data_root: Path) -> int:
    """Xuất processed/giao_trinh/giao_trinh.parquet từ checkpoint hiện có. Trả về số bản ghi."""
    out = data_root / OUT_PATH
    n = export_parquet(iter_records(spec, data_root), out)
    log.info("Đã xuất %d bản ghi ra %s", n, out)
    return n


def format_status(s: dict) -> str:
    """Tóm tắt status() thành vài dòng: số file, trang theo method, trang OCR VLM theo trạng thái / engine."""
    pages = ", ".join(f"{m}={n}" for m, n in sorted(s["pages"].items())) or "chưa có"
    lines = [f"[giao_trinh] {s['files']} file trong raw; {s['checkpoints']} đã trích text "
             f"({s['errors']} lỗi, {s['skipped']} bỏ qua); trang theo method: {pages}"]
    if s["ocr_status"]:
        n = sum(s["ocr_status"].values())
        lines.append(f"[giao_trinh] OCR VLM {n} trang: trạng thái {s['ocr_status']}, engine {s['ocr_tier']}, "
                     f"tỷ lệ flagged {s['ocr_status'].get('flagged', 0) / n:.2%}")
    if s.get("reasons"):
        lines.append(f"[giao_trinh] Trang có lý do OCR: {dict(sorted(s['reasons'].items()))}")
    if s["pages"].get("needs_ocr"):
        lines.append(f"[giao_trinh] Còn {s['pages']['needs_ocr']} trang needs_ocr: chạy lại có OCR (không --no-ocr) để làm tiếp")
    return "\n".join(lines)


def main() -> int:
    """Đọc tham số, rồi in tiến độ (--status), xuất Parquet (--export-only) hoặc trích text + xuất Parquet. Trả về 1 nếu có file lỗi hoặc chuỗi OCR hỏng."""
    parser = argparse.ArgumentParser(description="Xử lý thô giáo trình: trích text từng trang (scan thì OCR VLM) và xuất Parquet.")
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")),
                        help="Thư mục gốc dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).")
    parser.add_argument("--root", type=Path, default=None,
                        help="Thư mục giáo trình (mặc định <data-root>/raw/giao_trinh).")
    parser.add_argument("--limit-files", type=int, default=None, help="Chỉ xử lý N file đầu (chạy thử).")
    parser.add_argument("--retry-skipped", action="store_true",
                        help="Làm lại cả file từng bị bỏ qua (vd sau khi cài LibreOffice / djvulibre).")
    parser.add_argument("--multi-process", action="store_true",
                        help="Nhiều tiến trình (mỗi GPU một tiến trình) chạy cùng lúc: file chia nhau bằng khoá (run_extract_vlm.sh tự thêm).")
    parser.add_argument("--worker-name", default=None, help="Tên tiến trình, chống chạy trùng (mặc định gpu<N> hoặc main).")
    parser.add_argument("--no-export", action="store_true", help="Không xuất Parquet ở cuối (dùng khi nhiều tiến trình chạy cùng lúc).")
    parser.add_argument("--export-only", action="store_true", help="Chỉ xuất Parquet từ checkpoint hiện có rồi thoát.")
    parser.add_argument("--status", action="store_true", help="Chỉ in tiến độ rồi thoát.")
    add_cascade_args(parser)
    args = parser.parse_args()
    spec = SOURCES["giao_trinh"]
    root = args.root or args.data_root / spec.path
    if not root.is_dir() and not args.export_only:
        parser.error(f"Không thấy thư mục {root}")

    if args.status:
        print(format_status(status(root, args.data_root)))
        return 0
    if args.export_only:
        export(spec, args.data_root)
        return 0
    worker = args.worker_name or (f"gpu{args.gpu}" if args.gpu is not None else "main")
    state = args.data_root / "state" / "giao_trinh_extract" / worker
    _lock = acquire_run_lock(state)  # noqa: F841 — giữ khoá
    setup_logging(args.data_root / "logs", f"giao_trinh_extract_{worker}")
    try:
        getter = build_getter(args, args.data_root, state)
    except ValueError as e:
        parser.error(str(e))
    try:
        failed = run_extract(root, args.data_root, getter, ocr_enabled=not args.no_ocr, limit_files=args.limit_files,
                             retry_skipped=args.retry_skipped, multi_process=args.multi_process, syllables=args.syllables)
    except (ServerDown, EngineUnavailable) as exc:
        # exc_info: ghi cả chuỗi nguyên nhân, nếu không log chỉ còn dòng bọc ngoài
        log.error("%s. Dừng cả lượt (không loại trang âm thầm); sửa server / môi trường rồi chạy lại đúng lệnh cũ.", exc, exc_info=True)
        return 1
    finally:
        getter.close()
    if not args.no_export:
        export(spec, args.data_root)
    print(format_status(status(root, args.data_root)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
