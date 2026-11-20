"""
Kiểm chứng bộ dò chữ hỏng do font trên giáo trình bằng OCR (GT-06): lấy mẫu phân tầng trang bị gắn cờ + trang đối chứng, OCR lại bằng chuỗi VLM, đo độ trùng từ
giữa lớp chữ và OCR (logic ở vi_corpus/giao_trinh/verify.py).

Cần hai thư mục checkpoint: `--flagged-dir` (lần trích có bật bộ dò, trang hỏng là needs_ocr) và `--before-dir` (lần trích trước đó, để biết trang từng là text hay tcvn3).
Ghi <data-root>/processed/giao_trinh_verify_flagged/{pages.jsonl, summary.json}; chạy lại đúng lệnh cũ thì làm tiếp (bỏ qua trang đã có trong pages.jsonl).
Chuỗi OCR hỏng (server chết, engine không dựng được) dừng cả lượt, mã thoát 1.
Ví dụ (một GPU, dựng server vLLM):
    uv run --no-sync python scripts/giao_trinh/verify_flagged.py --data-root data --flagged-dir data/interim/giao_trinh_text_gt06_naive \
        --before-dir data/interim/giao_trinh_text_before_gt06 --per-stratum 150 --gpu 0 --serve --vllm-bin $(which vllm)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Thay thư mục script (sys.path[0]) bằng gốc repo (xem scripts/giao_trinh/extract.py).
sys.path[0] = str(Path(__file__).resolve().parents[2])

import pymupdf  # noqa: E402

from vi_corpus.common.ocr_cli import add_cascade_args, build_getter  # noqa: E402
from vi_corpus.common.state import acquire_run_lock, setup_logging  # noqa: E402
from vi_corpus.giao_trinh.verify import collect_pages, pick_sample, summarize  # noqa: E402
from vi_corpus.ocr_bakeoff.cascade import EngineUnavailable, ServerDown  # noqa: E402
from vi_corpus.vjol.font_canary import word_overlap  # noqa: E402

log = logging.getLogger("vi_corpus")
OUT_DIR = "processed/giao_trinh_verify_flagged"
BATCH = 96  # số trang OCR mỗi lô (mỗi lô mở tối đa từng đó PDF cùng lúc)


def load_done(path: Path) -> list[dict]:
    """Đọc các dòng đã ghi trong pages.jsonl (để chạy lại thì làm tiếp)."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ocr_batch(ocr, data_root: Path, batch: list[dict]) -> list[dict]:
    """OCR một lô trang (vẽ lại cả trang) và trả về các dòng kết quả: tầng, trạng thái OCR, độ trùng từ với lớp chữ, đoạn đầu của hai văn bản."""
    docs: dict[str, pymupdf.Document] = {}
    try:
        pages = []
        for row in batch:
            doc = docs.get(row["path"]) or docs.setdefault(row["path"], pymupdf.open(data_root / row["path"]))
            pages.append(doc[row["page"]])
        results = ocr.ocr_pages(pages, render=[True] * len(pages))
    finally:
        for doc in docs.values():
            doc.close()
    out = []
    for row, res in zip(batch, results):
        text = res["text"] or ""
        out.append({"sha256": row["sha256"], "page": row["page"], "stratum": row["stratum"], "path": row["path"], "density": row["density"],
                    "status": res["status"], "tier": res["tier"],
                    "overlap": round(word_overlap(row["text"], text), 4) if text.strip() else None,
                    "layer_head": row["text"][:240], "ocr_head": text[:240]})
    return out


def main() -> int:
    """Chọn mẫu, OCR theo lô, ghi pages.jsonl và summary.json. Trả về 1 nếu chuỗi OCR hỏng."""
    parser = argparse.ArgumentParser(description="Kiểm chứng trang giáo trình bị bộ dò chữ hỏng gắn cờ bằng OCR.")
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục gốc dữ liệu.")
    parser.add_argument("--flagged-dir", type=Path, required=True, help="Checkpoint của lần trích có bật bộ dò chữ hỏng.")
    parser.add_argument("--before-dir", type=Path, required=True, help="Checkpoint của lần trích trước khi bật bộ dò.")
    parser.add_argument("--per-stratum", type=int, default=150, help="Số trang tối đa mỗi tầng (mặc định %(default)s).")
    parser.add_argument("--seed", type=int, default=0, help="Hạt giống chọn mẫu.")
    add_cascade_args(parser)
    args = parser.parse_args()
    out_dir = args.data_root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _lock = acquire_run_lock(args.data_root / "state" / "giao_trinh_verify_flagged")  # noqa: F841 — giữ khoá
    setup_logging(args.data_root / "logs", "giao_trinh_verify_flagged")

    sample = pick_sample(collect_pages(args.flagged_dir, args.before_dir), args.per_stratum, args.seed)
    pages_path = out_dir / "pages.jsonl"
    done = load_done(pages_path)
    seen = {(r["sha256"], r["page"]) for r in done}
    todo = [r for r in sample if (r["sha256"], r["page"]) not in seen]
    log.info("Mẫu %d trang (%d đã có kết quả, %d còn lại)", len(sample), len(done), len(todo))
    try:
        getter = build_getter(args, args.data_root, args.data_root / "state" / "giao_trinh_verify_flagged")
    except ValueError as e:
        parser.error(str(e))
    try:
        if todo:
            ocr = getter()
            for i in range(0, len(todo), BATCH):
                rows = ocr_batch(ocr, args.data_root, todo[i:i + BATCH])
                with pages_path.open("a", encoding="utf-8") as f:
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
                done += rows
                log.info("Đã OCR %d / %d trang", min(i + BATCH, len(todo)), len(todo))
    except (ServerDown, EngineUnavailable) as exc:
        log.error("%s. Dừng cả lượt; sửa server / môi trường rồi chạy lại đúng lệnh cũ (làm tiếp từ pages.jsonl).", exc, exc_info=True)
        return 1
    finally:
        getter.close()
    summary = summarize(done)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for stratum, s in summary.items():
        print(f"{stratum:13s} {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
