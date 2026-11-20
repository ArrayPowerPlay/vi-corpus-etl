"""
Kiểm tra độ chính xác của lớp chữ giáo trình bằng OCR (GT-08): OCR lại một danh sách trang cho trước bằng chuỗi VLM (mỗi lần một GPU, chia phần bằng --shard)
rồi ghi kết quả đầy đủ để so với lớp chữ.

Khác scripts/giao_trinh/verify_flagged.py (chỉ kiểm trang bị bộ dò gắn cờ, tự chọn mẫu): tệp này nhận mẫu tuỳ ý từ `--sample-file` (mỗi dòng JSON có
"sha256", "path" tương đối so với data-root, "page", "stratum", "text" = lớp chữ), nên dùng được cho mọi câu hỏi kiểu "loại trang này có đúng không".
Ghi <out-dir>/pages_<phần>.jsonl, mỗi dòng gồm các trường của mẫu (trừ "text" dài, giữ lại dưới "layer") cộng "status", "tier", "ocr" (văn bản OCR đầy đủ);
chạy lại đúng lệnh cũ thì làm tiếp (bỏ qua trang đã có). Chuỗi OCR hỏng (server chết, engine không dựng được) dừng cả lượt, mã thoát 1.
Ví dụ (4 GPU, mỗi GPU một tiến trình, cổng 8100 + số GPU):
    for g in 0 1 2 3; do uv run --no-sync python scripts/giao_trinh/audit_pages.py --data-root data --sample-file mau.jsonl --out-dir data/processed/giao_trinh_audit \
        --shard $g/4 --gpu $g --port $((8100 + g)) --serve --vllm-bin $(which vllm) & done; wait
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
from vi_corpus.common.state import setup_logging  # noqa: E402
from vi_corpus.ocr_bakeoff.cascade import EngineUnavailable, ServerDown  # noqa: E402

log = logging.getLogger("vi_corpus")
BATCH = 96  # số trang OCR mỗi lô (mỗi lô mở tối đa từng đó PDF cùng lúc)


def read_jsonl(path: Path) -> list[dict]:
    """Đọc JSONL theo từng dòng của tệp (chỉ tách ở "\\n"); str.splitlines() cắt cả ở U+2028 / U+0085 nằm trong chuỗi JSON nên không dùng được."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_shard(text: str) -> tuple[int, int]:
    """Đọc "I/N" thành (I, N) với 0 <= I < N; sai khuôn dạng thì ném ValueError."""
    i, _, n = text.partition("/")
    shard, total = int(i), int(n)
    if not 0 <= shard < total:
        raise ValueError(f"--shard {text}: cần 0 <= I < N")
    return shard, total


def ocr_batch(ocr, data_root: Path, batch: list[dict]) -> list[dict]:
    """OCR một lô trang (vẽ lại cả trang) và trả về các dòng kết quả: các trường của mẫu (lớp chữ chuyển sang "layer") cộng trạng thái, engine, văn bản OCR."""
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
    return [{**{k: v for k, v in row.items() if k != "text"}, "layer": row["text"], "status": res["status"], "tier": res["tier"], "ocr": res["text"] or ""}
            for row, res in zip(batch, results)]


def main() -> int:
    """Chọn phần của mình trong mẫu, OCR theo lô, ghi pages_<phần>.jsonl. Trả về 1 nếu chuỗi OCR hỏng."""
    parser = argparse.ArgumentParser(description="OCR lại một mẫu trang giáo trình để đối chiếu với lớp chữ.")
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục gốc dữ liệu.")
    parser.add_argument("--sample-file", type=Path, required=True, help="JSONL mẫu trang (sha256, path, page, stratum, text).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Thư mục ghi kết quả.")
    parser.add_argument("--shard", default="0/1", help="Phần I/N của mẫu (mỗi GPU một phần), mặc định %(default)s.")
    add_cascade_args(parser)
    args = parser.parse_args()
    try:
        shard, total = parse_shard(args.shard)
    except ValueError as e:
        parser.error(str(e))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.data_root / "logs", f"giao_trinh_audit_{shard}of{total}")
    sample = read_jsonl(args.sample_file)
    mine = [r for k, r in enumerate(sample) if k % total == shard]
    out_path = args.out_dir / f"pages_{shard}of{total}.jsonl"
    seen = set()
    if out_path.is_file():
        seen = {(r["sha256"], r["page"]) for r in read_jsonl(out_path)}
    todo = [r for r in mine if (r["sha256"], r["page"]) not in seen]
    log.info("Phần %d/%d: %d trang (%d đã có, %d còn lại)", shard, total, len(mine), len(seen), len(todo))
    try:
        getter = build_getter(args, args.data_root, args.out_dir / f"state_{shard}of{total}")
    except ValueError as e:
        parser.error(str(e))
    try:
        if todo:
            ocr = getter()
            for i in range(0, len(todo), BATCH):
                rows = ocr_batch(ocr, args.data_root, todo[i:i + BATCH])
                with out_path.open("a", encoding="utf-8") as f:
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
                log.info("Đã OCR %d / %d trang", min(i + BATCH, len(todo)), len(todo))
    except (ServerDown, EngineUnavailable) as exc:
        log.error("%s. Dừng cả lượt; sửa server / môi trường rồi chạy lại đúng lệnh cũ (làm tiếp từ %s).", exc, out_path, exc_info=True)
        return 1
    finally:
        getter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
