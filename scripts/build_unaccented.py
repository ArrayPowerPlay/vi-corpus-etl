"""
Sinh danh sách từ không dấu cho tín hiệu "nodiac" / mã lost_diacritics (J-25 A) từ kho chữ sạch SEA-PILE v2. Chỉ dùng CPU.

Đọc tối đa --max-files file parquet trong <data-root>/raw/sea_vi/sea_pile_v2 (rải đều theo thứ tự tên), dừng sau --max-docs văn bản, rồi ghi
<data-root>/processed/ocr_bakeoff/unaccented.json cạnh syllables.json. Cách chọn từ: vi_corpus/common/unaccented.py. In 40 từ đầu và các từ cũ bị loại ("doanh", "cung")
để kiểm bằng mắt. Ngưỡng --min-accented tính theo cỡ mẫu (500 cho khoảng 300 nghìn văn bản); mẫu nhỏ hơn thì hạ ngưỡng tương ứng.
Ví dụ:
    uv run python scripts/build_unaccented.py --data-root /duong/dan/data
"""

import argparse
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

import pyarrow.parquet as pq  # noqa: E402

from vi_corpus.common.garbled import UNACCENTED_WORDS  # noqa: E402
from vi_corpus.common.unaccented import MAX_BARE_SHARE, MIN_ACCENTED, TOP_WORDS, build_unaccented, save_unaccented  # noqa: E402


def iter_texts(files: list[Path], text_col: str, max_docs: int):
    """Văn bản của các file parquet theo lô (không nạp cả file), dừng sau max_docs văn bản."""
    docs = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=[text_col]):
            for text in batch.column(0).to_pylist():
                yield text or ""
                docs += 1
                if docs >= max_docs:
                    return


def main() -> int:
    """Đọc tham số, đếm, ghi file và in kết quả để kiểm. Trả về 1 nếu danh sách rỗng."""
    p = argparse.ArgumentParser(description="Sinh danh sách từ không dấu từ SEA-PILE v2 (J-25 A).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục gốc dữ liệu.")
    p.add_argument("--source-dir", type=Path, default=None, help="Thư mục parquet (mặc định <data-root>/raw/sea_vi/sea_pile_v2).")
    p.add_argument("--out", type=Path, default=None, help="File kết quả (mặc định <data-root>/processed/ocr_bakeoff/unaccented.json).")
    p.add_argument("--max-files", type=int, default=10)
    p.add_argument("--max-docs", type=int, default=300_000)
    p.add_argument("--min-accented", type=int, default=MIN_ACCENTED, help="Số lần tối thiểu của dạng có dấu.")
    p.add_argument("--max-bare-share", type=float, default=MAX_BARE_SHARE, help="Tỷ lệ dạng trần tối đa.")
    p.add_argument("--top", type=int, default=TOP_WORDS)
    args = p.parse_args()
    src = args.source_dir or args.data_root / "raw/sea_vi/sea_pile_v2"
    files = sorted(src.rglob("*.parquet"))
    if not files:
        p.error(f"Không thấy file parquet trong {src}")
    files = files[:: max(1, len(files) // args.max_files)][: args.max_files]
    names = pq.ParquetFile(files[0]).schema_arrow.names
    text_col = "text" if "text" in names else next((n for n in names if "text" in n.lower()), None)
    if text_col is None:
        p.error(f"Không thấy cột văn bản trong {files[0]} (các cột: {names})")
    res = build_unaccented(iter_texts(files, text_col, args.max_docs), args.min_accented, args.max_bare_share, args.top)
    out = args.out or args.data_root / "processed/ocr_bakeoff/unaccented.json"
    save_unaccented(out, res)
    words = res["words"]
    print(f"{len(words)} từ từ {res['docs']} văn bản đủ chữ có dấu -> {out}")
    top = sorted(res["counts"].items(), key=lambda kv: -kv[1][0])[:40]
    print("40 từ đầu (khoá: có dấu / trần):", ", ".join(f"{k}: {a}/{b}" for k, (a, b) in top))
    print("Từ trong danh sách tối thiểu cũ nhưng KHÔNG nằm trong danh sách mới:", sorted(UNACCENTED_WORDS - set(words)) or "(không có)")
    print("Cũ bị loại oan trước đây, phải KHÔNG có trong danh sách mới:", [w for w in ("doanh", "cung") if w in words] or "đúng, không có")
    return 0 if words else 1


if __name__ == "__main__":
    sys.exit(main())
