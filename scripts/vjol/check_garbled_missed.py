"""
Kiểm bước (c) của J-48: bao nhiêu bài có lớp chữ hỏng (độ trùng TỪ giữa lớp chữ và OCR dưới ngưỡng) mà bộ dò `garbled` hiện tại BỎ SÓT, tức trang vẫn là `text` trong checkpoint trích text.
Chạy bằng CPU, chỉ đọc: canary_pages.jsonl (do font_canary.py --dump-below ghi) và interim/vjol_text/<sha256>.json. Kết quả quyết định có đáng thêm ngưỡng OCR lại cả bài theo độ trùng từ hay không:
ít hoặc 0 bài bị bỏ sót thì ngưỡng mới ít giá trị; nhiều thì dùng làm cơ sở chọn số. Ghi <data-root>/processed/vjol_font_canary/missed_by_garbled.json.
    uv run python scripts/vjol/check_garbled_missed.py --data-root data [--threshold 0.3] [--pages <canary_pages.jsonl>]
Dòng canary_pages.jsonl cũ (trước J-48) chưa có "overlap" thì độ trùng tính từ chữ đã cắt 700 ký tự đầu (gần đúng); chạy lại font_canary.py --dump-below để có số tính trên cả trang.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common.state import read_json, write_json_atomic  # noqa: E402
from vi_corpus.vjol.extract import TEXT_DIR  # noqa: E402
from vi_corpus.vjol.font_canary import WORD_OVERLAP_LOW, missed_by_garbled  # noqa: E402

OUT_REL = "processed/vjol_font_canary"
SHOW_DOCS = 20  # số bài bị bỏ sót in ra


def load_page_methods(data_root: Path) -> dict[str, list[str]]:
    """{đường dẫn PDF (như font_canary.load_docs ghi): [method từng trang]} từ mọi checkpoint không lỗi, không bị bỏ qua; đường dẫn tương đối được thử cả khi nối với data_root."""
    out: dict[str, list[str]] = {}
    for ckpt in sorted((data_root / TEXT_DIR).glob("*.json")):
        done = read_json(ckpt)
        if not done or done.get("error") or done.get("skipped"):
            continue
        path = Path(done["paths"][0])
        path = path if path.is_file() else data_root / done["paths"][0]
        out[str(path)] = [p.get("method", "") for p in done["pages"]]
    return out


def main() -> int:
    """Đọc tham số, đối chiếu trang đã đo với checkpoint, in số bài bị bỏ sót và ghi missed_by_garbled.json. Trả về 1 nếu thiếu canary_pages.jsonl hoặc checkpoint."""
    p = argparse.ArgumentParser(description="Bài có lớp chữ hỏng (độ trùng từ thấp) mà bộ dò garbled bỏ sót (J-48 c).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    p.add_argument("--threshold", type=float, default=WORD_OVERLAP_LOW, help="Ngưỡng độ trùng từ (mặc định %(default)s, điểm khởi đầu của J-48).")
    p.add_argument("--pages", type=Path, default=None, help="canary_pages.jsonl (mặc định <data-root>/%s/canary_pages.jsonl)." % OUT_REL)
    args = p.parse_args()
    pages_path = args.pages or args.data_root / OUT_REL / "canary_pages.jsonl"
    if not pages_path.is_file():
        print(f"Không thấy {pages_path} (chạy font_canary.py --dump-below 0.7 trước)", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in pages_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    methods = load_page_methods(args.data_root)
    if not methods:
        print(f"Không có checkpoint nào trong {args.data_root / TEXT_DIR}", file=sys.stderr)
        return 1
    res = missed_by_garbled(rows, methods, args.threshold)
    write_json_atomic(args.data_root / OUT_REL / "missed_by_garbled.json", res)
    print(f"Ngưỡng độ trùng từ {res['threshold']}: {res['pages_measured']} trang đối chiếu được ({res['no_checkpoint']} trang không tìm thấy checkpoint), {res['docs_measured']} bài")
    print(f"  trang dưới ngưỡng: {res['pages_low']} ({res['docs_low']} bài); trong đó vẫn là `text` (garbled bỏ sót): {res['pages_low_text']} trang, {res['docs_missed']} bài")
    print("  phân bố độ trùng từ (số trang): " + ", ".join(f"{k} {v}" for k, v in res["histogram"].items()))
    for m in res["missed"][:SHOW_DOCS]:
        print(f"  BỎ SÓT bài {m['article_id']} [{m['producer']} / {m['creator']}] trang (độ trùng): " + ", ".join(f"{pg} ({ov})" for pg, ov in m["pages"]))
    print(f"Ghi {args.data_root / OUT_REL / 'missed_by_garbled.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
