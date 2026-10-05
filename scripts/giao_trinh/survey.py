"""
Khảo sát nhanh giáo trình đã tải về (mặc định <data-root>/raw/giao_trinh; dùng --root cho nguồn khác,
vd VJOL: --root <data-root>/raw/VJOL, khi đó cột "ngành" là thư mục cấp 1 = năm) trước khi viết bước xử lý.

Với mỗi PDF, lấy tối đa --max-pages trang rải đều và phân loại từng trang như extract_pdf sẽ làm:
text / tcvn3 / scan (cần OCR) / empty. Ngoài ra đo tỉ lệ chữ có dấu tiếng Việt (để tách sách
tiếng Anh) và tỉ lệ ký tự lỗi (U+FFFD, vùng private-use: dấu hiệu font mã hoá lạ như VNI).
Không OCR, không sửa dữ liệu; chỉ ghi CSV + in bảng tóm tắt theo ngành.
Ví dụ:
    uv run --group ocr python scripts/giao_trinh/survey.py --data-root /duong/dan/data
    uv run --group ocr python scripts/giao_trinh/survey.py --data-root /duong/dan/data --per-dir 5
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # để import được vi_corpus

from vi_corpus.common.pdf_text import MIN_TEXT_CHARS, is_tcvn3, page_text  # noqa: E402

# Chữ cái chỉ có trong tiếng Việt (không kể a-z thường): dùng để ước lượng "có phải tiếng Việt".
_VI_CHARS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")


def profile_pdf(path: Path, max_pages: int) -> dict:
    """
    Đo một PDF, trả về một dòng thống kê (dict). Lỗi mở file được ghi vào cột `error`, không ném ra.

    Trang được lấy rải đều tối đa `max_pages` trang để sách nghìn trang vẫn chạy nhanh.
    """
    row = {"path": str(path), "size_mb": round(path.stat().st_size / 2**20, 1), "pages": 0,
           "error": "", "producer": "", "text": 0, "tcvn3": 0, "scan": 0, "empty": 0, "vi_ratio": 0.0, "bad_ratio": 0.0}
    try:
        with pymupdf.open(path) as doc:
            if doc.is_repaired:
                row["error"] = "repaired"
            if doc.needs_pass:
                row["error"] = "password"
                return row
            row["pages"] = len(doc)
            row["producer"] = (doc.metadata.get("creator") or doc.metadata.get("producer") or "")[:40]
            step = max(1, len(doc) // max_pages)
            texts, kinds = [], []
            for i in range(0, len(doc), step):
                page = doc[i]
                t = page_text(page)
                texts.append(t)
                n = len("".join(t.split()))
                kinds.append("empty" if n == 0 and not page.get_images()
                             else "scan" if n < MIN_TEXT_CHARS and page.get_images() else "text")
    except Exception as e:  # noqa: BLE001 - khảo sát: file hỏng chỉ ghi lại, không dừng
        row["error"] = f"{type(e).__name__}: {e}"[:80]
        return row
    joined = "".join(texts)
    letters = [c for c in joined if c.isalpha()]
    if is_tcvn3(joined):
        kinds = ["tcvn3" if k == "text" else k for k in kinds]
    for k, v in Counter(kinds).items():
        row[k] = v
    if letters:
        row["vi_ratio"] = round(sum(c in _VI_CHARS for c in letters) / len(letters), 3)
    if joined:
        row["bad_ratio"] = round(sum(c == "�" or "" <= c <= "" for c in joined) / len(joined), 4)
    return row


def main() -> int:
    """Duyệt thư mục giáo trình, đếm định dạng file, profile PDF, ghi CSV và in tóm tắt theo ngành."""
    parser = argparse.ArgumentParser(description="Khảo sát giáo trình đã tải.")
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    parser.add_argument("--root", type=Path, default=None, help="Mặc định <data-root>/raw/giao_trinh.")
    parser.add_argument("--max-pages", type=int, default=40, help="Số trang tối đa đo mỗi PDF.")
    parser.add_argument("--per-dir", type=int, default=None, help="Chỉ đo N PDF mỗi ngành (chạy thử).")
    args = parser.parse_args()
    root = args.root or args.data_root / "raw" / "giao_trinh"
    if not root.is_dir():
        parser.error(f"Không thấy thư mục {root}")

    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts)
    print("Định dạng:", dict(Counter(p.suffix.lower() or "(không đuôi)" for p in files)))
    by_dir = defaultdict(list)
    for p in files:
        if p.suffix.lower() == ".pdf":
            by_dir[p.relative_to(root).parts[0]].append(p)

    rows = []
    for d, pdfs in sorted(by_dir.items()):
        for p in pdfs[: args.per_dir]:
            rows.append({"nganh": d, **profile_pdf(p, args.max_pages)})
            print(f"\r{len(rows)} PDF", end="", flush=True)
    print()
    out = args.data_root / "interim" / "giao_trinh_profile.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'ngành':45} {'pdf':>5} {'text%':>6} {'tcvn3%':>7} {'scan%':>6} {'tiếng Việt%':>11} {'lỗi':>4}")
    for d in sorted({r["nganh"] for r in rows}):
        rs = [r for r in rows if r["nganh"] == d]
        pages = sum(r["text"] + r["tcvn3"] + r["scan"] + r["empty"] for r in rs) or 1
        vi = sum(r["vi_ratio"] > 0.03 for r in rs)  # >3% chữ có dấu riêng của tiếng Việt
        print(f"{d[:45]:45} {len(rs):>5} {100*sum(r['text'] for r in rs)/pages:>6.0f} "
              f"{100*sum(r['tcvn3'] for r in rs)/pages:>7.0f} {100*sum(r['scan'] for r in rs)/pages:>6.0f} "
              f"{100*vi/len(rs):>11.0f} {sum(bool(r['error']) for r in rs):>4}")
    print("Phần mềm tạo PDF (creator/producer):", dict(Counter(r["producer"] or "(không có)" for r in rows).most_common(8)))
    print("Chi tiết từng file:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
