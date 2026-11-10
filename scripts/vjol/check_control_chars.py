"""
Đo tỷ lệ ký tự điều khiển C0 của mọi trang `text` trong checkpoint `interim/vjol_text/` (J-53 bước 1 và 4), TRƯỚC khi thêm tín hiệu `control_chars` vào garbled.py.
Chạy bằng CPU, chỉ đọc. Trả lời: (a) phân bố tỷ lệ ký tự điều khiển của trang `text` (ngưỡng 20% có bắt oan trang tốt không), (b) ngưỡng đó sẽ đưa thêm bao nhiêu trang / bài đi OCR
(bài có >= GARBLED_DOC_SHARE số trang chữ vượt ngưỡng thì cả bài đi OCR), (c) khi có --pages: stage quality (mã `symbol_heavy` ...) có chặn các trang kiểu ký tự điều khiển không.
    uv run python scripts/vjol/check_control_chars.py --data-root data [--threshold 0.2] [--pages <canary_pages.jsonl>] [--dump 0.1]
Ghi <data-root>/processed/vjol_font_canary/control_chars.json. Không đổi checkpoint, không đổi ngưỡng nào của pipeline.
(c) dùng `layer` trong canary_pages.jsonl; dòng cũ chỉ có 700 ký tự đầu của lớp chữ nên số đo gần đúng.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common.garbled import GARBLED_DOC_SHARE, control_char_ratio  # noqa: E402
from vi_corpus.common.state import read_json, write_json_atomic  # noqa: E402
from vi_corpus.pipeline.config import PROFILES  # noqa: E402
from vi_corpus.pipeline.language import UNDETERMINED  # noqa: E402
from vi_corpus.pipeline.quality import band_of, compute_metrics, score_metrics  # noqa: E402
from vi_corpus.vjol.extract import TEXT_DIR  # noqa: E402

OUT_REL = "processed/vjol_font_canary"
DEFAULT_THRESHOLD = 0.2  # điểm khởi đầu của J-52 / J-53, chốt lại theo số đo
SHOW_DOCS = 20  # số bài in ra
# cạnh dưới của các cột biểu đồ (tỷ lệ ký tự điều khiển); cột đầu = đúng 0
BINS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def bin_label(ratio: float) -> str:
    """Tên cột của một tỷ lệ: "0" nếu đúng 0, ngược lại "(a,b]" theo BINS (cột cuối gồm cả 1.0)."""
    if ratio <= 0:
        return "0"
    for lo, hi in zip(BINS, BINS[1:]):
        if ratio <= hi:
            return f"({lo:g},{hi:g}]"
    return f"({BINS[-2]:g},{BINS[-1]:g}]"


def scan_checkpoints(data_root: Path, threshold: float) -> dict:
    """
    Duyệt checkpoint không lỗi, không bị bỏ qua; đo từng trang `text` có chữ.

    Returns:
        {"pages", "docs", "hist": {cột: số trang}, "pages_over", "docs_over" (có trang vượt ngưỡng), "docs_whole" (>= GARBLED_DOC_SHARE số trang chữ vượt ngưỡng: cả bài đi OCR),
        "top": [{"article_id", "path", "pages_over", "pages", "max_ratio", "producer"}] sắp theo số trang vượt giảm dần}.
    """
    hist: Counter = Counter()
    pages = docs = pages_over = 0
    over_docs: list[dict] = []
    for ckpt in sorted((data_root / TEXT_DIR).glob("*.json")):
        done = read_json(ckpt)
        if not done or done.get("error") or done.get("skipped"):
            continue
        docs += 1
        ratios = [control_char_ratio(p["text"])[0] for p in done["pages"] if p.get("method") == "text" and p["text"].strip()]
        pages += len(ratios)
        hist.update(bin_label(r) for r in ratios)
        n_over = sum(r >= threshold for r in ratios)
        pages_over += n_over
        if n_over:
            over_docs.append({"article_id": done.get("article_id"), "path": done["paths"][0], "pages_over": n_over, "pages": len(ratios), "max_ratio": round(max(ratios), 4)})
    over_docs.sort(key=lambda d: (-d["pages_over"], d["path"]))
    whole = sum(d["pages_over"] / d["pages"] >= GARBLED_DOC_SHARE for d in over_docs)
    order = ["0"] + [f"({lo:g},{hi:g}]" for lo, hi in zip(BINS, BINS[1:])]
    return {"pages": pages, "docs": docs, "hist": {k: hist[k] for k in order}, "pages_over": pages_over, "docs_over": len(over_docs), "docs_whole": whole, "top": over_docs}


def dump_pages(data_root: Path, low: float, out: Path, head: int = 400) -> int:
    """
    Ghi các trang `text` có tỷ lệ ký tự điều khiển >= `low` ra JSONL để đọc tay (vùng 0,1-0,3 chưa biết chữ tốt hay hỏng): bài, trang, tỷ lệ, producer / creator của PDF (nếu mở được) và `head` ký tự đầu của lớp chữ (ensure_ascii nên ký tự điều khiển hiện rõ dạng \\u00xx).

    Returns:
        Số trang đã ghi.
    """
    import pymupdf  # chỉ cần khi --dump

    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ckpt in sorted((data_root / TEXT_DIR).glob("*.json")):
            done = read_json(ckpt)
            if not done or done.get("error") or done.get("skipped"):
                continue
            hits = [(i, control_char_ratio(p["text"])[0]) for i, p in enumerate(done["pages"]) if p.get("method") == "text" and p["text"].strip()]
            hits = [(i, r) for i, r in hits if r >= low]
            if not hits:
                continue
            meta = {}
            try:
                path = Path(done["paths"][0])
                with pymupdf.open(path if path.is_file() else data_root / path) as doc:
                    meta = doc.metadata or {}
            except Exception:  # noqa: BLE001 - chỉ là thông tin phụ
                pass
            for i, r in hits:
                f.write(json.dumps({"article_id": done.get("article_id"), "page": i, "ratio": round(r, 4), "producer": meta.get("producer"), "creator": meta.get("creator"),
                                    "text": done["pages"][i]["text"][:head]}, ensure_ascii=True) + "\n")
                n += 1
    return n


def check_quality(pages_path: Path, threshold: float) -> dict:
    """
    (J-53 bước 4) Chấm các trang của canary_pages.jsonl bằng stage quality (hồ sơ vjol) và đếm mã lý do, riêng nhóm có tỷ lệ ký tự điều khiển >= `threshold`.

    Returns:
        {"pages", "pages_ctrl" (trang >= ngưỡng), "ctrl_band": {band: số trang}, "ctrl_reasons": {mã: số trang}, "ctrl_not_hard": số trang kiểu ký tự điều khiển KHÔNG bị band D (lọt vào kho)}.
        Chấm cả trang như một văn bản, ngôn ngữ truyền "und" (bỏ mã lang_not_allowed để mã khác lộ ra); chỉ để xem mã chất lượng bắt được hay không.
    """
    profile = PROFILES["vjol"]
    # tách theo "\n" chứ không dùng splitlines(): lớp chữ PDF có thể chứa  , \x85... nằm TRONG chuỗi JSON
    rows = [json.loads(line) for line in pages_path.read_text(encoding="utf-8").split("\n") if line.strip()]
    bands: Counter = Counter()
    reasons: Counter = Counter()
    n_ctrl = not_hard = 0
    for r in rows:
        text = r.get("layer", "")
        if control_char_ratio(text)[0] < threshold:
            continue
        n_ctrl += 1
        score, codes = score_metrics(compute_metrics(text), UNDETERMINED, profile)
        band = band_of(score)
        bands[band] += 1
        reasons.update(codes)
        not_hard += band != "D"
    return {"pages": len(rows), "pages_ctrl": n_ctrl, "ctrl_band": dict(sorted(bands.items())), "ctrl_reasons": dict(reasons.most_common()), "ctrl_not_hard": not_hard}


def main() -> int:
    """Đọc tham số, đo, in kết quả và ghi control_chars.json. Trả về 1 nếu không có checkpoint hoặc --pages không tồn tại."""
    p = argparse.ArgumentParser(description="Phân bố tỷ lệ ký tự điều khiển của trang text (J-53).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Ngưỡng tỷ lệ ký tự điều khiển (mặc định %(default)s).")
    p.add_argument("--pages", type=Path, default=None, help="canary_pages.jsonl để kiểm stage quality (J-53 bước 4); bỏ qua nếu không đưa.")
    p.add_argument("--dump", type=float, default=None, metavar="LOW", help="Ghi trang có tỷ lệ >= LOW vào control_chars_pages.jsonl để đọc tay (vd 0.1).")
    args = p.parse_args()
    res = scan_checkpoints(args.data_root, args.threshold)
    if not res["docs"]:
        print(f"Không có checkpoint nào trong {args.data_root / TEXT_DIR}", file=sys.stderr)
        return 1
    res["threshold"] = args.threshold
    print(f"{res['docs']} bài, {res['pages']} trang `text` có chữ. Ngưỡng {args.threshold:g}:")
    for k, v in res["hist"].items():
        print(f"  {k:>12}: {v}")
    print(f"  trang vượt ngưỡng (sẽ thêm đi OCR): {res['pages_over']}; bài có trang vượt: {res['docs_over']}; trong đó cả bài đi OCR (>= {GARBLED_DOC_SHARE:.0%} số trang chữ): {res['docs_whole']}")
    for d in res["top"][:SHOW_DOCS]:
        print(f"  bài {d['article_id']}: {d['pages_over']}/{d['pages']} trang, tối đa {d['max_ratio']}  {d['path']}")
    if args.pages:
        if not args.pages.is_file():
            print(f"Không thấy {args.pages}", file=sys.stderr)
            return 1
        q = check_quality(args.pages, args.threshold)
        res["quality"] = q
        print(f"Stage quality trên {q['pages_ctrl']}/{q['pages']} trang canary có tỷ lệ >= {args.threshold:g}: band {q['ctrl_band']}; KHÔNG bị band D (lọt vào kho): {q['ctrl_not_hard']}")
        print(f"  mã lý do: {q['ctrl_reasons']}")
    if args.dump is not None:
        dump = args.data_root / OUT_REL / "control_chars_pages.jsonl"
        print(f"Ghi {dump_pages(args.data_root, args.dump, dump)} trang vào {dump}")
    out = args.data_root / OUT_REL / "control_chars.json"
    write_json_atomic(out, res)
    print(f"Ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
