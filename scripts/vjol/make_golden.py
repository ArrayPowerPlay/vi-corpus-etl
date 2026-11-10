"""
Sinh ca thử vàng từ nhãn chuẩn VJOL (J-42 (1e), P7): đọc labeling_sample.csv đã gán nhãn, ghi các ca "bug" (đoạn rút ngắn quanh chỗ lỗi, đã kiểm bộ dò THỰC SỰ bắt được) và ca "keep" (đoạn nhãn
`clean` rút ngắn, đã kiểm không bộ dò nào gắn cờ) vào tests/data/vjol_golden.jsonl (xem tests/test_vjol_golden.py). Chỉ CPU. Ca sinh ra có id bắt đầu bằng "gt:"; chạy lại thì thay các ca "gt:" cũ, giữ
nguyên ca viết tay (hạt giống). Chỉ rút các câu cần thiết (--max-chars), không chép cả đoạn dài.
    uv run python scripts/vjol/make_golden.py --labels <run>/groundtruth/labeling_sample.csv --data-root data [--out tests/data/vjol_golden.jsonl] [--max-chars 600]
Nhãn -> bộ dò kỳ vọng: hyphen_font -> hyphen_font, running_header -> running_header, font_shift -> tone_shift, dropped_letters -> dropped_letters, joined_words -> joined_words, garbage -> gibberish.
Các nhãn còn lại (lost_diacritics, font_misread, old_ocr, reference ...) không có bộ dò đoạn ngắn đủ tin nên không thành ca "bug".
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common.defects import TONE_SHIFT_MAP, TONE_SHIFT_REL, load_tone_shift_map  # noqa: E402
from vi_corpus.common.garbled import latin_tokens, load_vocab  # noqa: E402
from vi_corpus.vjol.audit import HIT_LOCATORS, detect_all, refine_label  # noqa: E402
from vi_corpus.vjol.extract import SYLLABLES_REL  # noqa: E402

LABEL_DETECTORS = {"hyphen_font": "hyphen_font", "running_header": "running_header", "font_shift": "tone_shift", "dropped_letters": "dropped_letters", "joined_words": "joined_words",
                   "garbage": "gibberish"}
FORBID_ALL = frozenset({"dropped_letters", "gibberish", "joined_words", "tone_shift", "mid_header", "running_header", "hyphen_font", "u_split"})  # bộ dò không được gắn cờ lên ca "keep" sinh ra
DEFAULT_OUT = Path("tests/data/vjol_golden.jsonl")
GT_PREFIX = "gt:"


def excerpt(text: str, detector: str, max_chars: int) -> str:
    """Đoạn rút gọn của `text` dài tối đa `max_chars` ký tự, quanh chỗ lỗi nếu bộ dò có định vị (HIT_LOCATORS), không thì từ đầu; cắt ở khoảng trắng."""
    loc = HIT_LOCATORS.get(detector, lambda t: None)(text)
    start = 0 if loc is None else max(0, loc[0] - max_chars // 2)
    cut = text[start:start + max_chars]
    return cut if start == 0 else cut.partition(" ")[2] or cut


def needed_words(text: str, vocab: frozenset[str], table: dict[str, str]) -> set[str]:
    """Âm tiết trong `vocab` mà ca cần để bộ dò chạy y như với tập thật: từ của đoạn, dạng đã sửa lệch dấu và hai phần của từ dính liền."""
    out: set[str] = set()
    for tok in latin_tokens(text):
        low = tok.lower()
        if low in vocab:
            out.add(low)
            continue
        out |= {low[:i] + table[c] + low[i + 1:] for i, c in enumerate(low) if c in table and low[:i] + table[c] + low[i + 1:] in vocab}
        for i in range(2, len(low) - 1):
            if low[:i] in vocab and low[i:] in vocab:
                out |= {low[:i], low[i:]}
    return out


def main() -> int:
    """Đọc nhãn, sinh ca, ghi tệp. Trả về 1 nếu không sinh được ca nào."""
    p = argparse.ArgumentParser(description="Sinh ca thử vàng từ nhãn chuẩn VJOL (J-42 (1e)).")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data"), help="Để đọc tập âm tiết thật và bảng lệch dấu.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-chars", type=int, default=600)
    args = p.parse_args()
    vocab = load_vocab(args.data_root / SYLLABLES_REL)
    if vocab is None:
        print(f"Thiếu {args.data_root / SYLLABLES_REL}", file=sys.stderr)
        return 1
    table = load_tone_shift_map(args.data_root / TONE_SHIFT_REL) or TONE_SHIFT_MAP
    with open(args.labels, encoding="utf-8", newline="") as f:
        rows = [{**r, "label": refine_label(r)} for r in csv.DictReader(f)]
    cases, words, skipped = [], set(), {"bug_not_caught": 0, "clean_flagged": 0, "no_detector": 0}
    for r in rows:
        label, text = r["label"], r["text"]
        if label == "clean":
            text = excerpt(text, "", args.max_chars)
            got = detect_all(text, vocab, table)
            if got:
                skipped["clean_flagged"] += 1
                continue
            case = {"id": f"{GT_PREFIX}{r['doc_id']}", "kind": "keep", "source": "nhãn chuẩn: clean", "forbid": sorted(FORBID_ALL), "text": text}
        elif label in LABEL_DETECTORS:
            det = LABEL_DETECTORS[label]
            text = excerpt(text, det, args.max_chars)
            if det not in detect_all(text, vocab, table):
                skipped["bug_not_caught"] += 1
                continue
            case = {"id": f"{GT_PREFIX}{r['doc_id']}", "kind": "bug", "source": f"nhãn chuẩn: {label}", "detectors": [det], "text": text}
        else:
            skipped["no_detector"] += 1
            continue
        words |= needed_words(text, vocab, table)
        cases.append(case)
    kept = []
    if args.out.is_file():
        for ln in args.out.read_text(encoding="utf-8").splitlines():
            rec = json.loads(ln) if ln.strip() else None
            if rec and not str(rec.get("id", "")).startswith(GT_PREFIX) and rec.get("_source") != "gt":
                kept.append(rec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in kept + ([{"_vocab": " ".join(sorted(words)), "_source": "gt"}] if cases else []) + cases:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    by_kind = {k: sum(c["kind"] == k for c in cases) for k in ("bug", "keep")}
    print(f"Ghi {len(cases)} ca ({by_kind}) + {len(kept)} dòng giữ nguyên vào {args.out}; bỏ qua {skipped}")
    return 0 if cases else 1


if __name__ == "__main__":
    sys.exit(main())
