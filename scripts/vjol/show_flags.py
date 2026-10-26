"""
Xem các đoạn bị bộ dò gắn cờ NHẦM hoặc BỎ SÓT trên nhãn chuẩn VJOL (J-46): evaluate chỉ cho số (tp / fp / fn); script này in chữ cụ thể để biết vì sao một bộ dò sai và sửa đúng chỗ. Chỉ CPU.

Mỗi bộ dò được in: dương tính giả (đoạn có nhãn ngoài nhóm đích của bộ dò mà vẫn bị gắn cờ, kèm nhãn) với `--kind fp`, hoặc bỏ sót (đoạn có nhãn đích mà không bị gắn cờ) với `--kind fn`.
Với joined_words / tone_shift in thêm các TỪ trúng (không có định vị chữ nên in từ, không in đoạn); các bộ dò khác in đoạn quanh chỗ trúng (audit.excerpt_of).
    uv run python scripts/vjol/show_flags.py --labels <run>/groundtruth/labeling_sample.csv --data-root data --detectors joined_words,mid_header,tone_shift [--kind fp] [--max 15]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common import defects as Df  # noqa: E402
from vi_corpus.common.garbled import load_vocab  # noqa: E402
from vi_corpus.vjol.audit import DETECTOR_TARGETS, build_detectors, excerpt_of, refine_label  # noqa: E402
from vi_corpus.vjol.extract import SYLLABLES_REL  # noqa: E402


def main() -> int:
    """Đọc nhãn, chạy các bộ dò được chọn và in các đoạn sai (fp hoặc fn). Trả về 1 nếu tên bộ dò không có."""
    p = argparse.ArgumentParser(description="In các đoạn bộ dò gắn cờ nhầm / bỏ sót trên nhãn chuẩn VJOL.")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    p.add_argument("--detectors", required=True, help="Tên bộ dò, cách nhau dấu phẩy (xem evaluate).")
    p.add_argument("--kind", choices=["fp", "fn"], default="fp")
    p.add_argument("--max", type=int, default=15, help="Số đoạn tối đa mỗi bộ dò.")
    args = p.parse_args()
    vocab = load_vocab(args.data_root / SYLLABLES_REL)
    tone_map = Df.load_tone_shift_map(args.data_root / Df.TONE_SHIFT_REL)
    dets = build_detectors(vocab, tone_map)
    with open(args.labels, encoding="utf-8", newline="") as f:
        rows = [{**r, "label": refine_label(r)} for r in csv.DictReader(f) if r["label"]]
    for name in [n.strip() for n in args.detectors.split(",") if n.strip()]:
        if name not in dets:
            print(f"Không có bộ dò {name!r}; có: {sorted(dets)}", file=sys.stderr)
            return 1
        targets = DETECTOR_TARGETS.get(name, frozenset({name}))
        shown = 0
        print(f"\n===== {name} ({args.kind}) =====")
        for r in rows:
            hit = bool(dets[name](json.loads(r.get("quality_metrics") or "{}"), r["text"]))
            wrong = hit and r["label"] not in targets if args.kind == "fp" else (not hit) and r["label"] in targets
            if not wrong or shown >= args.max:
                continue
            shown += 1
            words = {"joined_words": Df.joined_words_found, "tone_shift": Df.tone_shift_found}.get(name)
            what = f"từ trúng: {(words(r['text'], vocab) if name == 'joined_words' else words(r['text'], vocab, tone_map))[:12]}" if words and vocab else excerpt_of(r["text"], name)
            print(f"- {r['doc_id']} nhãn={r['label']} ghi chú={r.get('note', '')!r}\n    {what}")
        print(f"({shown} đoạn in)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
