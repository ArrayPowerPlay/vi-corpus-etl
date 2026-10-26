"""
Bộ đáp án và tập chữ đúng cho VJOL (J-25 Bước 0 (a), (b)). Chỉ CPU.

    # (a) lập ~400 đoạn để ĐỌC VÀ GÁN NHÃN (cột `label` bằng một trong: clean, lost_diacritics, hyphen_font, font_misread, old_ocr, reference, running_header, other; nhãn con của other: font_shift, dropped_letters, joined_words, formula, chart_label, layout, garbage), chia hai nửa theo bài
    uv run python scripts/vjol/groundtruth.py sample --run-dir data/processed/pipeline_runs/vjol_10k_v3 --n 400
    # sau khi điền nhãn: đo độ chính xác / bao phủ từng bộ dò, tách nửa tune và holdout (J-42: thêm "all" gộp, nhãn `other` chia nhỏ theo cột `note`, cờ qua cổng precision >= 0,9; --data-root cho tone_shift / joined_words)
    uv run python scripts/vjol/groundtruth.py evaluate --labels data/processed/pipeline_runs/vjol_10k_v3/groundtruth/labeling_sample.csv --data-root data
    # (b) lập tập chữ đúng (kinh tế / luật / y học có dấu đầy đủ) và kiểm bộ dò không gắn cờ lên nó
    uv run python scripts/vjol/groundtruth.py clean-set --run-dir data/processed/pipeline_runs/vjol_10k_v3 --n 300
    uv run python scripts/vjol/groundtruth.py check-clean --clean-set data/processed/pipeline_runs/vjol_10k_v3/groundtruth/clean_set.jsonl --data-root data
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common.defects import TONE_SHIFT_REL, load_tone_shift_map  # noqa: E402
from vi_corpus.common.garbled import load_unaccented, load_vocab  # noqa: E402
from vi_corpus.pipeline import quality as Q  # noqa: E402
from vi_corpus.pipeline.io import read_rows  # noqa: E402
from vi_corpus.vjol.audit import check_clean_set, evaluate_labels, make_clean_set, make_labeling_sample, write_labeling_sample  # noqa: E402
from vi_corpus.vjol.extract import SYLLABLES_REL, UNACCENTED_REL  # noqa: E402


def _rows(run_dir: Path) -> list[dict]:
    """Bản ghi vjol đã qua stage quality (04_quality.parquet) của một lần chạy."""
    return [r for r in read_rows(run_dir / "04_quality.parquet") if r["source_key"] == "vjol"]


def main() -> int:
    """Chạy lệnh con sample | evaluate | clean-set | check-clean. Trả về 1 nếu check-clean tìm thấy cờ nhầm hoặc thiếu dữ liệu."""
    p = argparse.ArgumentParser(description="Bộ đáp án và tập chữ đúng cho VJOL (J-25 Bước 0).")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="Lập labeling_sample.csv / .md để gán nhãn.")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--n", type=int, default=400)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=None, help="Mặc định <run-dir>/groundtruth.")
    e = sub.add_parser("evaluate", help="Đo bộ dò trên các dòng đã gán nhãn.")
    e.add_argument("--labels", type=Path, required=True)
    e.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Để đọc tập âm tiết và bảng lệch dấu (cho tone_shift / joined_words).")
    c = sub.add_parser("clean-set", help="Lập tập chữ đúng clean_set.jsonl.")
    c.add_argument("--run-dir", type=Path, required=True)
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", type=Path, default=None)
    k = sub.add_parser("check-clean", help="Chạy bộ dò lên tập chữ đúng; phải không có cờ nào.")
    k.add_argument("--clean-set", type=Path, required=True)
    k.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    args = p.parse_args()

    if args.cmd == "sample":
        out = args.out or args.run_dir / "groundtruth"
        items = make_labeling_sample(_rows(args.run_dir), args.n, args.seed)
        write_labeling_sample(items, out)
        print(f"Ghi {len(items)} đoạn vào {out} (tune {sum(i['split'] == 'tune' for i in items)}, holdout {sum(i['split'] == 'holdout' for i in items)})")
        return 0
    if args.cmd == "evaluate":
        with open(args.labels, encoding="utf-8", newline="") as f:
            items = list(csv.DictReader(f))
        res = evaluate_labels(items, load_vocab(args.data_root / SYLLABLES_REL), load_tone_shift_map(args.data_root / TONE_SHIFT_REL))
        print(json.dumps(res, ensure_ascii=False, indent=2))
        eligible = [n for n, d in res["detectors"].items() if d["gate_ok"]]
        print(f"# bộ dò qua cổng precision >= {Q.DETECTOR_GATE_PRECISION} (có thể đặt ngưỡng loại ở quality.HARD_DETECTOR_THRESHOLDS): {eligible or 'không có'}", file=sys.stderr)
        return 0
    if args.cmd == "clean-set":
        out = args.out or args.run_dir / "groundtruth" / "clean_set.jsonl"
        items = make_clean_set(_rows(args.run_dir), args.n, args.seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
        print(f"Ghi {len(items)} đoạn chữ đúng vào {out}")
        return 0 if items else 1
    items = [json.loads(ln) for ln in args.clean_set.read_text(encoding="utf-8").split("\n") if ln.strip()]
    bad = check_clean_set(items, load_vocab(args.data_root / SYLLABLES_REL), load_unaccented(args.data_root / UNACCENTED_REL))
    print(f"{len(items)} đoạn chữ đúng, {len(bad)} bị gắn cờ nhầm")
    for b in bad[:30]:
        print(" ", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
