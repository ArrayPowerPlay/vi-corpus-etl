"""
Đo bộ chọn cách đọc của chỗ "ph-ơng / l-ơng" mơ hồ (J-57) bằng chữ ĐÚNG có sẵn, không cần gán nhãn tay.
Lấy các đoạn không có mã lỗi trong 04_quality.parquet của một lần chạy pipeline, tìm các từ "X + ư + ơ/ờ/ớ/ở/ỡ/ợ + ng/c/i/n" (phương, trường, hướng, người ...), phá thành "X-ơ..." (mất
chữ ư và dấu thanh, đúng như font hỏng), cho `LmChooser` chọn lại và đếm: đúng / sai / bỏ qua (chưa đủ chắc) / không mơ hồ. Quét nhiều `--margin` để chọn ngưỡng: cần "sai" gần 0.
    uv run --group curator python scripts/vjol/eval_hyphen_lm.py --run-dir data/processed/pipeline_runs/vjol_10k_v9 [--model Qwen/Qwen3-0.6B] [--device cuda] [--spots 600] [--margins 1,2,3,4]
Ghi <run-dir>/hyphen_lm_eval.json. Chỉ đọc, không đổi dữ liệu.
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

import pyarrow.parquet as pq  # noqa: E402

from vi_corpus.vjol.hyphen_fix import CONTEXT_LEFT, CONTEXT_RIGHT, Ambiguous  # noqa: E402
from vi_corpus.vjol.hyphen_lm import LmChooser  # noqa: E402

_SPOT = re.compile(r"(?<![\w-])([^\W\d_]{1,3})ư([ơờớởỡợ])(ng|c|i|n)(?![^\W\d_])", re.I)
MIN_VOCAB_COUNT = 5  # âm tiết xuất hiện ít nhất ngần này lần trong đoạn sạch mới vào tập âm tiết thử


def build_spots(texts: list[str], n: int, seed: int) -> list[tuple[str, str, str]]:
    """Chọn ngẫu nhiên tối đa `n` chỗ (ngữ cảnh trái, từ đúng, ngữ cảnh phải) từ các đoạn sạch."""
    rng = random.Random(seed)
    spots = []
    for t in rng.sample(texts, len(texts)):
        for m in _SPOT.finditer(t):
            spots.append((t[max(0, m.start() - CONTEXT_LEFT):m.start()], m.group(0), t[m.end():m.end() + CONTEXT_RIGHT]))
        if len(spots) >= n * 3:
            break
    return rng.sample(spots, min(n, len(spots)))


def main() -> int:
    """Chạy đo và in bảng đúng / sai / bỏ qua theo từng ngưỡng `margin`."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--spots", type=int, default=600)
    ap.add_argument("--margins", default="1,2,3,4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pq.read_table(args.run_dir / "04_quality.parquet", columns=["text", "reason_codes", "language"]).to_pylist()
    texts = [unicodedata.normalize("NFC", r["text"]) for r in df if not list(r["reason_codes"] or []) and r["language"] == "vi"]
    counts: Counter = Counter()
    for t in texts:
        counts.update(re.findall(r"[^\W\d_]+", t.lower()))
    vocab = frozenset(w for w, c in counts.items() if c >= MIN_VOCAB_COUNT)
    spots = build_spots(texts, args.spots, args.seed)
    print(f"{len(texts)} đoạn sạch, {len(vocab)} âm tiết, {len(spots)} chỗ thử")

    # chấm một lần ở margin nhỏ nhất rồi suy ra các margin lớn hơn từ điểm: dùng một chooser margin=0 và đọc chênh lệch qua score()
    chooser = LmChooser(args.model, args.device, margin=0.0)
    rows = []
    for left, truth, right in spots:
        first = truth.lower().index("ư")
        broken = Ambiguous(truth[:first] + "ư" + "ơ" + truth[first + 2:])
        cands = broken.alternatives(vocab)
        if len(cands) < 2 or truth not in cands:
            rows.append(None)
            continue
        scores = chooser.score([left + c + right for c in cands])
        order = sorted(range(len(cands)), key=lambda i: -scores[i])
        rows.append((cands[order[0]] == truth, scores[order[0]] - scores[order[1]], truth))
    report = {"spots": len(spots), "unambiguous": sum(r is None for r in rows), "by_margin": {}}
    for mg in (float(x) for x in args.margins.split(",")):
        right_n = sum(1 for r in rows if r and r[1] >= mg and r[0])
        wrong = [r[2] for r in rows if r and r[1] >= mg and not r[0]]
        abstain = sum(1 for r in rows if r and r[1] < mg)
        report["by_margin"][str(mg)] = {"right": right_n, "wrong": len(wrong), "abstain": abstain, "wrong_words": Counter(wrong).most_common(10)}
        print(f"margin {mg}: đúng {right_n}, SAI {len(wrong)}, bỏ qua {abstain}  {Counter(wrong).most_common(5)}")
    (args.run_dir / "hyphen_lm_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
