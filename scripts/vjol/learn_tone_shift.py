"""
Học bảng đổi "lệch dấu" (chữ có dấu bị đọc thành chữ có dấu khác: "nghiờn" thay "nghiên") từ dữ liệu thật (J-42 (1c): chọn bảng đổi bằng dữ liệu, không đoán). Chỉ CPU.

Đọc `04_quality.parquet` của một lần chạy (nguồn vjol, mọi band: chữ lệch dấu thường nằm ở đoạn bị loại nên không chỉ lấy đoạn giữ), đếm các cặp (ký tự sai -> ký tự đúng) làm từ lạ thành âm tiết thật
(vi_corpus.common.defects.learn_tone_shift_map), ghi <data-root>/processed/ocr_bakeoff/tone_shift.json ({"map", "pairs", "unknown_tokens"}); audit_run.py và groundtruth.py evaluate tự dùng file này.
Đọc `pairs` rồi báo lại: cặp nào hợp lý thì để lại trong `map`.
    uv run python scripts/vjol/learn_tone_shift.py --data-root data --run-dir data/processed/pipeline_runs/vjol_10k_v8 [--min-count 30 --min-share 0.3]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.common.defects import TONE_SHIFT_REL, learn_tone_shift_map  # noqa: E402
from vi_corpus.common.garbled import load_vocab  # noqa: E402
from vi_corpus.common.state import write_json_atomic  # noqa: E402
from vi_corpus.pipeline.io import read_rows  # noqa: E402
from vi_corpus.vjol.extract import SYLLABLES_REL  # noqa: E402


def main() -> int:
    """Đọc tham số, học bảng đổi, ghi file và in các cặp. Trả về 1 nếu thiếu tập âm tiết hoặc không có dữ liệu."""
    p = argparse.ArgumentParser(description="Học bảng đổi lệch dấu từ một lần chạy pipeline VJOL (CPU).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    p.add_argument("--run-dir", type=Path, required=True, help="Lần chạy có 04_quality.parquet.")
    p.add_argument("--min-count", type=int, default=30, help="Số lần tối thiểu của một cặp.")
    p.add_argument("--min-share", type=float, default=0.3, help="Tỷ lệ tối thiểu của cặp trên các từ lạ có chứa ký tự sai.")
    args = p.parse_args()
    vocab = load_vocab(args.data_root / SYLLABLES_REL)
    if vocab is None:
        print(f"Thiếu {args.data_root / SYLLABLES_REL}", file=sys.stderr)
        return 1
    texts = [r["text"] or "" for r in read_rows(args.run_dir / "04_quality.parquet") if r["source_key"] == "vjol"]
    if not texts:
        print("Không có đoạn vjol nào", file=sys.stderr)
        return 1
    res = learn_tone_shift_map(texts, vocab, args.min_count, args.min_share)
    out = args.data_root / TONE_SHIFT_REL
    write_json_atomic(out, res)
    print(f"{len(texts)} đoạn, {res['unknown_tokens']} từ lạ; bảng đổi {res['map']} -> {out}")
    for row in res["pairs"][:30]:
        print(f"  {row['wrong']} -> {row['right']}: {row['count']} lần, tỷ lệ {row['share']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
