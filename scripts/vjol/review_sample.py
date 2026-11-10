"""
Lập bộ đọc tay chuẩn cho VJOL (J-22 (10)): 300 đoạn ngẫu nhiên + 100 đoạn nghi ngờ từ một lần chạy pipeline, để đo tỷ lệ lỗi thật.

Đoạn nghi ngờ = đoạn còn lại sau lọc (band A / B / C) mà có một trong: tỷ lệ từ lạ >= 0,08, tỷ lệ không dấu >= 0,01, độ lợi chữ đảo ngược >= 0,15,
còn ký tự vùng riêng hoặc chữ Ấn Độ / Ả Rập / Thái, hoặc có email / số điện thoại. Ghi review_sample.csv (doc_id, nhóm, band, điểm, reason code, các số đo,
văn bản đầy đủ, cột trống `nhan_tay`) và review_sample.md (đọc nhanh). Cùng seed cho cùng bộ đoạn.
    uv run python scripts/vjol/review_sample.py --run-dir data/processed/pipeline_runs/vjol_10k_v3
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.pipeline.io import read_rows  # noqa: E402
from vi_corpus.vjol.audit import suspicious  # noqa: E402


def main() -> int:
    """Đọc 04_quality.parquet, chọn đoạn, ghi CSV + Markdown. Trả về 1 nếu không có đoạn nào."""
    p = argparse.ArgumentParser(description="Lập bộ đọc tay VJOL: ngẫu nhiên + nghi ngờ.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--random", type=int, default=300, help="Số đoạn ngẫu nhiên.")
    p.add_argument("--suspect", type=int, default=100, help="Số đoạn nghi ngờ.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="Thư mục kết quả (mặc định <run-dir>/review).")
    args = p.parse_args()
    rows = [r for r in read_rows(args.run_dir / "04_quality.parquet") if r["source_key"] == "vjol" and r["quality_band"] in ("A", "B", "C")]
    if not rows:
        print("Không có đoạn vjol nào còn lại sau lọc chất lượng")
        return 1
    rng = random.Random(args.seed)
    sus = [r for r in rows if suspicious(r)]
    picked = [("nghi_ngờ", r) for r in rng.sample(sus, min(args.suspect, len(sus)))]
    taken = {r["doc_id"] for _, r in picked}
    rest = [r for r in rows if r["doc_id"] not in taken]
    picked += [("ngẫu_nhiên", r) for r in rng.sample(rest, min(args.random, len(rest)))]
    out = args.out or args.run_dir / "review"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "review_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "nhom", "band", "diem", "reason_codes", "nghi_ngo", "text", "nhan_tay"])
        for group, r in picked:
            w.writerow([r["doc_id"], group, r["quality_band"], r["quality_score"], "+".join(r["reason_codes"]), "+".join(suspicious(r)), r["text"], ""])
    with open(out / "review_sample.md", "w", encoding="utf-8") as f:
        for i, (group, r) in enumerate(picked, 1):
            f.write(f"## {i}. {r['doc_id']} [{group}] band {r['quality_band']} ({r['quality_score']}) {'+'.join(r['reason_codes'])}\n\n{r['text'][:1500]}\n\n")
    print(f"Ghi {len(picked)} đoạn ({len(sus)} đoạn nghi ngờ có sẵn) vào {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
