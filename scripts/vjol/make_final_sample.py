"""
Mẫu thi cuối của VJOL (J-42 (7), P3). Chỉ CPU. Mẫu này CHỈ dùng để đo MỘT lần sau khi các sửa xong; không chỉnh ngưỡng theo nó (chỉnh thì phải sinh mẫu mới với hạt giống khác).

    # 1) sinh mẫu: >= 600 đoạn, hạt giống 20261026, 100% ngẫu nhiên (không bốc theo bộ dò), loại các bài đã nằm trong mẫu gán nhãn cũ (--exclude-labels, có thể lặp)
    uv run python scripts/vjol/make_final_sample.py sample --run-dir data/processed/pipeline_runs/vjol_10k_v9 --exclude-labels <v6>/groundtruth/labeling_sample.csv
    # 2) Claude gán nhãn cột `label` / `note` như J-36 (nhãn hợp lệ: audit.LABELS); chọn 50 dòng cho chủ dự án xem tay
    uv run python scripts/vjol/make_final_sample.py review-sample --labels <run>/final_sample/labeling_sample.csv
    # 3) chủ dự án điền cột `user_label` của review_sample.csv; báo cáo theo ngân sách lỗi, kèm khoảng tin cậy Wilson 95% và độ đồng ý
    uv run python scripts/vjol/make_final_sample.py report --labels <run>/final_sample/labeling_sample.csv [--review <run>/final_sample/review_sample.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.pipeline.io import read_rows  # noqa: E402
from vi_corpus.vjol.audit import FINAL_MIN_N, FINAL_SEED, final_report, make_final_sample, make_review_sample, write_labeling_sample  # noqa: E402

SUBDIR = "final_sample"


def _read_csv(path: Path) -> list[dict]:
    """Đọc một file CSV (UTF-8) thành danh sách dòng."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    """Chạy lệnh con sample | review-sample | report. Trả về 1 nếu thiếu dữ liệu hoặc mẫu ít hơn yêu cầu."""
    p = argparse.ArgumentParser(description="Mẫu thi cuối của VJOL (J-42 (7)).")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="Sinh mẫu thi cuối.")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--n", type=int, default=FINAL_MIN_N)
    s.add_argument("--seed", type=int, default=FINAL_SEED)
    s.add_argument("--exclude-labels", type=Path, action="append", default=[], help="labeling_sample.csv cũ: bài trong đó bị loại khỏi mẫu thi cuối.")
    s.add_argument("--out", type=Path, default=None, help="Mặc định <run-dir>/final_sample.")
    r = sub.add_parser("review-sample", help="Chọn 50 dòng đã gán nhãn cho chủ dự án xem tay.")
    r.add_argument("--labels", type=Path, required=True)
    r.add_argument("--n", type=int, default=50)
    r.add_argument("--out", type=Path, default=None, help="Mặc định cạnh file nhãn (review_sample.csv).")
    e = sub.add_parser("report", help="Báo cáo theo ngân sách lỗi.")
    e.add_argument("--labels", type=Path, required=True)
    e.add_argument("--review", type=Path, default=None)
    args = p.parse_args()

    if args.cmd == "sample":
        rows = [x for x in read_rows(args.run_dir / "04_quality.parquet") if x["source_key"] == "vjol"]
        excluded = frozenset(str(x["article_id"]) for path in args.exclude_labels for x in _read_csv(path) if x.get("article_id"))
        items = make_final_sample(rows, args.n, args.seed, excluded)
        out = args.out or args.run_dir / SUBDIR
        write_labeling_sample(items, out)
        print(f"Ghi {len(items)} đoạn (yêu cầu {args.n}) vào {out}; đã loại {len(excluded)} bài của các mẫu gán nhãn cũ")
        return 0 if len(items) >= args.n else 1
    items = _read_csv(args.labels)
    if args.cmd == "review-sample":
        picked = make_review_sample(items, args.n)
        out = args.out or args.labels.with_name("review_sample.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["doc_id", "label", "note", "text", "user_label"])
            w.writeheader()
            w.writerows(picked)
        print(f"Ghi {len(picked)} dòng xem tay vào {out}; điền cột user_label rồi chạy `report --review`")
        return 0 if picked else 1
    print(json.dumps(final_report(items, _read_csv(args.review) if args.review else None), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
