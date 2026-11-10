"""
Tổng hợp các lần chạy pipeline của VISTA (VS-06): số đoạn, token giữ lại, phễu loại, RAM đỉnh, thời gian, theo từng phần `--shard` và tổng.

Đọc manifest.json + audit.json trong <data-root>/processed/pipeline_runs/<prefix>*/ (không đọc Parquet). Ghi cả JSON ra --out nếu truyền.

Ví dụ:
    uv run python scripts/vista/summarize_runs.py --data-root /duong/dan/data --prefix vista_full
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.vista.summary import summarize  # noqa: E402


def main() -> int:
    """Đọc tham số, in bảng từng lần chạy và dòng tổng. Trả về 1 nếu không có lần chạy nào hoặc còn lần chạy dang dở."""
    p = argparse.ArgumentParser(description="Tổng hợp các lần chạy pipeline VISTA.")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục gốc dữ liệu (mặc định SEA_DATA_ROOT hoặc ./data).")
    p.add_argument("--prefix", default="vista_full", help="Tiền tố tên lần chạy (mặc định vista_full: vista_full_s0of8, ...).")
    p.add_argument("--out", type=Path, default=None, help="Ghi thêm kết quả JSON ra file này.")
    args = p.parse_args()
    res = summarize(args.data_root, args.prefix)
    if not res["runs"]:
        print(f"Không thấy lần chạy nào có tên bắt đầu bằng {args.prefix!r} trong {args.data_root}/processed/pipeline_runs/")
        return 1
    print(f"{'run':28} {'xong':>5} {'đoạn':>10} {'giữ':>10} {'token giữ':>14} {'RAM MB':>8} {'giây':>8} {'verdict':>8}")
    for r in res["runs"]:
        print(f"{r['run']:28} {str(r['finished']):>5} {r['chunks'] or 0:>10} {r['kept'] or 0:>10} {r['tokens_kept'] or 0:>14} {r['peak_rss_mb']:>8} {r['seconds']:>8} {str(r['global_verdict']):>8}")
    t = res["total"]
    print(f"{'TỔNG (lần chạy đã xong)':28} {'':>5} {t['chunks']:>10} {t['kept']:>10} {t['tokens_kept']:>14}")
    print("phễu:", t["funnel"])
    if t["unfinished"]:
        print("CHƯA XONG:", ", ".join(t["unfinished"]))
    if t["without_global_verdict"]:
        print("CHƯA ÁP verdict dedup vòng 2 (token có thể còn trùng giữa các phần / giữa VJOL và VISTA):", ", ".join(t["without_global_verdict"]))
    if args.out:
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if t["unfinished"] else 0


if __name__ == "__main__":
    sys.exit(main())
