"""
Xuất đầu ra của make pipeline-cpu (VJOL + VISTA) thành thư mục dataset sẵn sàng đẩy lên Hugging Face.

Ví dụ:
    uv run --no-sync python scripts/hf/export_dataset.py --data-root data --repo-id arraypowerplay/vi-scientific-corpus
Kết quả: <data-root>/processed/hf_export/{data/,README.md,export_stats.json}. Không động tới mạng. Chạy lại thì ghi đè.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vi_corpus import hf_export as hx  # noqa: E402


def _run(args: tuple) -> dict:
    """Chạy một nhóm ghi file trong tiến trình con (đối số đã là tuple picklable)."""
    return hx.write_group(*args)


def main() -> int:
    """Đọc tham số, ghi parquet song song theo file, tính số liệu thật, ghi README.md và export_stats.json."""
    p = argparse.ArgumentParser(description="Xuất VJOL + VISTA thành dataset Hugging Face (parquet + dataset card).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    p.add_argument("--out", type=Path, default=None, help="Mặc định <data-root>/processed/hf_export.")
    p.add_argument("--repo-id", default="arraypowerplay/vi-scientific-corpus")
    p.add_argument("--shards", type=int, default=128, help="Số phần của make pipeline-cpu (SHARDS).")
    p.add_argument("--vjol-files", type=int, default=16, help="Số file parquet của config vjol.")
    p.add_argument("--vista-files", type=int, default=4, help="Số file parquet của config vista.")
    p.add_argument("--cpt-files", type=int, default=2, help="Số file cpt_segments cho mỗi nguồn.")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    runs = args.data_root / "processed" / "pipeline_runs"
    out = args.out or args.data_root / "processed" / "hf_export"
    if out.exists():
        shutil.rmtree(out)
    tasks = (hx.plan_tasks(runs, out, "vjol", args.shards, args.vjol_files, args.cpt_files)
             + hx.plan_tasks(runs, out, "vista", args.shards, args.vista_files, args.cpt_files))
    written = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(_run, tasks):
            logging.info("xong %s: %s dòng, %s token, %.0f MB", r["file"], r["rows"], r["tokens"], r["bytes"] / 1e6)
            written.append(r)
    stats = hx.collect_stats(out)
    stats["files"] = written
    (out / "export_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "README.md").write_text(hx.render_card(stats, args.repo_id), encoding="utf-8")
    logging.info("Xong: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
