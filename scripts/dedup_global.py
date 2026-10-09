"""
Dedup vòng 2 (D-09): gộp kho dấu vân tay của mọi nguồn (<data-root>/state/dedup_index/<nguồn>/*.parquet, do stage dedup
của scripts/run_pipeline.py ghi), tìm họ trùng toàn cục, ghi <data-root>/processed/dedup_global/verdict.parquet.

Chỉ đánh dấu, không xoá dữ liệu (R-01). Sau đó áp verdict vào một lần chạy (sinh lại clean / knowledge unit / khối CPT
theo họ trùng toàn cục, các stage trước không chạy lại):
    uv run python scripts/dedup_global.py --data-root /duong/dan/data
    uv run python scripts/run_pipeline.py --data-root /duong/dan/data --total 10000 --global-verdict \\
        /duong/dan/data/processed/dedup_global/verdict.parquet
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được vi_corpus

from vi_corpus.common.state import (  # noqa: E402
    acquire_run_lock,
    setup_logging,
    write_json_atomic,
)
from vi_corpus.pipeline.config import RunConfig  # noqa: E402
from vi_corpus.pipeline.dedup_global import (  # noqa: E402
    global_verdict,
    read_index,
    write_verdict,
)

logger = logging.getLogger("vi_corpus")


def main() -> int:
    """Đọc kho dấu vân tay, tính verdict toàn cục, ghi verdict.parquet + verdict_stats.json. Trả về 1 nếu kho rỗng."""
    p = argparse.ArgumentParser(description="Dedup vòng 2 giữa mọi nguồn (D-09).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    defaults = RunConfig(mix={})
    p.add_argument("--threshold", type=float, default=defaults.fuzzy_threshold,
                   help="Ngưỡng Jaccard MinHash (mặc định = RunConfig.fuzzy_threshold, cấu hình A, R-34).")
    p.add_argument("--priority", default="",
                   help="Đổi mức ưu tiên giữ bản khi trùng, vd 'stbook=0,sea_pile_v2=1' (mặc định SOURCE_PRIORITY, R-15).")
    p.add_argument("--rights-gate", choices=["tag", "enforce"], default="tag",
                   help="tag = chỉ gắn nhãn (R-01, mặc định); enforce = văn bản quyền chưa rõ bị đánh 'rights'.")
    p.add_argument("--out", type=Path, default=None,
                   help="File verdict (mặc định <data-root>/processed/dedup_global/verdict.parquet).")
    args = p.parse_args()
    for part in filter(None, args.priority.split(",")):
        key, _, value = part.partition("=")
        if not key.strip() or not value.strip().lstrip("-").isdigit():
            p.error(f"--priority: '{part}' phải có dạng nguồn=số nguyên")
        defaults.source_priority[key.strip()] = int(value)
    _lock = acquire_run_lock(args.data_root / "state" / "dedup_global")  # giữ khoá tới khi thoát
    setup_logging(args.data_root / "logs", "dedup_global")
    index_dir = args.data_root / "state" / "dedup_index"
    items = read_index(index_dir, with_dropped=True)
    if not any(it.get("alive", True) for it in items):
        logger.error("Kho dấu vân tay rỗng: %s (chạy scripts/run_pipeline.py trước)", index_dir)
        return 1
    verdict, stats = global_verdict(items, args.threshold, defaults.priority, args.rights_gate)
    out = args.out or args.data_root / "processed" / "dedup_global" / "verdict.parquet"
    write_verdict(out, verdict)
    write_json_atomic(out.with_name("verdict_stats.json"), {"threshold": args.threshold, "rights_gate": args.rights_gate,
                                                            "priority": defaults.source_priority, "groups": stats})
    logger.info("Đã ghi %d dòng verdict ra %s", len(verdict), out)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
