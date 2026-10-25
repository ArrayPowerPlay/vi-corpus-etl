"""
So hai lần chạy pipeline VJOL bằng kết quả audit (J-33 "cách đo xong"): giữ / loại, từng lỗi còn lọt, năm 1970, độ lệch tạp chí. Chỉ CPU, đọc <run>/audit/audit.json
(chạy scripts/vjol/audit_run.py trên mỗi lần chạy trước).
    uv run python scripts/vjol/compare_runs.py --old data/processed/pipeline_runs/vjol_10k_v5 --new data/processed/pipeline_runs/vjol_10k_v6
"""

import argparse
import json
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.vjol.audit import OUT_SUBDIR, compare_audits  # noqa: E402


def main() -> int:
    """Đọc hai audit.json, in bảng so sánh. Trả về 1 nếu thiếu file audit."""
    p = argparse.ArgumentParser(description="So hai lần chạy pipeline VJOL theo audit.json.")
    p.add_argument("--old", type=Path, required=True, help="Thư mục lần chạy cũ (vd vjol_10k_v5).")
    p.add_argument("--new", type=Path, required=True, help="Thư mục lần chạy mới (vd vjol_10k_v6).")
    args = p.parse_args()
    audits = []
    for run in (args.old, args.new):
        path = run / OUT_SUBDIR / "audit.json"
        if not path.is_file():
            print(f"Thiếu {path}: chạy scripts/vjol/audit_run.py --run-dir {run} trước", file=sys.stderr)
            return 1
        audits.append(json.loads(path.read_text(encoding="utf-8")))
    print(compare_audits(audits[0], audits[1], args.old.name[-8:], args.new.name[-8:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
