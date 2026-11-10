"""
So hai lần chạy pipeline VJOL bằng kết quả audit (J-33 "cách đo xong"): giữ / loại, từng lỗi còn lọt, năm 1970, độ lệch tạp chí. Chỉ CPU, đọc <run>/audit/audit.json
(chạy scripts/vjol/audit_run.py trên mỗi lần chạy trước).
    uv run python scripts/vjol/compare_runs.py --old data/processed/pipeline_runs/vjol_10k_v5 --new data/processed/pipeline_runs/vjol_10k_v6

--paired (J-40 b, J-42 (1b)): so GHÉP ĐÔI theo doc_id, chỉ các đoạn có ở cả hai lần chạy (đọc 04_quality.parquet, không cần audit.json): đoạn đổi giữ / loại kèm mã, đoạn đổi chữ, và --samples
mẫu đổi chữ ngẫu nhiên để đọc. Dùng khi hai lần chạy khác mẫu (kết quả không chỉ do sửa code gây ra); khi có audit.json của cả hai thì bảng audit cũng được in.
    uv run python scripts/vjol/compare_runs.py --old <v7> --new <v8> --paired [--samples 20] | tee <v8>/paired_v7_v8.txt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.pipeline.io import read_rows  # noqa: E402
from vi_corpus.vjol.audit import OUT_SUBDIR, compare_audits  # noqa: E402
from vi_corpus.vjol.paired import format_paired, paired_compare  # noqa: E402


def _vjol_rows(run: Path) -> list[dict]:
    """Bản ghi vjol đã qua stage quality (04_quality.parquet) của một lần chạy."""
    return [r for r in read_rows(run / "04_quality.parquet") if r["source_key"] == "vjol"]


def main() -> int:
    """Đọc hai audit.json, in bảng so sánh. Trả về 1 nếu thiếu file audit."""
    p = argparse.ArgumentParser(description="So hai lần chạy pipeline VJOL theo audit.json.")
    p.add_argument("--old", type=Path, required=True, help="Thư mục lần chạy cũ (vd vjol_10k_v5).")
    p.add_argument("--new", type=Path, required=True, help="Thư mục lần chạy mới (vd vjol_10k_v6).")
    p.add_argument("--paired", action="store_true", help="So ghép đôi theo doc_id (đọc 04_quality.parquet của hai lần chạy).")
    p.add_argument("--samples", type=int, default=20, help="Với --paired: số đoạn đổi chữ lấy ngẫu nhiên để đọc.")
    p.add_argument("--seed", type=int, default=0, help="Với --paired: hạt giống chọn mẫu.")
    args = p.parse_args()
    if args.paired:
        for run in (args.old, args.new):
            if not (run / "04_quality.parquet").is_file():
                print(f"Thiếu {run / '04_quality.parquet'}", file=sys.stderr)
                return 1
        print(format_paired(paired_compare(_vjol_rows(args.old), _vjol_rows(args.new), args.samples, args.seed), args.old.name[-8:], args.new.name[-8:]))
        if not all((run / OUT_SUBDIR / "audit.json").is_file() for run in (args.old, args.new)):
            return 0
        print()
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
