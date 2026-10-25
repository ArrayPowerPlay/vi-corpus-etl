"""
Kiểm một lần chạy pipeline có VJOL (J-25 Bước 0 (c)): lỗi còn lọt, tạp chí / năm bị loại bất thường, đoạn bị loại chỉ vì một luật, kèm mẫu. Chỉ CPU, vài giây.
Ghi <run-dir>/audit/{audit.json,audit_journals.csv,audit.md}. Chạy trên MỌI lần chạy pipeline có VJOL trước khi tin kết quả.
    uv run python scripts/vjol/audit_run.py --run-dir data/processed/pipeline_runs/vjol_10k_v3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.vjol.audit import audit_run  # noqa: E402


def main() -> int:
    """Đọc tham số, chạy kiểm, in thư mục kết quả. Trả về 1 nếu có nhóm bị loại bất thường."""
    p = argparse.ArgumentParser(description="Kiểm một lần chạy pipeline VJOL.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--min-group", type=int, default=20, help="Nhóm tạp chí (x năm) ít đoạn hơn ngần này thì bỏ qua.")
    p.add_argument("--factor", type=float, default=3.0, help="Bất thường nếu tỷ lệ loại > factor x trung vị...")
    p.add_argument("--min-rate", type=float, default=0.15, help="... và >= mức này.")
    p.add_argument("--syllables", type=Path, default=None, help="Tập âm tiết để đo họ lỗi đọc font (J-33 H); mặc định <data-root>/processed/ocr_bakeoff/syllables.json.")
    args = p.parse_args()
    out = audit_run(args.run_dir, min_group=args.min_group, factor=args.factor, min_rate=args.min_rate, syllables=args.syllables)
    res = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    print(f"Ghi {out}: giữ {res['kept']}/{res['chunks']} đoạn; lỗi còn lọt {res['error_rates']}; {len(res['anomalies'])} nhóm bị loại bất thường")
    return 1 if res["anomalies"] else 0


if __name__ == "__main__":
    sys.exit(main())
