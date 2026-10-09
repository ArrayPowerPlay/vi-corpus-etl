"""
Lấy mẫu ~50.000 đoạn cho LLM lớn chấm điểm (R-11, G-06; cách lấy ở vi_corpus/pipeline/judge_sample.py).

Đọc 04_quality.parquet của một hoặc nhiều lần chạy (scripts/run_pipeline.py --until quality là đủ), ghi:
    <out>                       mẫu đã chọn (doc_id, source_key, text, meta, token_count, length_bucket, stratum,
                                rule_pass, quality_band, quality_score, reason_codes, language, lang_score, lang_mix, run)
    <out>.stats.json            số mẫu theo nhóm x qua / bị loại rule, theo tầng nguồn x độ dài
Ví dụ:
    uv run python scripts/make_judge_sample.py --run-dir /data/processed/pipeline_runs/run_100000_seed42 \\
        --n 50000 --out /data/processed/judge/sample_50k.parquet
Việc chấm (LLM trên GPU server) chưa có script: thang điểm 0-5 chưa duyệt (Q6-thang).
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được vi_corpus

from vi_corpus.common.state import write_json_atomic  # noqa: E402
from vi_corpus.pipeline.judge_sample import draw_sample  # noqa: E402

COLUMNS = ["doc_id", "source_key", "text", "meta", "token_count", "quality_band", "quality_score", "reason_codes",
           "language", "lang_score", "lang_mix"]
OUT_SCHEMA = pa.schema([
    ("doc_id", pa.string()), ("source_key", pa.string()), ("text", pa.string()), ("meta", pa.string()),
    ("token_count", pa.int64()), ("length_bucket", pa.string()), ("stratum", pa.string()), ("judge_group", pa.string()),
    ("rule_pass", pa.bool_()), ("quality_band", pa.string()), ("quality_score", pa.float64()),
    ("reason_codes", pa.list_(pa.string())), ("language", pa.string()), ("lang_score", pa.float64()),
    ("lang_mix", pa.string()), ("run", pa.string()),
])


def main() -> int:
    """Đọc tham số, lấy mẫu, ghi Parquet + thống kê. Trả về 1 nếu không lấy được mẫu nào."""
    p = argparse.ArgumentParser(description="Lấy mẫu phân tầng cho LLM chấm điểm (R-11).")
    p.add_argument("--run-dir", type=Path, action="append", required=True,
                   help="Thư mục một lần chạy pipeline (lặp lại cờ này để gộp nhiều lần chạy).")
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--pass-share", type=float, default=0.7, help="Phần mẫu qua rule (còn lại là mẫu bị rule loại).")
    p.add_argument("--sft-share", type=float, default=0.1, help="Phần mẫu hỏi - đáp SEA-Instruct (~5.000 / 50.000).")
    p.add_argument("--keep-bands", default=None,
                   help="Band coi là qua rule, vd 'A,B,C'. Mặc định lấy keep_bands trong manifest.json của lần chạy "
                        "(chỉ có sau finalize), không có thì A,B,C.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows = []
    for run in args.run_dir:
        table = pq.read_table(run / "04_quality.parquet", columns=COLUMNS)
        rows += [{**r, "run": run.name} for r in table.to_pylist()]
    if args.keep_bands:
        keep_bands = tuple(b.strip() for b in args.keep_bands.split(",") if b.strip())
    else:
        manifest = args.run_dir[0] / "manifest.json"
        config = json.loads(manifest.read_text(encoding="utf-8")).get("config", {}) if manifest.exists() else {}
        keep_bands = tuple(config.get("keep_bands") or ("A", "B", "C"))
    sample, stats = draw_sample(rows, args.n, args.pass_share, args.sft_share, keep_bands, seed=args.seed)
    stats["keep_bands"] = list(keep_bands)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    pq.write_table(pa.Table.from_pylist([{k: r.get(k) for k in OUT_SCHEMA.names} for r in sample], schema=OUT_SCHEMA), tmp)
    tmp.replace(args.out)
    write_json_atomic(args.out.with_name(args.out.name + ".stats.json"), stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "strata"}, ensure_ascii=False, indent=1))
    return 0 if sample else 1


if __name__ == "__main__":
    sys.exit(main())
