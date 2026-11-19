"""
Kiểm tra nhanh chất lượng các nguồn ứng viên (docs/sources/SOURCE_CANDIDATES.md): lấy mẫu từ xa rồi đo bằng số đo của pipeline.

Kết quả ở <data-root>/processed/source_probe/:
- <nguồn>/sample.jsonl: bản ghi mẫu thô (có cache: chạy lại không tải lại, trừ khi --resample);
- <nguồn>/measured.jsonl: số đo từng bản ghi (2.000 ký tự đầu văn bản để đọc lại);
- summary.json: tổng hợp mọi nguồn đã đo.

Ví dụ:
    uv run --no-sync --with zstandard python scripts/source_probe/probe_hf.py --data-root data --n 300
    uv run --no-sync python scripts/source_probe/probe_hf.py --data-root data --only finepdfs_edu_vi,vi_pubmed --n 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vi_corpus.source_probe.measure import dump_rows, measure_records, summarize  # noqa: E402
from vi_corpus.source_probe.sample import sample_source  # noqa: E402
from vi_corpus.source_probe.specs import SPECS  # noqa: E402

log = logging.getLogger("source_probe")


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data", help="thư mục dữ liệu (kết quả ở processed/source_probe/)")
    ap.add_argument("--only", default="", help="danh sách nguồn, cách nhau bởi dấu phẩy; mặc định mọi nguồn trong specs")
    ap.add_argument("--n", type=int, default=300, help="số bản ghi mẫu mỗi nguồn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resample", action="store_true", help="lấy mẫu lại dù đã có sample.jsonl")
    ap.add_argument("--lang-model", default="fasttext:lid.176")
    ap.add_argument("--tokenizer", default="hf:Qwen/Qwen3-0.6B")
    return ap.parse_args()


def load_or_sample(name: str, out_dir: Path, n: int, seed: int, resample: bool) -> list[dict]:
    """Đọc mẫu đã cache, hoặc lấy mẫu mới và ghi nguyên tử (file .part rồi đổi tên)."""
    path = out_dir / "sample.jsonl"
    if path.exists() and not resample:
        return [json.loads(line) for line in path.open(encoding="utf-8")]
    records = sample_source(SPECS[name], n, seed)
    tmp = path.with_suffix(".jsonl.part")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return records


def main() -> int:
    """Lấy mẫu + đo từng nguồn; một nguồn lỗi không chặn nguồn khác; exit 1 nếu có nguồn lỗi."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args()
    names = [s for s in args.only.split(",") if s] or list(SPECS)
    unknown = [s for s in names if s not in SPECS]
    if unknown:
        log.error("nguồn không có trong specs: %s", unknown)
        return 2
    root = Path(args.data_root) / "processed/source_probe"
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    failed = []
    for name in names:
        spec = SPECS[name]
        out_dir = root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            log.info("%s: lấy mẫu %d bản ghi (%s)", name, args.n, spec.kind)
            records = load_or_sample(name, out_dir, args.n, args.seed, args.resample)
            log.info("%s: đo %d bản ghi", name, len(records))
            rows = measure_records(records, is_web=spec.web, is_sft=spec.kind == "hf_rows",
                                   lang_model=args.lang_model, tokenizer=args.tokenizer)
            dump_rows(rows, out_dir / "measured.jsonl")
            summary[name] = {"note": spec.note, **summarize(rows)}
        except Exception as exc:  # noqa: BLE001 - ghi lỗi của nguồn này rồi đo tiếp nguồn khác
            log.exception("%s: lỗi %s", name, exc)
            failed.append(name)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("xong; tổng hợp ở %s; nguồn lỗi: %s", summary_path, failed or "không")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
