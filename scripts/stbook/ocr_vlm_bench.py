"""
Thử concurrency và tỷ lệ trang bị báo của chuỗi OCR VLM trên ~1.000 trang stbook thật (O-20, mục "chưa duyệt").

Bốn lệnh con (logic ở vi_corpus/ocr_bakeoff/vlm_bench.py), thư mục làm việc mặc định <data-root>/processed/ocr_vlm_bench:
    sample  : chọn trang ngẫu nhiên từ nhiều sách stbook, ghi ảnh gốc (CPU, chạy một lần).
    run     : chạy cả bộ trang qua `qwen3vl_8b_mt4k` ở một mức --concurrency, trên GPU --gpu; --serve tự dựng / tắt server vLLM.
              Chạy lại thì chỉ làm trang còn thiếu / lỗi.
    retry   : chạy lại các trang bị bộ kiểm báo (ở mức đầu tiên) bằng `qwen3vl_8b_rp110`.
    report  : tổng hợp và in bảng (trang/giây, độ trễ, tỷ lệ trang bị báo, giờ cả kho theo từng mức).
Chạy trọn bằng scripts/stbook/run_ocr_vlm_bench.sh (mỗi mức concurrency một GPU, song song). Ví dụ chạy tay một mức:
    uv run --no-sync python scripts/stbook/ocr_vlm_bench.py sample --data-root /duong/dan/data
    uv run --no-sync python scripts/stbook/ocr_vlm_bench.py run --data-root /duong/dan/data --concurrency 32 --gpu 1 --serve --vllm-bin $(which vllm)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # để import được vi_corpus

from vi_corpus.common.registry import SOURCES  # noqa: E402
from vi_corpus.common.state import acquire_run_lock, setup_logging  # noqa: E402
from vi_corpus.ocr_bakeoff.cascade import retarget  # noqa: E402
from vi_corpus.ocr_bakeoff.engines import (  # noqa: E402
    DEFAULT_ENGINES,
    load_engines,
    run_engine,
    start_server,
    stop_server,
    wait_ready,
)
from vi_corpus.ocr_bakeoff.vlm_bench import (  # noqa: E402
    DEFAULT_CORPUS_PAGES,
    flagged_pages,
    format_report,
    make_retry_dir,
    sample_stbook_pages,
    summarize,
)
from vi_corpus.stbook.ocr_books import find_books  # noqa: E402


def _run(bench: Path, run_name: str, engine: str, args, cfg_dir: Path | None = None) -> int:
    """Chạy `engine` trên bộ trang trong `bench` (hoặc `cfg_dir`) với tên lượt `run_name`; dựng / tắt server nếu --serve. Trả về mã thoát."""
    engines = load_engines(args.config)
    cfg = retarget(engines[engine], port=args.port, gpu_mem=args.gpu_mem)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    target = cfg_dir or bench
    run_dir = target / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _lock = acquire_run_lock(run_dir)  # noqa: F841 — giữ khoá
    proc = None
    try:
        if args.serve and cfg.get("serve"):
            proc = start_server(cfg, run_dir, args.gpu, args.vllm_bin)
        t0 = time.time()
        wait_ready(cfg["base_url"], cfg.get("health", "/models"), args.server_timeout, proc)
        res = run_engine(target, run_name, cfg, concurrency=args.concurrency, limit=args.limit,
                         server_wait_s=round(time.time() - t0, 1))
        print(json.dumps(res, ensure_ascii=False))
        return 1 if res["errors"] else 0
    finally:
        if proc is not None:
            stop_server(proc)


def main() -> int:
    """Đọc tham số, chạy lệnh con. Trả về 1 nếu có trang lỗi."""
    p = argparse.ArgumentParser(description="Thử concurrency và tỷ lệ trang bị báo của chuỗi OCR VLM (O-20).")
    p.add_argument("command", choices=["sample", "run", "retry", "report"])
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    p.add_argument("--stbook-root", type=Path, default=None, help="Mặc định <data-root>/raw/stbook.")
    p.add_argument("--out", type=Path, default=None, help="Thư mục làm việc (mặc định <data-root>/processed/ocr_vlm_bench).")
    p.add_argument("--config", type=Path, default=DEFAULT_ENGINES, help="File cấu hình engine.")
    p.add_argument("--pages", type=int, default=1000, help="sample: số trang (mặc định 1000).")
    p.add_argument("--books", type=int, default=50, help="sample: số cuốn (mặc định 50, ~20 trang mỗi cuốn).")
    p.add_argument("--seed", type=int, default=20261016, help="sample: hạt giống ngẫu nhiên.")
    p.add_argument("--engine", default="qwen3vl_8b_mt4k", help="run: engine chính (mặc định qwen3vl_8b_mt4k).")
    p.add_argument("--retry-engine", default="qwen3vl_8b_rp110", help="retry: engine chạy lại (mặc định qwen3vl_8b_rp110).")
    p.add_argument("--concurrency", type=int, default=16, help="run / retry: số trang gửi đồng thời.")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES cho server.")
    p.add_argument("--serve", action="store_true", help="Tự dựng / tắt server vLLM.")
    p.add_argument("--vllm-bin", default=None, help="Đường dẫn lệnh vllm.")
    p.add_argument("--port", type=int, default=None, help="Cổng server (mỗi mức chạy song song một cổng khác nhau).")
    p.add_argument("--gpu-mem", type=float, default=0.85, help="--gpu-memory-utilization của server (mặc định 0,85 như lúc chạy thật).")
    p.add_argument("--server-timeout", type=float, default=3600)
    p.add_argument("--limit", type=int, default=None, help="run / retry: chỉ chạy N trang (thử nhanh).")
    p.add_argument("--corpus-pages", type=int, default=DEFAULT_CORPUS_PAGES, help="report: tổng số trang cần OCR cả kho.")
    p.add_argument("--num-gpus", type=int, default=4, help="report: số GPU quy ra giờ cả kho.")
    args = p.parse_args()
    out = args.out or args.data_root / "processed/ocr_vlm_bench"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(args.data_root / "logs", f"ocr_vlm_bench_{args.command}")

    if args.command == "sample":
        stbook_root = args.stbook_root or args.data_root / SOURCES["stbook"].path
        if not stbook_root.is_dir():
            p.error(f"Không thấy thư mục {stbook_root}")
        if (out / "pages.jsonl").exists():
            print(f"Đã có {out / 'pages.jsonl'}: giữ nguyên (xoá thư mục {out} để chọn lại)")
            return 0
        recs = sample_stbook_pages(list(find_books(stbook_root)), out, args.pages, args.books, args.seed)
        print(f"Đã chọn {len(recs)} trang vào {out}")
        return 0
    if not (out / "pages.jsonl").exists():
        p.error(f"Chưa có {out}/pages.jsonl: chạy lệnh con sample trước")
    if args.command == "run":
        return _run(out, f"c{args.concurrency}", args.engine, args)
    if args.command == "retry":
        first = sorted((d for d in (out / "runs").glob("c*")), key=lambda d: int(d.name[1:]))
        if not first:
            p.error("Chưa có lượt run nào để lấy trang bị báo")
        ids = list(flagged_pages(first[0]))
        print(f"{len(ids)} trang bị báo ở {first[0].name}, chạy lại bằng {args.retry_engine}")
        if not ids:
            return 0
        return _run(out, "rp110", args.retry_engine, args, cfg_dir=make_retry_dir(out, ids))
    summary = summarize(out, args.corpus_pages, args.num_gpus)
    print(format_report(summary))
    print(f"Chi tiết: {out / 'report/summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
