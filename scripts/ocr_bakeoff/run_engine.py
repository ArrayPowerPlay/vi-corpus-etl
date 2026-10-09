"""
Bước 2 của đợt so sánh engine OCR (F-11): chạy một hoặc nhiều engine trên bộ trang, lưu đầu ra thô theo trang.

Engine khai báo ở configs/ocr_bakeoff/engines.json (baseline, paddleocr_vl, dots_ocr, qwen3vl_8b, qwen3vl_4b). Mỗi
engine chạy trên một GPU (--gpu). Với engine dạng server (vLLM, PaddleOCR genai_server), --serve tự dựng server bằng
lệnh "serve" trong cấu hình, chờ sẵn sàng, chạy xong thì tắt; không có --serve thì phải dựng server trước. Kết quả:
<bakeoff-dir>/runs/<engine>/outputs.jsonl (chạy lại thì chỉ làm trang còn thiếu / lỗi), run_info.json, server.log.
Mỗi engine có khoá chạy riêng: hai tiến trình không chạy cùng một engine.

Môi trường: baseline cần `uv sync --group ocr` (+ paddlepaddle-gpu nếu muốn detect trên GPU); paddleocr_vl cần
paddleocr[doc-parser] + paddlepaddle-gpu; vLLM nên cài ở môi trường riêng và trỏ bằng --vllm-bin (client chỉ cần requests).
Ví dụ:
    uv run --group ocr python scripts/ocr_bakeoff/run_engine.py --data-root /duong/dan/data --engine baseline --gpu 0
    uv run python scripts/ocr_bakeoff/run_engine.py --data-root /duong/dan/data --engine qwen3vl_8b --gpu 1 --serve \
        --vllm-bin /opt/vllm-env/bin/vllm --concurrency 16
    uv run python scripts/ocr_bakeoff/run_engine.py --data-root /duong/dan/data --engine dots_ocr --limit 5   # thử nhanh
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])

from vi_corpus.common.state import acquire_run_lock, setup_logging  # noqa: E402
from vi_corpus.ocr_bakeoff.engines import (  # noqa: E402
    DEFAULT_ENGINES,
    load_engines,
    pending_pages,
    run_engine,
    start_server,
    stop_server,
    wait_ready,
)


def main() -> int:
    """Đọc tham số, chạy lần lượt từng engine. Trả về 1 nếu có engine lỗi hoặc có trang lỗi."""
    p = argparse.ArgumentParser(description="Chạy engine OCR trên bộ trang so sánh (F-11).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")),
                   help="Thư mục gốc dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).")
    p.add_argument("--bakeoff-dir", type=Path, default=None,
                   help="Thư mục đợt so sánh (mặc định <data-root>/processed/ocr_bakeoff).")
    p.add_argument("--config", type=Path, default=DEFAULT_ENGINES, help="File cấu hình engine.")
    p.add_argument("--engine", action="append", required=True, help="Tên engine trong cấu hình; lặp được.")
    p.add_argument("--gpu", default=None, help="Giá trị CUDA_VISIBLE_DEVICES cho engine và server (vd 0).")
    p.add_argument("--serve", action="store_true", help="Tự dựng / tắt server cho engine có khoá 'serve'.")
    p.add_argument("--vllm-bin", default=None, help="Đường dẫn lệnh vllm (môi trường riêng); mặc định 'vllm'.")
    p.add_argument("--concurrency", type=int, default=16,
                   help="Số trang gửi đồng thời cho engine dạng API (mặc định 16; vLLM tự gom lô).")
    p.add_argument("--limit", type=int, default=None, help="Chỉ chạy N trang (thử nhanh).")
    p.add_argument("--set", choices=["A", "B"], default=None, help="Chỉ chạy một bộ trang.")
    p.add_argument("--redo", action="store_true", help="Xoá kết quả cũ của engine rồi chạy lại.")
    p.add_argument("--server-timeout", type=float, default=3600, help="Thời gian chờ server sẵn sàng (giây).")
    args = p.parse_args()
    bakeoff = args.bakeoff_dir or args.data_root / "processed/ocr_bakeoff"
    if not (bakeoff / "pages.jsonl").exists():
        p.error(f"Chưa có {bakeoff}/pages.jsonl: chạy select_pages.py trước")
    engines = load_engines(args.config)
    unknown = [e for e in args.engine if e not in engines]
    if unknown:
        p.error(f"Engine không có trong {args.config}: {unknown} (có: {', '.join(engines)})")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_logging(args.data_root / "logs", "ocr_bakeoff_" + "_".join(args.engine))
    results, failed = [], False
    for name in args.engine:
        cfg = engines[name]
        _lock = acquire_run_lock(bakeoff / "runs" / name)  # noqa: F841 - giữ khoá suốt engine này
        proc = None
        if not args.redo and not pending_pages(bakeoff, name, args.set, args.limit):
            print(json.dumps({"engine": name, "pages": 0, "note": "mọi trang đã có kết quả"}, ensure_ascii=False))
            continue
        try:
            if args.serve and cfg.get("serve"):
                proc = start_server(cfg, bakeoff / "runs" / name, args.gpu, args.vllm_bin)
            if cfg.get("base_url"):
                wait_ready(cfg["base_url"], cfg.get("health", "/models"), args.server_timeout, proc)
            res = run_engine(bakeoff, name, cfg, concurrency=args.concurrency, limit=args.limit, redo=args.redo,
                             only_set=args.set)
            failed |= res["errors"] > 0
        except Exception as e:  # noqa: BLE001 - một engine hỏng (thiếu thư viện, server không lên) không chặn engine sau
            res = {"engine": name, "error": f"{type(e).__name__}: {e}"}
            failed = True
        finally:
            if proc is not None:
                stop_server(proc)
        results.append(res)
        print(json.dumps(res, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
