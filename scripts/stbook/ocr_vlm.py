"""
OCR sách stbook.vn bằng chuỗi engine mô hình thị giác - ngôn ngữ (O-20), thay cho scripts/stbook/ocr.py (Paddle + VietOCR).

Mỗi trang đi qua `qwen3vl_8b_mt4k` (Qwen3-VL-8B qua vLLM); trang bị bộ kiểm từng trang báo lỗi (cắt cụt, rỗng, lặp, ảnh nhúng, ...)
chạy lại bằng `qwen3vl_8b_rp110`, còn bị báo thì dùng OCR cũ (Paddle + VietOCR); engine cuối vẫn bị báo thì trang bị loại khỏi
kết quả (xem vi_corpus/stbook/ocr_vlm.py). Cấu hình engine ở configs/ocr_bakeoff/engines.json. Ghi vào cùng thư mục với bộ OCR
cũ (<data-root>/interim/stbook_ocr/); cuốn đã OCR bằng bộ cũ được làm lại, file cũ chép sang interim/stbook_ocr_baseline/.

--serve tự dựng server vLLM trên GPU được chọn (--gpu), chờ sẵn sàng, chạy xong thì tắt. Dùng 4 GPU: chạy 4 tiến trình, mỗi
tiến trình một --gpu và một --port khác nhau (scripts/stbook/run_ocr_vlm.sh); các tiến trình tự chia cuốn bằng khoá, không cần
chia trước. Cần `uv sync --group ocr` nếu chuỗi có engine baseline (nạp khi có trang cần tới); vLLM cài riêng (--vllm-bin).
Ví dụ:
    uv run --no-sync python scripts/stbook/ocr_vlm.py --data-root /duong/dan/data --gpu 0 --serve --vllm-bin $(which vllm) --limit-books 1
    uv run --no-sync python scripts/stbook/ocr_vlm.py --data-root /duong/dan/data --status
"""

import argparse
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # để import được vi_corpus

from vi_corpus.common.registry import SOURCES  # noqa: E402
from vi_corpus.common.state import acquire_run_lock, setup_logging  # noqa: E402
from vi_corpus.ocr_bakeoff.cascade import DEFAULT_TIERS, CascadeOcr, EngineUnavailable, ServerDown, build_tiers  # noqa: E402
from vi_corpus.ocr_bakeoff.engines import DEFAULT_ENGINES, load_engines, start_server, stop_server, wait_ready  # noqa: E402
from vi_corpus.stbook.ocr_vlm import dump_report, run_ocr_vlm, vlm_status  # noqa: E402


def main() -> int:
    """Đọc tham số, rồi in tiến độ (--status) hoặc dựng server + chạy chuỗi OCR. Trả về 1 nếu có cuốn lỗi hoặc server chết."""
    p = argparse.ArgumentParser(description="OCR sách stbook.vn bằng chuỗi engine VLM (O-20).")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")),
                   help="Thư mục gốc dữ liệu (mặc định: biến SEA_DATA_ROOT hoặc ./data).")
    p.add_argument("--stbook-root", type=Path, default=None, help="Thư mục data/ của stbook-crawler (mặc định <data-root>/raw/stbook).")
    p.add_argument("--config", type=Path, default=DEFAULT_ENGINES, help="File cấu hình engine.")
    p.add_argument("--tiers", default=",".join(DEFAULT_TIERS),
                   help="Chuỗi engine, cách nhau dấu phẩy (mặc định %(default)s). Các engine VLM phải cùng mô hình.")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES cho server và engine baseline (vd 0).")
    p.add_argument("--serve", action="store_true", help="Tự dựng / tắt server vLLM bằng lệnh 'serve' của engine VLM đầu.")
    p.add_argument("--vllm-bin", default=None, help="Đường dẫn lệnh vllm (môi trường riêng); mặc định 'vllm'.")
    p.add_argument("--port", type=int, default=None, help="Cổng server (mặc định theo cấu hình; mỗi GPU một cổng khác nhau).")
    p.add_argument("--gpu-mem", type=float, default=0.85,
                   help="--gpu-memory-utilization của server (mặc định 0,85, chừa chỗ cho engine baseline chạy cùng GPU).")
    p.add_argument("--concurrency", type=int, default=32,
                   help="Số trang gửi đồng thời (mặc định 32 = mức chọn theo bộ thử 1.000 trang, O-22: gần nhanh bằng 64 nhưng độ trễ chỉ bằng nửa).")
    p.add_argument("--fallback-device", default="cuda", choices=["cuda", "cpu"], help="Thiết bị của engine baseline (mặc định cuda).")
    p.add_argument("--worker-name", default=None, help="Tên tiến trình, chống chạy trùng (mặc định gpu<N> hoặc main).")
    p.add_argument("--server-timeout", type=float, default=3600, help="Thời gian chờ server sẵn sàng (giây).")
    p.add_argument("--limit-books", type=int, default=None, help="Chỉ OCR N cuốn (chạy thử).")
    p.add_argument("--redo", action="store_true", help="OCR lại mọi cuốn, kể cả cuốn đã xong bằng chuỗi VLM.")
    p.add_argument("--status", action="store_true", help="Chỉ in tiến độ và thống kê trang bị loại rồi thoát.")
    args = p.parse_args()
    stbook_root = args.stbook_root or args.data_root / SOURCES["stbook"].path
    if not stbook_root.is_dir():
        p.error(f"Không thấy thư mục {stbook_root}")

    if args.status:
        st = vlm_status(stbook_root, args.data_root)
        print(f"[stbook] OCR bằng chuỗi VLM xong {st['vlm']}/{st['total']} cuốn có PDF "
              f"({st['other']} cuốn đã OCR bằng bộ cũ, sẽ được làm lại)")
        print(dump_report(args.data_root))
        return 0
    engines = load_engines(args.config)
    names = [t.strip() for t in args.tiers.split(",") if t.strip()]
    try:
        tiers = build_tiers(names, engines, port=args.port, device=args.fallback_device, gpu_mem=args.gpu_mem)
    except ValueError as e:
        p.error(str(e))
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    worker = args.worker_name or (f"gpu{args.gpu}" if args.gpu is not None else "main")
    state = args.data_root / "state" / "stbook_ocr_vlm" / worker
    _lock = acquire_run_lock(state)  # noqa: F841 — giữ khoá
    setup_logging(args.data_root / "logs", f"stbook_ocr_vlm_{worker}")
    vlm = next((t for t in tiers if t.cfg.get("type") == "openai"), None)
    proc = None
    try:
        if vlm is not None:
            if args.serve:
                proc = start_server(vlm.cfg, state, args.gpu, args.vllm_bin)
            wait_ready(vlm.cfg["base_url"], vlm.cfg.get("health", "/models"), args.server_timeout, proc)
        return 1 if run_ocr_vlm(stbook_root, args.data_root, CascadeOcr(tiers), args.concurrency, args.limit_books, args.redo) else 0
    except (ServerDown, EngineUnavailable):
        return 1
    finally:
        if proc is not None:
            stop_server(proc)


if __name__ == "__main__":
    sys.exit(main())
