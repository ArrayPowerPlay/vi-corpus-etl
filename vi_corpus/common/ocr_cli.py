"""
Tham số dòng lệnh và cách dựng bộ OCR chuỗi engine VLM (O-20) dùng chung cho các script trích text theo lô (giáo trình, ...).

- add_cascade_args: thêm nhóm tham số OCR (engine, GPU, server vLLM, số trang đồng thời, ...) vào một ArgumentParser; giống nhóm tham số của
  vi_corpus.vjol.cli để cách chạy các nguồn giống nhau.
- build_getter: dựng CascadeOcrGetter từ các tham số đó (nạp lười: server vLLM chỉ được bật khi gặp trang scan đầu tiên).
"""

import argparse
import logging
import os
from pathlib import Path

from vi_corpus.common.garbled import load_vocab
from vi_corpus.common.ocr_cascade import CascadeOcrGetter
from vi_corpus.ocr_bakeoff.cascade import DEFAULT_TIERS, build_tiers
from vi_corpus.ocr_bakeoff.engines import DEFAULT_ENGINES, load_engines

SYLLABLES_REL = "processed/ocr_bakeoff/syllables.json"  # tập âm tiết cho việc xoay thử trang scan (J-24); giống VJOL

logger = logging.getLogger("vi_corpus")


def add_cascade_args(p: argparse.ArgumentParser) -> None:
    """Thêm nhóm tham số "OCR trang scan (chuỗi engine VLM, O-20)" vào `p`."""
    g = p.add_argument_group("OCR trang scan (chuỗi engine VLM, O-20)")
    g.add_argument("--no-ocr", action="store_true", help="Không OCR: trang scan để needs_ocr (không cần GPU).")
    g.add_argument("--config", type=Path, default=DEFAULT_ENGINES, help="File cấu hình engine.")
    g.add_argument("--tiers", default=",".join(DEFAULT_TIERS),
                   help="Chuỗi engine, cách nhau dấu phẩy (mặc định %(default)s). Các engine VLM phải cùng mô hình.")
    g.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES cho server và engine baseline (vd 0).")
    g.add_argument("--serve", action="store_true", help="Tự dựng / tắt server vLLM (lúc gặp trang scan đầu tiên).")
    g.add_argument("--vllm-bin", default=None, help="Đường dẫn lệnh vllm (môi trường riêng); mặc định 'vllm'.")
    g.add_argument("--port", type=int, default=None, help="Cổng server (mặc định theo cấu hình; mỗi GPU một cổng khác nhau).")
    g.add_argument("--gpu-mem", type=float, default=0.85, help="--gpu-memory-utilization của server (chừa chỗ cho engine baseline).")
    g.add_argument("--concurrency", type=int, default=32, help="Số trang scan gửi đồng thời (mặc định 32, O-22).")
    g.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Thiết bị của engine baseline (mặc định cuda).")
    g.add_argument("--server-timeout", type=float, default=3600, help="Thời gian chờ server sẵn sàng (giây).")
    g.add_argument("--syllables", type=Path, default=None,
                   help="Tập âm tiết để xoay thử trang scan bị xoay (mặc định <data-root>/%s; thiếu file thì không xoay)." % SYLLABLES_REL)
    g.add_argument("--no-rotate", action="store_true", help="Không xoay thử trang scan nghi bị xoay (J-24).")


def build_getter(args: argparse.Namespace, data_root: Path, state_dir: Path) -> CascadeOcrGetter:
    """
    Dựng CascadeOcrGetter từ tham số của add_cascade_args.

    Đặt CUDA_VISIBLE_DEVICES theo --gpu. Chuỗi engine sai tên / sai cấu hình thì ném ValueError (script báo bằng parser.error).

    Args:
        args:       Kết quả parse_args.
        data_root:  Thư mục gốc dữ liệu (tìm tập âm tiết).
        state_dir:  Nơi ghi server.log của vLLM (cần khi --serve).
    """
    tiers = []
    if not args.no_ocr:
        tiers = build_tiers([t.strip() for t in args.tiers.split(",") if t.strip()], load_engines(args.config),
                            port=args.port, device=args.device, gpu_mem=args.gpu_mem)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    vocab = None if args.no_rotate else load_vocab(args.syllables or data_root / SYLLABLES_REL)
    if vocab is None and not args.no_rotate and not args.no_ocr:
        logger.warning("Không thấy tập âm tiết: không xoay thử trang scan bị xoay 180 / 90 / 270 độ (J-24)")
    return CascadeOcrGetter(tiers, args.concurrency, enabled=not args.no_ocr, serve=args.serve, gpu=args.gpu,
                            vllm_bin=args.vllm_bin, state_dir=state_dir, server_timeout=args.server_timeout, vocab=vocab)
