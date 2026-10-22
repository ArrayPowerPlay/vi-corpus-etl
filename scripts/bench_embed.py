"""
Đo riêng tốc độ nhúng (J-22 (8)): vì sao 10.000 đoạn mất 4.252 giây (~2,4 đoạn/giây) trong lần chạy vjol_10k_v2?

Không sửa mù: script in thông tin torch / GPU, các tiến trình đang chiếm GPU (nvidia-smi, vd vLLM còn sót), rồi nhúng N đoạn đầu của một lần chạy
pipeline (02_prepare.parquet) ở từng cỡ lô và in số đoạn mỗi giây. Dùng --device cuda (mặc định) nên không có GPU thì dừng với lỗi thay vì chạy CPU.
Ví dụ (trên server, dừng hết vLLM trước):
    uv run --group viz python scripts/bench_embed.py --run-dir data/processed/pipeline_runs/vjol_10k_v2 --n 640
"""

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # thay thư mục script bằng gốc repo (xem scripts/vjol/extract.py)

from vi_corpus.pipeline.embed import DEFAULT_PROMPT, DEFAULT_SPEC, HfEmbedder  # noqa: E402
from vi_corpus.pipeline.io import read_rows  # noqa: E402


def gpu_processes() -> str:
    """Danh sách tiến trình đang chiếm GPU theo nvidia-smi (chuỗi rỗng nếu không có nvidia-smi)."""
    if not shutil.which("nvidia-smi"):
        return "(không có nvidia-smi)"
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30)
    return out.stdout.strip() or "(không có tiến trình nào chiếm GPU)"


def main() -> int:
    """Đọc tham số, in môi trường, nhúng N đoạn ở từng cỡ lô, in tốc độ. Trả về 0."""
    p = argparse.ArgumentParser(description="Đo tốc độ nhúng riêng (J-22 (8)).")
    p.add_argument("--run-dir", type=Path, required=True, help="Thư mục một lần chạy pipeline (cần 02_prepare.parquet).")
    p.add_argument("--n", type=int, default=640, help="Số đoạn đầu để nhúng.")
    p.add_argument("--model", default=DEFAULT_SPEC.removeprefix("hf:"), help="Mô hình HF (mặc định như pipeline).")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-seq", type=int, default=512)
    p.add_argument("--batch-sizes", default="16,64,128", help="Các cỡ lô thử, cách nhau dấu phẩy.")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import torch
    print(f"torch {torch.__version__}, cuda.is_available={torch.cuda.is_available()}, số GPU={torch.cuda.device_count()}")
    print("Tiến trình đang chiếm GPU:\n" + gpu_processes())
    texts = [r["text"] for r in read_rows(args.run_dir / "02_prepare.parquet")[:args.n]]
    print(f"{len(texts)} đoạn, trung bình {sum(map(len, texts)) / max(len(texts), 1):.0f} ký tự")
    for bs in (int(x) for x in args.batch_sizes.split(",")):
        model = HfEmbedder(args.model, args.device, args.max_seq, bs, args.prompt)
        model.embed(texts[:bs])  # khởi động (nạp kernel), không tính giờ
        t0 = time.time()
        model.embed(texts)
        took = time.time() - t0
        print(f"batch={bs}: {took:.1f} giây, {len(texts) / took:.1f} đoạn/giây, device={model.device}")
        del model
    print("Tiến trình chiếm GPU sau khi đo:\n" + gpu_processes())
    return 0


if __name__ == "__main__":
    sys.exit(main())
