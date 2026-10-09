"""
Chạy pipeline end-to-end (ingest -> prepare -> language -> quality -> dedup -> embed -> reduce -> finalize) trên một mẫu
N bản ghi từ data/raw, để kiểm tra chất lượng pipeline và xem bản đồ embedding. Chi tiết các stage: vi_corpus/pipeline/runner.py.

Chạy toàn bộ (10.000 mẫu), tuần tự trong một tiến trình:
    uv run python scripts/run_pipeline.py --data-root /duong/dan/data --total 10000
Chạy từng bước (mỗi lệnh dừng sau stage chọn, lệnh sau làm tiếp từ checkpoint):
    uv run python scripts/run_pipeline.py --data-root ... --total 10000 --until prepare
    uv run python scripts/run_pipeline.py --data-root ... --total 10000 --until quality
    uv run python scripts/run_pipeline.py --data-root ... --total 10000            # làm nốt các stage còn lại
Chạy song song 4 GPU (NeMo Curator + Ray; cần `uv sync --group curator --group viz`):
    uv run python scripts/run_pipeline.py --data-root ... --total 10000 --executor xenna --num-gpus 4
Thử nhanh trên máy không có mạng / GPU (đếm từ thay token, heuristic thay fastText, nhúng tfidf):
    uv run python scripts/run_pipeline.py --data-root ... --total 2000 --tokenizer words --lang-model heuristic --embedder tfidf
Đổi tham số rồi chạy lại cùng lệnh: stage nào có tham số / đầu vào đổi thì tự chạy lại từ đó (vân tay stage);
--keep-stale để cố ý dùng kết quả cũ.
Kết quả ở <data-root>/processed/pipeline_runs/<run-name>/ (mở report.html để xem). Kho dấu vân tay dedup vòng 1 ghi ở
<data-root>/state/dedup_index/; dedup vòng 2 giữa mọi nguồn: scripts/dedup_global.py, rồi chạy lại với --global-verdict.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được vi_corpus

from vi_corpus.common.registry import load_sources  # noqa: E402
from vi_corpus.common.state import acquire_run_lock, setup_logging  # noqa: E402
from vi_corpus.pipeline.config import (DEFAULT_LANG_MODEL, DEFAULT_MIX, DEFAULT_TOKENIZER, RunConfig,  # noqa: E402
                                       scale_mix)
from vi_corpus.pipeline.reduce import CLUSTER_SPACES  # noqa: E402
from vi_corpus.pipeline.runner import STAGES, run_pipeline  # noqa: E402

EXECUTORS = ("xenna", "ray_actor_pool", "ray_data")


def parse_mix(text: str | None) -> dict[str, int]:
    """Đọc 'nguồn=trọng số,...' thành dict; None thì dùng DEFAULT_MIX."""
    if not text:
        return dict(DEFAULT_MIX)
    return {k.strip(): int(v) for k, v in (item.split("=") for item in text.split(","))}


def build_parser() -> argparse.ArgumentParser:
    """Dựng bộ đọc tham số dòng lệnh."""
    p = argparse.ArgumentParser(description="Chạy pipeline end-to-end trên mẫu N bản ghi.")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")))
    p.add_argument("--total", type=int, default=10_000, help="Tổng số mẫu (nguồn sách tính theo số đoạn).")
    p.add_argument("--mix", default=None, help=f"Trọng số nguồn, vd sea_pile_v2=30,stbook=25 (mặc định {DEFAULT_MIX}).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None, help="Tên thư mục kết quả (mặc định run_<total>_seed<seed>).")
    p.add_argument("--until", choices=STAGES, default=None, help="Dừng sau stage này (chạy từng bước).")
    p.add_argument("--force-from", choices=STAGES, default=None, help="Xoá kết quả từ stage này và chạy lại.")
    p.add_argument("--keep-stale", action="store_true",
                   help="Dùng lại kết quả stage dù tham số / đầu vào đã đổi (chỉ cảnh báo).")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help="Bộ đếm token: 'hf:<repo>', 'tiktoken:<enc>' hoặc 'words' (mặc định %(default)s).")
    p.add_argument("--compare-tokenizers", default="",
                   help="Các tokenizer khác (cách nhau bằng dấu phẩy) chỉ để so tổng token giữ lại trong báo cáo.")
    p.add_argument("--lang-model", default=DEFAULT_LANG_MODEL,
                   help="Nhận diện ngôn ngữ: 'fasttext:lid.176' (tự tải), 'fasttext:<file.bin>' hoặc 'heuristic'.")
    p.add_argument("--rights-gate", choices=["tag", "enforce"], default="tag")
    p.add_argument("--no-dedup-index", action="store_true", help="Không ghi kho dấu vân tay dedup vòng 1 (state/dedup_index/).")
    p.add_argument("--global-verdict", type=Path, default=None,
                   help="verdict.parquet của dedup vòng 2 (scripts/dedup_global.py), áp vào trước khi sinh knowledge unit.")
    p.add_argument("--cpt-context", type=int, default=4096, help="Độ dài tối đa (token) một khối CPT khi nối đoạn.")
    p.add_argument("--ocr-books", type=int, default=0, help="OCR trước N cuốn stbook chưa OCR (cần --group ocr và GPU).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Thiết bị OCR.")
    g = p.add_argument_group("embedding / bản đồ")
    g.add_argument("--embedder", default="hf:intfloat/multilingual-e5-base",
                   help="'hf:<model_id>' (GPU), 'tfidf' (CPU, thử nhanh, không song song) hoặc 'none' để bỏ embed/reduce/bản đồ.")
    g.add_argument("--embed-prompt", default="passage: ", help="Tiền tố trước văn bản khi nhúng (E5 cần 'passage: ').")
    g.add_argument("--embed-batch-size", type=int, default=64)
    g.add_argument("--embed-scope", choices=["all", "kept"], default="all",
                   help="all: nhúng mọi bản ghi (bản đồ thấy cả bản bị loại); kept: chỉ phần còn sống (R-05, chạy lớn).")
    g.add_argument("--no-gpu-reduce", action="store_true", help="Giảm chiều bằng scikit-learn/umap-learn dù có cuML.")
    g.add_argument("--cluster-space", choices=CLUSTER_SPACES, default="raw",
                   help="Không gian gom cụm HDBSCAN: raw = embedding gốc (mặc định), pca50 / umap10 khi raw quá chậm.")
    g.add_argument("--min-cluster-size", type=int, default=None,
                   help="HDBSCAN min_cluster_size (mặc định max(2, min(20, n // 10)) như ViLA).")
    g.add_argument("--min-samples", type=int, default=None, help="HDBSCAN min_samples (mặc định để thư viện tự đặt).")
    g = p.add_argument_group("chạy song song (NeMo Curator + Ray)")
    g.add_argument("--executor", choices=EXECUTORS, default=None,
                   help="Bật chạy song song language/quality/embed/OCR bằng executor này (mặc định: tuần tự).")
    g.add_argument("--num-gpus", type=int, default=None, help="Số GPU Ray được dùng (mặc định: tự phát hiện).")
    g.add_argument("--num-cpus", type=int, default=None, help="Số CPU Ray được dùng (mặc định: tự phát hiện).")
    g.add_argument("--ray-address", default=None, help="Nối vào cụm Ray có sẵn (vd auto) thay vì chạy Ray cục bộ.")
    g.add_argument("--rows-per-task", type=int, default=500, help="Số bản ghi mỗi phân vùng gửi cho một actor.")
    g.add_argument("--gpus-per-worker", type=float, default=1.0, help="Phần GPU mỗi actor giữ (0.5 = 2 actor / GPU).")
    return p


def main() -> int:
    """Đọc tham số, (tuỳ chọn) OCR stbook, rồi chạy pipeline. Trả về 1 nếu kết quả không có bản ghi nào."""
    args = build_parser().parse_args()
    specs = load_sources(Path("configs/sources.json"))
    mix = scale_mix(parse_mix(args.mix), args.total)
    run_dir = args.data_root / "processed" / "pipeline_runs" / (args.run_name or f"run_{args.total}_seed{args.seed}")
    _lock = acquire_run_lock(args.data_root / "state" / f"pipeline_{run_dir.name}")  # noqa: F841 — giữ khoá
    setup_logging(args.data_root / "logs", "pipeline")

    cfg = RunConfig(mix=mix, seed=args.seed, tokenizer=args.tokenizer,
                    compare_tokenizers=tuple(t.strip() for t in args.compare_tokenizers.split(",") if t.strip()),
                    lang_model=args.lang_model, rights_gate=args.rights_gate, dedup_index=not args.no_dedup_index,
                    global_verdict=str(args.global_verdict) if args.global_verdict else None,
                    cpt_context_tokens=args.cpt_context,
                    embed_spec=None if args.embedder == "none" else args.embedder, embed_prompt=args.embed_prompt,
                    embed_batch_size=args.embed_batch_size, embed_scope=args.embed_scope,
                    prefer_gpu=not args.no_gpu_reduce, cluster_space=args.cluster_space,
                    hdbscan_min_cluster_size=args.min_cluster_size, hdbscan_min_samples=args.min_samples)

    backend = None
    if args.executor:
        from vi_corpus.pipeline.curator import CuratorBackend  # import muộn: cần `uv sync --group curator`
        backend = CuratorBackend(args.executor, args.num_gpus, args.num_cpus, args.ray_address,
                                 args.rows_per_task, args.gpus_per_worker)
        backend.__enter__()
    try:
        if args.ocr_books and "stbook" in mix:
            stbook_root = args.data_root / specs["stbook"].path
            if backend:
                from vi_corpus.stbook.ocr_parallel import run_ocr_parallel
                run_ocr_parallel(stbook_root, args.data_root, backend, args.device, args.ocr_books)
            else:
                from vi_corpus.stbook.ocr_books import run_ocr
                run_ocr(stbook_root, args.data_root, args.device, args.ocr_books)
        manifest = run_pipeline(specs, args.data_root, run_dir, cfg, force_from=args.force_from,
                                until=args.until, backend=backend, keep_stale=args.keep_stale)
    finally:
        if backend:
            backend.__exit__(None, None, None)
    if "finalize" not in manifest:
        print(f"\nĐã dừng sau stage '{args.until}'. Chạy lại cùng lệnh (bỏ --until) để làm tiếp. Thư mục: {run_dir}")
        return 0
    print(f"\nXong. Mở: {run_dir / 'report.html'}")
    return 0 if manifest["finalize"]["clean_rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
