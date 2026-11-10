"""
Chuẩn bị dữ liệu huấn luyện CPT (T-07, T-08, T-09): cắt + ghép cpt_segments.parquet của một hay nhiều lần chạy pipeline
thành chuỗi độ dài cố định (mặc định 8.192 token), best-fit packing, EOS chỉ ở cuối thật của văn bản. Chi tiết:
vi_corpus/pipeline/cpt_pack.py. Chỉ dùng CPU, đọc từng phần nên chạy được với hàng tỷ token.

    uv run python scripts/cpt_pack.py --segments <run_dir>/cpt_segments.parquet [<run_dir2>/cpt_segments.parquet ...] \\
        --out <data-root>/processed/cpt_packed/train_8k.parquet [--seq-len 8192] [--buckets 0-89] [--use-metadata-prefix]

--use-metadata-prefix TẮT mặc định (T-09): bật thì mỗi mảnh có cột "prefix" (tiêu đề, tạp chí, năm, ngành, môn...) và
prefix_tokens; cần tokenizer (--tokenizer, mặc định Qwen3 như D-01) để đếm token tiền tố. Thống kê ghi cạnh file đầu ra
(<out>.stats.json). Exit 1 nếu không có segment nào.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được vi_corpus

from vi_corpus.common.state import setup_logging, write_json_atomic  # noqa: E402
from vi_corpus.pipeline.cpt_pack import DEFAULT_BUFFER_TOKENS, DEFAULT_SEQ_LEN, pack, parse_buckets  # noqa: E402

logger = logging.getLogger("vi_corpus")


def main() -> int:
    """Đọc tham số, ghép chuỗi, ghi Parquet + thống kê. Trả về 1 nếu đầu vào rỗng hoặc thiếu file."""
    p = argparse.ArgumentParser(description="Cắt + ghép cpt_segments.parquet thành chuỗi CPT (best-fit packing).")
    p.add_argument("--segments", type=Path, nargs="+", required=True, help="Một hay nhiều file cpt_segments.parquet.")
    p.add_argument("--out", type=Path, required=True, help="File Parquet đầu ra (mỗi dòng một mảnh, xem PACKED_SCHEMA).")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Độ dài chuỗi (token), mặc định 8.192 (T-08).")
    p.add_argument("--buckets", default="", help="Chỉ lấy split_bucket A-B (vd 0-89 cho train), mặc định mọi bucket.")
    p.add_argument("--use-metadata-prefix", action="store_true",
                   help="Bật tiền tố metadata ở cột prefix (mặc định TẮT, T-09).")
    p.add_argument("--tokenizer", default="hf:Qwen/Qwen3-0.6B",
                   help="Tokenizer đếm token tiền tố (chỉ dùng khi --use-metadata-prefix), spec như run_pipeline.py.")
    p.add_argument("--buffer-tokens", type=int, default=DEFAULT_BUFFER_TOKENS,
                   help="Cỡ vùng đệm best-fit (token); lớn hơn = chuỗi đầy hơn, tốn RAM hơn.")
    p.add_argument("--log-dir", type=Path, default=None, help="Thư mục log (mặc định cạnh file đầu ra).")
    args = p.parse_args()
    setup_logging(args.log_dir or args.out.parent, "cpt_pack")
    missing = [s for s in args.segments if not s.is_file()]
    if missing:
        logger.error("Không thấy file: %s", ", ".join(map(str, missing)))
        return 1
    count = None
    if args.use_metadata_prefix:
        from vi_corpus.pipeline.tokens import get_counter

        count = get_counter(args.tokenizer).count_batch
    stats = pack(args.segments, args.out, args.seq_len, args.use_metadata_prefix, count, parse_buckets(args.buckets),
                 args.buffer_tokens)
    stats.update({"segments_files": [str(s) for s in args.segments], "seq_len": args.seq_len, "buckets": args.buckets or "all",
                  "use_metadata_prefix": args.use_metadata_prefix})
    write_json_atomic(args.out.with_name(args.out.name + ".stats.json"), stats)
    if not stats["segments"]:
        logger.error("Không có segment nào (kiểm tra --segments / --buckets)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
