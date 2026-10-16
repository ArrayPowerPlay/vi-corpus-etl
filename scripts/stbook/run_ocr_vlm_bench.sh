#!/usr/bin/env bash
# Thử concurrency + tỷ lệ trang bị báo của chuỗi OCR VLM trên ~1.000 trang stbook thật (O-20), trên server GPU.
# Các bước: (1) chọn 1.000 trang (CPU); (2) mỗi mức concurrency một GPU + một server vLLM riêng, song song, cùng bộ trang;
# (3) chạy lại trang bị bộ kiểm báo bằng qwen3vl_8b_rp110 trên GPU đầu; (4) in bảng tổng hợp. Chạy lại được (chỉ làm trang còn thiếu).
# Dùng (từ gốc repo, sau `source /home/jovyan/dinh/env.sh`; trong tmux hoặc nohup):
#   DATA_ROOT=/duong/dan/data VLLM_BIN=$(which vllm) ./scripts/stbook/run_ocr_vlm_bench.sh 2>&1 | tee ocr_vlm_bench.log
# Biến tuỳ chọn: LEVELS (mặc định "16 32 64 128" -> GPU 0 1 2 3; tối đa 4 mức), PAGES (1000), BOOKS (50), BENCH_DIR, GPU_MEM (0.85).
# Mỗi mức một GPU nên số trang/giây các mức so sánh trực tiếp được; cần các GPU rảnh (kiểm bằng nvidia-smi trước).
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-${SEA_DATA_ROOT:-data}}"
BENCH_DIR="${BENCH_DIR:-$DATA_ROOT/processed/ocr_vlm_bench}"
VLLM_BIN="${VLLM_BIN:-vllm}"
LEVELS="${LEVELS:-16 32 64 128}"
PAGES="${PAGES:-1000}"
BOOKS="${BOOKS:-50}"
GPU_MEM="${GPU_MEM:-0.85}"
PY=(uv run --no-sync python)
S=scripts/stbook/ocr_vlm_bench.py
COMMON=(--data-root "$DATA_ROOT" --out "$BENCH_DIR")
command -v "$VLLM_BIN" > /dev/null || { echo "Không tìm thấy lệnh vLLM '$VLLM_BIN' (source env.sh?)"; exit 1; }
uv sync --group ocr || exit 1
mkdir -p "$BENCH_DIR/logs"
echo "DATA_ROOT=$DATA_ROOT  BENCH_DIR=$BENCH_DIR  LEVELS=$LEVELS  PAGES=$PAGES"

"${PY[@]}" $S sample "${COMMON[@]}" --pages "$PAGES" --books "$BOOKS" || exit 1

STATUS_DIR="$BENCH_DIR/logs/status"; rm -rf "$STATUS_DIR"; mkdir -p "$STATUS_DIR"
gpu=0
for c in $LEVELS; do
  (
    echo "GPU $gpu: concurrency $c (log: $BENCH_DIR/logs/c$c.log)"
    "${PY[@]}" $S run "${COMMON[@]}" --concurrency "$c" --gpu "$gpu" --port $((8200 + gpu)) --serve --vllm-bin "$VLLM_BIN" \
      --gpu-mem "$GPU_MEM" > "$BENCH_DIR/logs/c$c.log" 2>&1
    echo $? > "$STATUS_DIR/c$c"
  ) &
  gpu=$((gpu + 1))
done
wait
for f in "$STATUS_DIR"/*; do
  name=$(basename "$f")
  if [ "$(cat "$f")" != 0 ]; then
    echo "=== $name LỖI (mã thoát $(cat "$f")), 15 dòng cuối log:"; tail -n 15 "$BENCH_DIR/logs/$name.log"
  else
    echo "=== $name xong: $(tail -n 1 "$BENCH_DIR/logs/$name.log")"
  fi
done

"${PY[@]}" $S retry "${COMMON[@]}" --concurrency 16 --gpu 0 --port 8200 --serve --vllm-bin "$VLLM_BIN" --gpu-mem "$GPU_MEM" \
  > "$BENCH_DIR/logs/retry.log" 2>&1 || { echo "=== retry LỖI:"; tail -n 15 "$BENCH_DIR/logs/retry.log"; }
"${PY[@]}" $S report "${COMMON[@]}"
