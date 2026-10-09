#!/usr/bin/env bash
# Chạy trọn đợt so sánh engine OCR (F-11) trên server 4 x A100: chọn trang, lập tập âm tiết, chạy 5 engine song song
# trên 4 GPU (mỗi engine một GPU; GPU 0 chạy baseline rồi qwen3vl_4b), chấm điểm. Chạy lại được: bước nào đã xong
# (pages.jsonl, syllables.json, trang đã có kết quả) thì bỏ qua.
# Dùng (từ gốc repo, trong tmux hoặc nohup):
#   DATA_ROOT=/duong/dan/data VLLM_BIN=/opt/vllm-env/bin/vllm HOUR_BUDGET=30 ./scripts/ocr_bakeoff/run_bakeoff.sh
# Biến tuỳ chọn: BAKEOFF_DIR (mặc định $DATA_ROOT/processed/ocr_bakeoff), HOUR_BUDGET (giới hạn số giờ server chạy OCR cả kho),
# PADDLE_VL_RUN (lệnh python có paddleocr[doc-parser], mặc định "uv run --group ocr python").
set -uo pipefail

DATA_ROOT="${DATA_ROOT:?Đặt DATA_ROOT=/duong/dan/data}"
BAKEOFF_DIR="${BAKEOFF_DIR:-$DATA_ROOT/processed/ocr_bakeoff}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PADDLE_VL_RUN="${PADDLE_VL_RUN:-uv run --group ocr python}"
COMMON=(--data-root "$DATA_ROOT" --bakeoff-dir "$BAKEOFF_DIR")
mkdir -p "$BAKEOFF_DIR/logs"

if [ ! -f "$BAKEOFF_DIR/pages.jsonl" ]; then
  uv run python scripts/ocr_bakeoff/select_pages.py "${COMMON[@]}" || exit 1
fi
if [ ! -f "$BAKEOFF_DIR/syllables.json" ]; then
  uv run python scripts/ocr_bakeoff/build_syllables.py "${COMMON[@]}" || echo "Không lập được tập âm tiết, chấm tiếp không có số đo này"
fi

run() {  # run <gpu> <tên log> <lệnh python...> : chạy nền, log riêng
  local gpu=$1 log=$2; shift 2
  echo "GPU $gpu: $*"
  "$@" --gpu "$gpu" > "$BAKEOFF_DIR/logs/$log.log" 2>&1
}

( run 0 baseline uv run --group ocr python scripts/ocr_bakeoff/run_engine.py "${COMMON[@]}" --engine baseline
  run 0 qwen3vl_4b uv run python scripts/ocr_bakeoff/run_engine.py "${COMMON[@]}" --engine qwen3vl_4b --serve --vllm-bin "$VLLM_BIN" ) &
( run 1 paddleocr_vl $PADDLE_VL_RUN scripts/ocr_bakeoff/run_engine.py "${COMMON[@]}" --engine paddleocr_vl --serve ) &
( run 2 dots_ocr uv run python scripts/ocr_bakeoff/run_engine.py "${COMMON[@]}" --engine dots_ocr --serve --vllm-bin "$VLLM_BIN" ) &
( run 3 qwen3vl_8b uv run python scripts/ocr_bakeoff/run_engine.py "${COMMON[@]}" --engine qwen3vl_8b --serve --vllm-bin "$VLLM_BIN" ) &
wait
echo "Các engine đã chạy xong (xem $BAKEOFF_DIR/logs/*.log)"

SCORE=(uv run python scripts/ocr_bakeoff/score.py "${COMMON[@]}" --ppl)
HOUR_BUDGET="${HOUR_BUDGET:-${GPU_HOUR_BUDGET:-}}"
if [ -n "$HOUR_BUDGET" ]; then SCORE+=(--hour-budget "$HOUR_BUDGET"); fi
CUDA_VISIBLE_DEVICES=0 "${SCORE[@]}"
