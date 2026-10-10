#!/usr/bin/env bash
# Chạy trọn đợt so sánh engine OCR (F-11) trên server 4 x A100: chọn trang, lập tập âm tiết, chạy 5 engine song song
# trên 4 GPU (mỗi engine một GPU; GPU 0 chạy baseline rồi qwen3vl_4b), chấm điểm. Chạy lại được: bước nào đã xong
# (pages.jsonl, syllables.json, trang đã có kết quả) thì bỏ qua.
# Dùng (từ gốc repo, trong tmux hoặc nohup):
#   DATA_ROOT=/duong/dan/data VLLM_BIN=/opt/vllm-env/bin/vllm HOUR_BUDGET=30 ./scripts/ocr_bakeoff/run_bakeoff.sh
# Biến phải đặt trên CÙNG dòng lệnh (như trên) hoặc bằng `export`; đặt ở dòng riêng không có export thì script không thấy.
# DATA_ROOT mặc định giống các script khác: biến SEA_DATA_ROOT, không có thì ./data.
# Biến tuỳ chọn: BAKEOFF_DIR (mặc định $DATA_ROOT/processed/ocr_bakeoff), HOUR_BUDGET (giới hạn số giờ server chạy OCR cả kho),
# PADDLE_VL_RUN (lệnh python có paddleocr[doc-parser], mặc định "uv run --no-sync python").
# Môi trường: cài MỘT lần đầu script (`uv sync --group ocr`, gồm cả group mặc định), sau đó mọi lệnh dùng
# `uv run --no-sync`. Không làm vậy thì các tiến trình song song `uv run` / `uv run --group ocr` gỡ - cài torch của nhau.
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-${SEA_DATA_ROOT:-data}}"
BAKEOFF_DIR="${BAKEOFF_DIR:-$DATA_ROOT/processed/ocr_bakeoff}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PADDLE_VL_RUN="${PADDLE_VL_RUN:-uv run --no-sync python}"
PY=(uv run --no-sync python)
COMMON=(--data-root "$DATA_ROOT" --bakeoff-dir "$BAKEOFF_DIR")
echo "DATA_ROOT=$DATA_ROOT  BAKEOFF_DIR=$BAKEOFF_DIR  VLLM_BIN=$VLLM_BIN"
if [ ! -d "$DATA_ROOT/raw" ]; then
  echo "Không thấy $DATA_ROOT/raw: đặt DATA_ROOT đúng thư mục dữ liệu (cùng dòng lệnh hoặc export)"; exit 1
fi
if ! command -v "$VLLM_BIN" > /dev/null; then
  echo "Không tìm thấy lệnh vLLM '$VLLM_BIN': dots_ocr, qwen3vl_8b, qwen3vl_4b sẽ lỗi (xem docs/README.md Phần F để cài)"
fi
mkdir -p "$BAKEOFF_DIR/logs"

uv sync --group ocr || { echo "uv sync --group ocr lỗi: sửa môi trường trước (xem docs/README.md Phần F)"; exit 1; }
"${PY[@]}" -c "import torch, paddle, vietocr, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || { echo "Môi trường thiếu torch / paddle / vietocr / transformers sau uv sync"; exit 1; }

if [ ! -f "$BAKEOFF_DIR/pages.jsonl" ]; then
  "${PY[@]}" scripts/ocr_bakeoff/select_pages.py "${COMMON[@]}" || exit 1
fi
if ! grep -q '"set": "A"' "$BAKEOFF_DIR/pages.jsonl"; then
  echo "pages.jsonl không có trang bộ A (bộ có đáp án): chạy lại select_pages.py --overwrite sau khi có PDF giáo trình"; exit 1
fi
if [ ! -f "$BAKEOFF_DIR/syllables.json" ]; then
  "${PY[@]}" scripts/ocr_bakeoff/build_syllables.py "${COMMON[@]}" || echo "Không lập được tập âm tiết, chấm tiếp không có số đo này"
fi

STATUS_DIR="$BAKEOFF_DIR/logs/status"
rm -rf "$STATUS_DIR"; mkdir -p "$STATUS_DIR"
run() {  # run <gpu> <tên log> <lệnh python...> : log riêng, ghi mã thoát vào logs/status/<tên>
  local gpu=$1 log=$2; shift 2
  echo "GPU $gpu: $* (log: $BAKEOFF_DIR/logs/$log.log)"
  "$@" --gpu "$gpu" > "$BAKEOFF_DIR/logs/$log.log" 2>&1
  echo $? > "$STATUS_DIR/$log"
}

ENGINE=(scripts/ocr_bakeoff/run_engine.py "${COMMON[@]}" --engine)
( run 0 baseline "${PY[@]}" "${ENGINE[@]}" baseline
  run 0 qwen3vl_4b "${PY[@]}" "${ENGINE[@]}" qwen3vl_4b --serve --vllm-bin "$VLLM_BIN" ) &
( run 1 paddleocr_vl $PADDLE_VL_RUN "${ENGINE[@]}" paddleocr_vl --serve ) &
( run 2 dots_ocr "${PY[@]}" "${ENGINE[@]}" dots_ocr --serve --vllm-bin "$VLLM_BIN" ) &
( run 3 qwen3vl_8b "${PY[@]}" "${ENGINE[@]}" qwen3vl_8b --serve --vllm-bin "$VLLM_BIN" ) &
wait
FAILED=0
for f in "$STATUS_DIR"/*; do
  name=$(basename "$f")
  if [ "$(cat "$f")" != 0 ]; then
    FAILED=1
    echo "=== $name LỖI (mã thoát $(cat "$f")), 15 dòng cuối $BAKEOFF_DIR/logs/$name.log:"
    tail -n 15 "$BAKEOFF_DIR/logs/$name.log"
  else
    echo "=== $name xong: $(tail -n 1 "$BAKEOFF_DIR/logs/$name.log")"
  fi
done
[ "$FAILED" = 0 ] || echo "Có engine lỗi; sửa rồi chạy lại script (trang đã xong được giữ). Vẫn chấm điểm phần đã có."

SCORE=("${PY[@]}" scripts/ocr_bakeoff/score.py "${COMMON[@]}" --ppl)
HOUR_BUDGET="${HOUR_BUDGET:-${GPU_HOUR_BUDGET:-}}"
if [ -n "$HOUR_BUDGET" ]; then SCORE+=(--hour-budget "$HOUR_BUDGET"); fi
CUDA_VISIBLE_DEVICES=0 "${SCORE[@]}"
