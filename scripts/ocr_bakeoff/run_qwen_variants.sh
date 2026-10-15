#!/usr/bin/env bash
# Thử các biến thể của qwen3vl_8b để giảm vòng lặp / cắt cụt (O-14 iv, O-16, O-17), song song trên GPU 1-3 (mặc định 3 biến thể
# đầu; biến thể khác: VARIANTS="tên1 tên2 tên3"):
#   qwen3vl_8b_stop       : max_tokens 4096 + chuỗi dừng <img / data:image / ![ (cắt gốc lỗi nhúng ảnh base64, O-17)
#   qwen3vl_8b_stop_rp105 : như trên + repetition_penalty 1,05 (sửa thêm vòng lặp dấu chấm)
#   qwen3vl_8b_stop_rp110 : như trên + repetition_penalty 1,10
#   (lượt O-16, đã chạy: qwen3vl_8b_mt4k, qwen3vl_8b_rp105, qwen3vl_8b_rp110: không có chuỗi dừng)
# Mỗi biến thể chạy đủ bộ A + B (giống qwen3vl_8b), chấm cùng baseline / qwen3vl_8b. Chạy lại được (chỉ làm trang còn thiếu).
# Dùng (từ gốc repo, sau `source /home/jovyan/dinh/env.sh`; trong tmux hoặc nohup):
#   DATA_ROOT=/duong/dan/data VLLM_BIN=$(which vllm) ./scripts/ocr_bakeoff/run_qwen_variants.sh
# Biến tuỳ chọn: BAKEOFF_DIR, HOUR_BUDGET (mặc định 60), CONCURRENCY (mặc định 16), VARIANTS (danh sách, cách nhau dấu cách).
# Cần bộ trang và baseline / qwen3vl_8b đã chạy (run_bakeoff.sh). Dùng `uv run --no-sync` như run_bakeoff.sh.
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-${SEA_DATA_ROOT:-data}}"
BAKEOFF_DIR="${BAKEOFF_DIR:-$DATA_ROOT/processed/ocr_bakeoff}"
VLLM_BIN="${VLLM_BIN:-vllm}"
CONCURRENCY="${CONCURRENCY:-16}"
VARIANTS="${VARIANTS:-qwen3vl_8b_stop qwen3vl_8b_stop_rp105 qwen3vl_8b_stop_rp110}"
PY=(uv run --no-sync python)
COMMON=(--data-root "$DATA_ROOT" --bakeoff-dir "$BAKEOFF_DIR")
echo "DATA_ROOT=$DATA_ROOT  BAKEOFF_DIR=$BAKEOFF_DIR  VLLM_BIN=$VLLM_BIN  VARIANTS=$VARIANTS"
[ -f "$BAKEOFF_DIR/pages.jsonl" ] || { echo "Không thấy $BAKEOFF_DIR/pages.jsonl: chạy run_bakeoff.sh trước"; exit 1; }
command -v "$VLLM_BIN" > /dev/null || { echo "Không tìm thấy lệnh vLLM '$VLLM_BIN' (source env.sh?)"; exit 1; }
mkdir -p "$BAKEOFF_DIR/logs"
STATUS_DIR="$BAKEOFF_DIR/logs/status_variants"
rm -rf "$STATUS_DIR"; mkdir -p "$STATUS_DIR"

gpu=1
for v in $VARIANTS; do
  (
    echo "GPU $gpu: $v (log: $BAKEOFF_DIR/logs/$v.log)"
    "${PY[@]}" scripts/ocr_bakeoff/run_engine.py "${COMMON[@]}" --engine "$v" --serve --vllm-bin "$VLLM_BIN" \
      --concurrency "$CONCURRENCY" --gpu "$gpu" > "$BAKEOFF_DIR/logs/$v.log" 2>&1
    echo $? > "$STATUS_DIR/$v"
  ) &
  gpu=$((gpu + 1))
done
wait
for f in "$STATUS_DIR"/*; do
  name=$(basename "$f")
  if [ "$(cat "$f")" != 0 ]; then
    echo "=== $name LỖI (mã thoát $(cat "$f")), 15 dòng cuối log:"; tail -n 15 "$BAKEOFF_DIR/logs/$name.log"
  else
    echo "=== $name xong: $(tail -n 1 "$BAKEOFF_DIR/logs/$name.log")"
  fi
done

CUDA_VISIBLE_DEVICES=0 "${PY[@]}" scripts/ocr_bakeoff/score.py "${COMMON[@]}" --ppl --hour-budget "${HOUR_BUDGET:-60}"
