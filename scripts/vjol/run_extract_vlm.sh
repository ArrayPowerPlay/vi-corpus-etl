#!/usr/bin/env bash
# Trích text VJOL trên nhiều GPU (J-04): mỗi GPU một server vLLM + một tiến trình scripts/vjol/extract.py (cổng 8100 + số GPU).
# Các tiến trình tự chia PDF bằng khoá nên GPU nhanh làm nhiều hơn; chạy lại được (chỉ làm PDF còn thiếu). Server chỉ được dựng khi
# tiến trình gặp trang scan đầu tiên. Xong thì xuất Parquet một lần và in thống kê.
# Dùng (từ gốc repo, sau `source env.sh` nếu server có; trong tmux hoặc nohup):
#   DATA_ROOT=/duong/dan/data VLLM_BIN=$(which vllm) SAMPLE_DOCS=3000 ./scripts/vjol/run_extract_vlm.sh
# Biến tuỳ chọn: GPUS (mặc định "0 1 2 3"), SAMPLE_DOCS (bỏ trống = mọi PDF), SEED (mặc định 0), CONCURRENCY (mặc định 32), LIMIT_FILES, TIERS, GPU_MEM, SYLLABLES (tập âm tiết cho bộ dò chữ hỏng, mặc định $DATA_ROOT/processed/ocr_bakeoff/syllables.json).
# Cần `uv sync --group ocr` (engine baseline); dùng `uv run --no-sync` như run_ocr_vlm.sh để các tiến trình không gỡ - cài thư viện của nhau.
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-${SEA_DATA_ROOT:-data}}"
VLLM_BIN="${VLLM_BIN:-vllm}"
GPUS="${GPUS:-0 1 2 3}"
SEED="${SEED:-0}"
CONCURRENCY="${CONCURRENCY:-32}"
PY=(uv run --no-sync python)
EXTRA=()
[ -n "${SAMPLE_DOCS:-}" ] && EXTRA+=(--sample-docs "$SAMPLE_DOCS")
[ -n "${LIMIT_FILES:-}" ] && EXTRA+=(--limit-files "$LIMIT_FILES")
[ -n "${TIERS:-}" ] && EXTRA+=(--tiers "$TIERS")
[ -n "${GPU_MEM:-}" ] && EXTRA+=(--gpu-mem "$GPU_MEM")
[ -n "${SYLLABLES:-}" ] && EXTRA+=(--syllables "$SYLLABLES")
command -v "$VLLM_BIN" > /dev/null || { echo "Không tìm thấy lệnh vLLM '$VLLM_BIN' (source env.sh?)"; exit 1; }
uv sync --group ocr || exit 1
LOG_DIR="$DATA_ROOT/logs"; STATUS_DIR="$LOG_DIR/status_vjol_extract"
mkdir -p "$LOG_DIR"; rm -rf "$STATUS_DIR"; mkdir -p "$STATUS_DIR"
echo "DATA_ROOT=$DATA_ROOT  GPUS=$GPUS  SAMPLE_DOCS=${SAMPLE_DOCS:-tất cả}  SEED=$SEED  CONCURRENCY=$CONCURRENCY  VLLM_BIN=$VLLM_BIN"

for g in $GPUS; do
  (
    "${PY[@]}" scripts/vjol/extract.py --data-root "$DATA_ROOT" --seed "$SEED" --gpu "$g" --port $((8100 + g)) --serve \
      --vllm-bin "$VLLM_BIN" --concurrency "$CONCURRENCY" --no-export ${EXTRA[@]+"${EXTRA[@]}"} > "$LOG_DIR/vjol_extract_gpu$g.out" 2>&1
    echo $? > "$STATUS_DIR/gpu$g"
  ) &
done
wait
rc=0
for f in "$STATUS_DIR"/*; do
  name=$(basename "$f")
  if [ "$(cat "$f")" != 0 ]; then
    rc=1; echo "=== $name LỖI (mã thoát $(cat "$f")), 15 dòng cuối:"; tail -n 15 "$LOG_DIR/vjol_extract_$name.out"
  else
    echo "=== $name xong"
  fi
done
"${PY[@]}" scripts/vjol/extract.py --data-root "$DATA_ROOT" --export-only
"${PY[@]}" scripts/vjol/extract.py --data-root "$DATA_ROOT" --status
exit $rc
