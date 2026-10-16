#!/usr/bin/env bash
# OCR toàn bộ sách stbook bằng chuỗi engine VLM (O-20) trên nhiều GPU: mỗi GPU một server vLLM + một tiến trình scripts/stbook/ocr_vlm.py
# (cổng 8100 + số GPU). Các tiến trình tự chia cuốn bằng khoá nên GPU nhanh làm nhiều hơn; chạy lại được (chỉ làm cuốn còn thiếu).
# Dùng (từ gốc repo, sau `source /home/jovyan/dinh/env.sh`; trong tmux hoặc nohup):
#   DATA_ROOT=/duong/dan/data VLLM_BIN=$(which vllm) CONCURRENCY=32 ./scripts/stbook/run_ocr_vlm.sh
# Biến tuỳ chọn: GPUS (mặc định "0 1 2 3"), CONCURRENCY (mặc định 16), LIMIT_BOOKS (chạy thử), TIERS, GPU_MEM, STBOOK_ROOT.
# Cần `uv sync --group ocr` (engine baseline); dùng `uv run --no-sync` như run_bakeoff.sh để các tiến trình không gỡ - cài thư viện của nhau.
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-${SEA_DATA_ROOT:-data}}"
VLLM_BIN="${VLLM_BIN:-vllm}"
GPUS="${GPUS:-0 1 2 3}"
CONCURRENCY="${CONCURRENCY:-16}"
PY=(uv run --no-sync python)
EXTRA=()
[ -n "${LIMIT_BOOKS:-}" ] && EXTRA+=(--limit-books "$LIMIT_BOOKS")
[ -n "${TIERS:-}" ] && EXTRA+=(--tiers "$TIERS")
[ -n "${GPU_MEM:-}" ] && EXTRA+=(--gpu-mem "$GPU_MEM")
[ -n "${STBOOK_ROOT:-}" ] && EXTRA+=(--stbook-root "$STBOOK_ROOT")
command -v "$VLLM_BIN" > /dev/null || { echo "Không tìm thấy lệnh vLLM '$VLLM_BIN' (source env.sh?)"; exit 1; }
uv sync --group ocr || exit 1
LOG_DIR="$DATA_ROOT/logs"; STATUS_DIR="$LOG_DIR/status_ocr_vlm"
mkdir -p "$LOG_DIR"; rm -rf "$STATUS_DIR"; mkdir -p "$STATUS_DIR"
echo "DATA_ROOT=$DATA_ROOT  GPUS=$GPUS  CONCURRENCY=$CONCURRENCY  VLLM_BIN=$VLLM_BIN"

for g in $GPUS; do
  (
    "${PY[@]}" scripts/stbook/ocr_vlm.py --data-root "$DATA_ROOT" --gpu "$g" --port $((8100 + g)) --serve \
      --vllm-bin "$VLLM_BIN" --concurrency "$CONCURRENCY" ${EXTRA[@]+"${EXTRA[@]}"} > "$LOG_DIR/stbook_ocr_vlm_gpu$g.out" 2>&1
    echo $? > "$STATUS_DIR/gpu$g"
  ) &
done
wait
rc=0
for f in "$STATUS_DIR"/*; do
  name=$(basename "$f")
  if [ "$(cat "$f")" != 0 ]; then
    rc=1; echo "=== $name LỖI (mã thoát $(cat "$f")), 15 dòng cuối:"; tail -n 15 "$LOG_DIR/stbook_ocr_vlm_$name.out"
  else
    echo "=== $name xong"
  fi
done
"${PY[@]}" scripts/stbook/ocr_vlm.py --data-root "$DATA_ROOT" --status
exit $rc
