#!/usr/bin/env bash
# Trích text giáo trình trên nhiều GPU: mỗi GPU một server vLLM + một tiến trình scripts/giao_trinh/extract.py (cổng 8100 + số GPU).
# Các tiến trình tự chia file bằng khoá (--multi-process) nên GPU nhanh làm nhiều hơn; chạy lại được (chỉ làm file còn thiếu, và trang đã OCR
# nằm trong interim/giao_trinh_ocr_cache/ nên cuốn scan dài bị ngắt không phải OCR lại từ đầu). Server chỉ được dựng khi tiến trình gặp trang
# scan đầu tiên. Xong thì xuất Parquet một lần và in thống kê.
# Dùng (từ gốc repo, trong tmux hoặc nohup):
#   DATA_ROOT=/duong/dan/data VLLM_BIN=$(which vllm) ./scripts/giao_trinh/run_extract_vlm.sh
# Biến tuỳ chọn: GPUS (mặc định "0 1 2 3"), CONCURRENCY (mặc định 32), LIMIT_FILES (chạy thử), TIERS, GPU_MEM, SYLLABLES (tập âm tiết để xoay thử
# trang scan bị xoay, mặc định $DATA_ROOT/processed/ocr_bakeoff/syllables.json), RETRY_SKIPPED=1 (làm lại cả file từng bị bỏ qua).
# Cần `uv sync --group ocr` (engine baseline); dùng `uv run --no-sync` như run_ocr_vlm.sh để các tiến trình không gỡ - cài thư viện của nhau.
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-${SEA_DATA_ROOT:-data}}"
VLLM_BIN="${VLLM_BIN:-vllm}"
GPUS="${GPUS:-0 1 2 3}"
CONCURRENCY="${CONCURRENCY:-32}"
PY=(uv run --no-sync python)
EXTRA=()
[ -n "${LIMIT_FILES:-}" ] && EXTRA+=(--limit-files "$LIMIT_FILES")
[ -n "${TIERS:-}" ] && EXTRA+=(--tiers "$TIERS")
[ -n "${GPU_MEM:-}" ] && EXTRA+=(--gpu-mem "$GPU_MEM")
[ -n "${SYLLABLES:-}" ] && EXTRA+=(--syllables "$SYLLABLES")
[ -n "${RETRY_SKIPPED:-}" ] && EXTRA+=(--retry-skipped)
command -v "$VLLM_BIN" > /dev/null || { echo "Không tìm thấy lệnh vLLM '$VLLM_BIN' (đặt VLLM_BIN=/duong/dan/vllm)"; exit 1; }
uv sync --group ocr || exit 1
LOG_DIR="$DATA_ROOT/logs"; STATUS_DIR="$LOG_DIR/status_giao_trinh_extract"
mkdir -p "$LOG_DIR"; rm -rf "$STATUS_DIR"; mkdir -p "$STATUS_DIR"
echo "DATA_ROOT=$DATA_ROOT  GPUS=$GPUS  CONCURRENCY=$CONCURRENCY  VLLM_BIN=$VLLM_BIN"

for g in $GPUS; do
  (
    "${PY[@]}" scripts/giao_trinh/extract.py --data-root "$DATA_ROOT" --gpu "$g" --port $((8100 + g)) --serve --multi-process \
      --vllm-bin "$VLLM_BIN" --concurrency "$CONCURRENCY" --no-export ${EXTRA[@]+"${EXTRA[@]}"} > "$LOG_DIR/giao_trinh_extract_gpu$g.out" 2>&1
    echo $? > "$STATUS_DIR/gpu$g"
  ) &
done
wait
rc=0
for f in "$STATUS_DIR"/*; do
  name=$(basename "$f")
  if [ "$(cat "$f")" != 0 ]; then
    rc=1; echo "=== $name LỖI (mã thoát $(cat "$f")), 15 dòng cuối:"; tail -n 15 "$LOG_DIR/giao_trinh_extract_$name.out"
  else
    echo "=== $name xong"
  fi
done
"${PY[@]}" scripts/giao_trinh/extract.py --data-root "$DATA_ROOT" --export-only
"${PY[@]}" scripts/giao_trinh/extract.py --data-root "$DATA_ROOT" --status
if [ "$rc" != 0 ]; then
  echo "!!! TRÍCH TEXT CHƯA XONG (có tiến trình lỗi): đừng chạy pipeline lúc này, nó sẽ đọc bản trích cũ. Sửa lỗi ở trên rồi chạy lại đúng lệnh cũ."
fi
exit $rc
