#!/usr/bin/env bash
# Chạy pipeline CHUNG (ingest -> dedup -> finalize) cho VJOL + VISTA chỉ bằng CPU, chia SHARDS phần độc lập và chạy JOBS tiến trình song song.
#
# Gọi qua Makefile:  make pipeline-cpu ROOT=<data-root> [JOBS=128] [SHARDS=128] [SOURCES="vjol vista"] [PIPE_ARGS="..."] [STEP=all]
# Các bước (STEP), theo thứ tự; STEP=all chạy hết, dừng ngay nếu một bước có phần lỗi (xem <ROOT>/logs/pipe_failed_<bước>.txt):
#   shards    ingest -> dedup vòng 1 cho từng phần (--until dedup), song song JOBS tiến trình
#   dedup     dedup vòng 2 toàn cục (scripts/dedup_global.py) gộp mọi phần của mọi nguồn
#   finalize  chạy lại từng phần với --global-verdict tới hết finalize (knowledge units, CPT segments, báo cáo); không embed
#   summary   bảng tổng hợp từng nguồn (đoạn, token, RAM đỉnh, giây) -> <ROOT>/processed/<nguồn>_full_summary.json
# Chạy lại đúng lệnh cũ thì dùng lại kết quả stage đã xong (vân tay stage). PIPE_ARGS phải GIỐNG NHAU giữa các lần chạy.
# Trang needs_ocr bị bỏ qua (VJOL / VISTA: thành trang rỗng): đây là bản CPU thuần, sau OCR GPU phải chạy lại (PP-01).
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-${SEA_DATA_ROOT:-data}}"
SHARDS="${SHARDS:-128}"
JOBS="${JOBS:-128}"
SEED="${SEED:-42}"
SOURCES="${SOURCES:-vjol vista}"
STEP="${1:-all}"
VERDICT="$DATA_ROOT/processed/dedup_global/verdict.parquet"
PY=(uv run --no-sync python)
mkdir -p "$DATA_ROOT/logs"
# Mỗi tiến trình dùng đúng một luồng, tránh JOBS x nhiều luồng chồng nhau.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export DATA_ROOT SHARDS SEED VERDICT PIPE_ARGS="${PIPE_ARGS:-}"

# Chạy pipeline cho một phần. $1 = nguồn, $2 = chỉ số phần. Biến PHASE: shards (--until dedup) hoặc finalize (--global-verdict).
# Phần lỗi được ghi vào $FAILED để bước sau biết dừng; nhật ký từng phần nằm ở <ROOT>/logs/<nguồn>_pipe_s<i>of<N>.log.
run_part() {
  local src="$1" i="$2" extra=() pa=($PIPE_ARGS)
  if [ "$PHASE" = shards ]; then extra=(--until dedup); else extra=(--global-verdict "$VERDICT"); fi
  uv run --no-sync python scripts/run_pipeline.py --data-root "$DATA_ROOT" --mix "$src=1" --take-all --shard "$i/$SHARDS" \
    --seed "$SEED" --embedder none --run-name "${src}_full_s${i}of${SHARDS}" "${extra[@]}" ${pa[@]+"${pa[@]}"} \
    >> "$DATA_ROOT/logs/${src}_pipe_s${i}of${SHARDS}.log" 2>&1 || echo "$src $i" >> "$FAILED"
}
export -f run_part

# Chạy một giai đoạn cho mọi (nguồn, phần), JOBS song song. $1 = shards | finalize. Trả về 1 nếu có phần lỗi.
run_phase() {
  export PHASE="$1" FAILED="$DATA_ROOT/logs/pipe_failed_$1.txt"
  : > "$FAILED"
  echo "=== $1: $SOURCES x $SHARDS phần, $JOBS song song $(date '+%F %T')"
  for src in $SOURCES; do
    for ((i = 0; i < SHARDS; i++)); do echo "$src $i"; done
  done | xargs -P "$JOBS" -L 1 bash -c 'run_part "$0" "$1"'
  if [ -s "$FAILED" ]; then
    echo "!!! $(wc -l < "$FAILED") phần lỗi (danh sách: $FAILED); chạy lại đúng lệnh này để làm tiếp phần lỗi"; return 1
  fi
  echo "=== $1 xong $(date '+%F %T')"
}

# Chạy dedup toàn cục vòng 2 trên mọi lần chạy đã có trong state/dedup_index.
run_dedup() {
  echo "=== dedup toàn cục $(date '+%F %T')"
  "${PY[@]}" scripts/dedup_global.py --data-root "$DATA_ROOT" || return 1
}

# In bảng tổng hợp từng nguồn.
run_summary() {
  for src in $SOURCES; do
    "${PY[@]}" scripts/vista/summarize_runs.py --data-root "$DATA_ROOT" --prefix "${src}_full" \
      --out "$DATA_ROOT/processed/${src}_full_summary.json"
  done
}

case "$STEP" in
  shards)   run_phase shards ;;
  dedup)    run_dedup ;;
  finalize) run_phase finalize ;;
  summary)  run_summary ;;
  all)      run_phase shards && run_dedup && run_phase finalize && run_summary ;;
  *)        sed -n 2,11p "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
