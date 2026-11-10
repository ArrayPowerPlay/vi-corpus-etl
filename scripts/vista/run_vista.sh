#!/usr/bin/env bash
# Chạy từng bước của pipeline VISTA trên server (VS-06). Mỗi lần chạy MỘT bước; bước nào cũng chạy lại được (checkpoint / vân tay stage).
# Dùng (từ gốc repo, sau `source env.sh`, trong tmux):  DATA_ROOT=/duong/dan/data VLLM_BIN=$(which vllm) ./scripts/vista/run_vista.sh <bước>
# Các bước theo thứ tự:
#   check            CPU   so raw/vista với nhật ký tải (thiếu file / sai kích thước)            -> $DATA_ROOT/logs/vista_check_raw.json
#   survey           CPU   khảo sát producer PDF (producer nào còn lớp chữ mất dấu mà luật chưa phủ)
#   sample-extract   GPU   trích text SAMPLE_DOCS (mặc định 3000) PDF chia đều theo năm, 4 GPU
#   sample-pipeline  GPU   pipeline trên mẫu (SAMPLE_TOTAL đoạn, mặc định 20000) có bản đồ embedding -> processed/pipeline_runs/vista_sample/report.html
#   full-extract     GPU   trích text CẢ KHO, 4 GPU (chạy lại tiếp tục từ checkpoint)
#   full-pipeline    CPU   pipeline cả kho, chia SHARDS phần (mặc định 8) chạy tuần tự, --take-all, không embed -> vista_full_s<i>of<N>
#   dedup            CPU   dedup vòng 2 toàn cục (VJOL + VISTA + mọi nguồn có trong state/dedup_index) rồi áp verdict vào từng phần
#   summary          CPU   bảng tổng hợp các phần: đoạn, token giữ lại, phễu, RAM đỉnh, thời gian
# Biến: DATA_ROOT, VLLM_BIN, SHARDS (8), SAMPLE_DOCS (3000), SAMPLE_TOTAL (20000), SEED (42), PIPE_ARGS (tham số thêm cho run_pipeline.py, vd "--perplexity-model Qwen/Qwen3-0.6B"
# hoặc "--executor xenna --num-gpus 4"; phải GIỐNG NHAU ở full-pipeline và dedup), GPUS, CONCURRENCY (truyền cho run_extract_vlm.sh).
# KHÔNG chạy bước GPU (sample-extract, sample-pipeline có embed, full-extract) khi lượt full VJOL còn chạy (VS-02).
set -uo pipefail

STEP="${1:-}"
DATA_ROOT="${DATA_ROOT:-${SEA_DATA_ROOT:-data}}"
SHARDS="${SHARDS:-8}"
SEED="${SEED:-42}"
SAMPLE_DOCS="${SAMPLE_DOCS:-3000}"
SAMPLE_TOTAL="${SAMPLE_TOTAL:-20000}"
PY=(uv run --no-sync python)
PIPE_ARGS=(${PIPE_ARGS:-})
mkdir -p "$DATA_ROOT/logs"

# Chạy pipeline cho phần $1 (có thêm tham số $2... nếu có); tên lần chạy cố định để chạy lại thì dùng lại kết quả stage.
shard_run() {
  local i="$1"; shift
  "${PY[@]}" scripts/run_pipeline.py --data-root "$DATA_ROOT" --mix vista=1 --take-all --shard "$i/$SHARDS" --seed "$SEED" --embedder none \
    --run-name "vista_full_s${i}of${SHARDS}" ${PIPE_ARGS[@]+"${PIPE_ARGS[@]}"} "$@"
}

# Chạy shard_run cho mọi phần, báo phần nào lỗi; trả về 1 nếu có phần lỗi.
all_shards() {
  local rc=0
  for ((i = 0; i < SHARDS; i++)); do
    echo "=== phần $i/$SHARDS $(date '+%F %T')"
    shard_run "$i" "$@" 2>&1 | tee -a "$DATA_ROOT/logs/vista_pipeline_s${i}of${SHARDS}.log"
    if [ "${PIPESTATUS[0]}" != 0 ]; then rc=1; echo "!!! phần $i/$SHARDS LỖI (xem $DATA_ROOT/logs/vista_pipeline_s${i}of${SHARDS}.log)"; fi
  done
  return $rc
}

case "$STEP" in
  check)
    "${PY[@]}" scripts/vista/check_raw.py --data-root "$DATA_ROOT" | tee "$DATA_ROOT/logs/vista_check_raw.json" ;;
  survey)
    "${PY[@]}" scripts/vista/producer_survey.py --data-root "$DATA_ROOT" --sample 20000 ;;
  sample-extract)
    SAMPLE_DOCS="$SAMPLE_DOCS" DATA_ROOT="$DATA_ROOT" ./scripts/vista/run_extract_vlm.sh ;;
  sample-pipeline)
    "${PY[@]}" scripts/run_pipeline.py --data-root "$DATA_ROOT" --mix vista=1 --total "$SAMPLE_TOTAL" --seed "$SEED" --run-name vista_sample \
      ${PIPE_ARGS[@]+"${PIPE_ARGS[@]}"} ;;
  full-extract)
    unset SAMPLE_DOCS; DATA_ROOT="$DATA_ROOT" ./scripts/vista/run_extract_vlm.sh ;;
  full-pipeline)
    all_shards; exit $? ;;
  dedup)
    "${PY[@]}" scripts/dedup_global.py --data-root "$DATA_ROOT" || exit 1
    all_shards --global-verdict "$DATA_ROOT/processed/dedup_global/verdict.parquet"; exit $? ;;
  summary)
    "${PY[@]}" scripts/vista/summarize_runs.py --data-root "$DATA_ROOT" --prefix vista_full --out "$DATA_ROOT/processed/vista_full_summary.json" ;;
  *)
    sed -n 2,20p "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
