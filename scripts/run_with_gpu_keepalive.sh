#!/usr/bin/env bash
# Chạy một script Python bất kỳ của repo (mọi tham số được chuyển thẳng cho `python`)
# song song với gpu_keepalive.py, cả hai chạy nền (nohup) để sống sót qua việc đóng
# terminal/tab Jupyter. Khi job chạy xong (hoặc bị dừng), gpu_keepalive tự động bị
# kill theo — không giữ GPU bận vô ích. Dùng cho các job không dùng GPU (crawl stbook,
# tải SEA, xử lý giáo trình) trên cluster Run:ai, nơi workspace tự dừng khi GPU nhàn rỗi.
#
# Cách dùng (chạy từ thư mục gốc repo, sau khi `source .venv/bin/activate`):
#   ./scripts/run_with_gpu_keepalive.sh scripts/stbook/crawl.py --download-pdf
#   ./scripts/run_with_gpu_keepalive.sh scripts/sea/download_all.py --data-root /duong/dan/data
# Log của job: logs/<tên-script>_output.log (vd logs/crawl_output.log).
set -euo pipefail
cd "$(dirname "$0")/.."

JOB_NAME="$(basename "${1:?Thiếu đường dẫn script cần chạy}" .py)"
mkdir -p logs
nohup python scripts/gpu_keepalive.py > logs/gpu_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
echo "GPU keepalive đã chạy nền, PID=$KEEPALIVE_PID (log: logs/gpu_keepalive.log)"

# Job được khởi động BÊN TRONG subshell này (không phải ở shell cha) để JOB_PID là
# con trực tiếp của subshell — nếu không, `wait` bên dưới sẽ báo lỗi "pid ... is not
# a child of this shell" và thoát ngay, khiến GPU keepalive bị kill ngay sau khi khởi động.
(
  nohup python "$@" > "logs/${JOB_NAME}_output.log" 2>&1 &
  JOB_PID=$!
  echo "Job $JOB_NAME đã chạy nền, PID=$JOB_PID (log: logs/${JOB_NAME}_output.log)"
  wait "$JOB_PID"
  echo "Job $JOB_NAME đã dừng, tắt GPU keepalive (PID=$KEEPALIVE_PID)."
  kill "$KEEPALIVE_PID" 2>/dev/null || true
) > "logs/${JOB_NAME}_watcher.log" 2>&1 &

disown -a
echo "Xong. Cả 2 tiến trình chạy độc lập với terminal này, có thể đóng terminal an toàn."
echo "Output của job xem trong logs/${JOB_NAME}_output.log."
