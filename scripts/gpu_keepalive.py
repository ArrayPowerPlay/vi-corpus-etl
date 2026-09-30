"""Giữ workload Run:ai không bị tự dừng do "GPU idle" trong lúc chạy job dài không dùng GPU.

Các job như crawl stbook, tải SEA (scripts/stbook/crawl.py, scripts/sea/...) hoàn toàn không dùng GPU (chỉ CPU + network),
nhưng chạy trên workspace Jupyter của cluster Run:ai (ailab.vnptai.io) —
nơi có chính sách tự dừng workload khi GPU không được sử dụng trong một
khoảng thời gian, bất kể CPU/network vẫn đang bận. Script này chạy một
phép nhân ma trận nhỏ trên GPU theo chu kỳ, chỉ để giữ GPU utilization > 0
trong lúc job crawl dài chạy nền — không phục vụ mục đích tính toán nào
khác. Dừng script này (Ctrl+C hoặc kill tiến trình) ngay khi crawler đã
chạy xong, không cần giữ GPU bận thêm.

Cách dùng: chạy song song với crawler trên cùng workspace Run:ai, ví dụ
qua scripts/run_with_gpu_keepalive.sh (xem file đó) hoặc thủ công:

    nohup python scripts/gpu_keepalive.py > gpu_keepalive.log 2>&1 &
"""

from __future__ import annotations

import time

INTERVAL_SECONDS = 600  # 10 phút giữa 2 lần "đánh thức" GPU
MATRIX_SIZE = 2048


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Không tìm thấy GPU khả dụng (torch.cuda.is_available() == False).")

    device = torch.device("cuda")
    print(f"[gpu_keepalive] Bắt đầu, mỗi {INTERVAL_SECONDS}s nhân 1 ma trận {MATRIX_SIZE}x{MATRIX_SIZE} trên GPU.")
    while True:
        a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        result = (a @ b).sum().item()
        print(f"[gpu_keepalive] {time.strftime('%Y-%m-%d %H:%M:%S')} nhịp GPU OK (checksum={result:.2f})")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
