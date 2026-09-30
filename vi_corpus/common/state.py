"""
Tiện ích dùng chung cho mọi bước chạy lâu: ghi JSON nguyên tử (checkpoint), khoá chống chạy
trùng, và cấu hình log ra màn hình + file.

Mọi bước (tải SEA, OCR stbook, xử lý giáo trình) đều ghi log qua logger "vi_corpus".
"""

import fcntl
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger("vi_corpus")


def write_json_atomic(path: Path, data: Any) -> None:
    """
    Ghi `data` ra `path` dưới dạng JSON một cách nguyên tử.

    Ghi vào file tạm cùng thư mục, fsync xuống đĩa, rồi os.replace sang tên thật.
    os.replace là thao tác nguyên tử, nên người đọc chỉ thấy bản cũ hoặc bản mới hoàn chỉnh.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any | None:
    """Đọc file JSON; trả về None nếu file không tồn tại hoặc bị hỏng."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def setup_logging(log_dir: Path, key: str) -> Path:
    """
    Cấu hình log ra màn hình và ra file logs/<key>_<thời-gian>.log.

    Returns:
        Đường dẫn file log.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{key}_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

    logger.setLevel(logging.INFO)
    logger.propagate = False  # paddle gắn handler vào root logger, tránh in log 2 lần
    logger.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return log_file


def acquire_run_lock(state_dir: Path) -> TextIO:
    """
    Đảm bảo mỗi việc (tải một bộ, OCR, xử lý giáo trình) chỉ có một tiến trình chạy tại một thời điểm.

    Dùng flock trên <state_dir>/.run.lock; hệ điều hành tự nhả khoá khi tiến trình
    chết (kể cả kill -9), nên không bao giờ bị kẹt khoá sau crash.

    Returns:
        File đang giữ khoá. Phải giữ biến này sống suốt lần chạy.

    Raises:
        SystemExit: nếu đã có tiến trình khác đang chạy việc này.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_file = open(state_dir / ".run.lock", "w")  # noqa: SIM115 — giữ mở suốt lần chạy
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(
            f"Đang có một tiến trình khác chạy việc này ({state_dir.name}). "
            "Xem tiến độ bằng --status, hoặc dừng tiến trình kia trước."
        ) from None
    return lock_file
