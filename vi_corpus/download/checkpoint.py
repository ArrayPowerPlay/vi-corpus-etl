"""
Checkpoint: ghi nhớ file nào đã tải xong để lần chạy sau bỏ qua.

Mỗi file trên Hugging Face có một file state JSON riêng trong state/<ds>/.
File state luôn được ghi theo kiểu "nguyên tử" (ghi ra file .tmp rồi đổi tên),
nên dù máy crash giữa lúc ghi cũng không bao giờ để lại file state hỏng.

Ngoài ra còn một file _manifest.json lưu danh sách toàn bộ file cần tải,
giúp lệnh --status báo tiến độ mà không cần mạng.
"""

import json
import os
from pathlib import Path
from typing import Any

MANIFEST_NAME = "_manifest.json"


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


class Checkpoint:
    """
    Quản lý trạng thái tải của một bộ dữ liệu (thư mục state/<ds>/).

    Mỗi file trên HF ứng với một file state có trường "status":
        "done"   : đã tải xong và đã kiểm tra.
        "failed" : lần tải gần nhất bị lỗi (sẽ được thử lại ở lần chạy sau).
    """

    def __init__(self, state_dir: Path):
        """Khởi tạo với thư mục chứa state; tạo thư mục nếu chưa có."""
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def state_path(self, repo_path: str) -> Path:
        """Đường dẫn file state của một file trong repo (đổi "/" thành "__" để phẳng thư mục)."""
        return self.state_dir / (repo_path.replace("/", "__") + ".json")

    def get(self, repo_path: str) -> dict | None:
        """Đọc state của một file; None nếu chưa từng xử lý."""
        return read_json(self.state_path(repo_path))

    def is_done(self, repo_path: str, expected_size: int, expected_sha256: str | None) -> bool:
        """
        True nếu file đã được đánh dấu xong VÀ vẫn khớp với phiên bản trên HF.

        Nếu tác giả cập nhật file trên HF (kích thước hoặc sha256 đổi) thì trả về False
        để file được tải lại.
        """
        state = self.get(repo_path)
        if not state or state.get("status") != "done":
            return False
        if state.get("size") != expected_size:
            return False
        if expected_sha256 and state.get("sha256") and state["sha256"] != expected_sha256:
            return False
        return True

    def mark_done(self, repo_path: str, record: dict) -> None:
        """Đánh dấu một file đã tải xong, kèm thông tin (kích thước, sha256, thời gian...)."""
        write_json_atomic(self.state_path(repo_path), {"path": repo_path, "status": "done", **record})

    def mark_failed(self, repo_path: str, error: str) -> None:
        """Ghi lại lỗi của một file để --status hiển thị và lần chạy sau thử lại."""
        write_json_atomic(
            self.state_path(repo_path), {"path": repo_path, "status": "failed", "error": error}
        )

    def all_states(self) -> list[dict]:
        """Đọc toàn bộ file state trong thư mục (bỏ qua manifest và file tạm)."""
        states = []
        for p in sorted(self.state_dir.glob("*.json")):
            if p.name == MANIFEST_NAME:
                continue
            data = read_json(p)
            if data:
                states.append(data)
        return states

    def write_manifest(self, manifest: dict) -> None:
        """Lưu danh sách file cần tải (dùng cho --status khi không có mạng)."""
        write_json_atomic(self.state_dir / MANIFEST_NAME, manifest)

    def read_manifest(self) -> dict | None:
        """Đọc manifest đã lưu; None nếu chưa chạy lần nào."""
        return read_json(self.state_dir / MANIFEST_NAME)
