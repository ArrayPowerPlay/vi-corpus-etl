"""
Làm việc với Hugging Face Hub: đọc token, liệt kê file, tải file có retry, kiểm tra sha256.

Token được đọc từ file .env ở gốc repo (HF_TOKEN=...). Nếu biến môi trường HF_TOKEN
đã được đặt sẵn thì biến môi trường được ưu tiên.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import RepoFile

from .datasets import DatasetSpec

logger = logging.getLogger("vi_corpus")

REPO_ROOT = Path(__file__).resolve().parent.parent
T = TypeVar("T")


class CorruptDownloadError(Exception):
    """File tải về không khớp kích thước hoặc sha256 với file trên HF."""


@dataclass(frozen=True)
class RemoteFile:
    """
    Một file cần tải trên HF.

    Attributes:
        path:   Đường dẫn trong repo (vd "vi/train-00000-of-00253.parquet").
        size:   Kích thước (byte).
        sha256: Mã sha256 của nội dung (chỉ có với file LFS; None nếu không có).
    """

    path: str
    size: int
    sha256: str | None


def load_token() -> str:
    """
    Lấy HF token từ biến môi trường hoặc file .env ở gốc repo.

    Raises:
        SystemExit: nếu không tìm thấy token (SEA-Instruct-2602 bị khoá nên bắt buộc phải có).
    """
    load_dotenv(REPO_ROOT / ".env", override=False)
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            f"Không tìm thấy HF_TOKEN. Hãy tạo file {REPO_ROOT / '.env'} với nội dung:\n"
            "HF_TOKEN=hf_xxx   (xem mẫu trong .env.example)"
        )
    return token


def _is_retryable(exc: Exception) -> bool:
    """
    Quyết định lỗi nào đáng thử lại.

    Không thử lại khi lỗi do cấu hình (sai repo, không có quyền, file không tồn tại),
    vì thử lại cũng vô ích. Mọi lỗi khác (mạng chập chờn, HF quá tải 429/5xx,
    file tải về bị hỏng...) đều được thử lại.
    """
    if isinstance(exc, (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError)):
        return False
    if isinstance(exc, HfHubHTTPError):
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403, 404):
            return False
    return True


def with_retry(fn: Callable[[], T], what: str, max_retries: int) -> T:
    """
    Gọi `fn()`, nếu lỗi thì thử lại tối đa `max_retries` lần với thời gian chờ tăng dần
    (10s, 20s, 40s, ... tối đa 5 phút).

    Args:
        fn:          Hàm không tham số cần gọi.
        what:        Mô tả việc đang làm, để ghi log.
        max_retries: Số lần thử lại tối đa (không tính lần đầu).
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — phân loại lỗi ngay bên dưới
            if not _is_retryable(exc) or attempt == max_retries:
                raise
            wait = min(10 * 2**attempt, 300)
            logger.warning(
                "%s lỗi (lần %d/%d): %s: %s — thử lại sau %ds",
                what, attempt + 1, max_retries + 1, type(exc).__name__, exc, wait,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def resolve_revision(api: HfApi, repo_id: str, max_retries: int) -> str:
    """
    Lấy mã commit mới nhất của repo.

    Cả lần chạy dùng cố định mã này, để mọi file đều cùng một phiên bản
    kể cả khi tác giả cập nhật repo giữa chừng.
    """
    info = with_retry(
        lambda: api.dataset_info(repo_id), f"Lấy thông tin {repo_id}", max_retries
    )
    return info.sha


def list_remote_files(
    api: HfApi, spec: DatasetSpec, revision: str, max_retries: int
) -> list[RemoteFile]:
    """Liệt kê mọi file trong thư mục tiếng Việt của bộ dữ liệu, sắp xếp theo tên."""

    def _list() -> list[RemoteFile]:
        entries = api.list_repo_tree(
            spec.repo_id,
            path_in_repo=spec.vi_dir,
            recursive=True,
            revision=revision,
            repo_type="dataset",
        )
        return [
            RemoteFile(path=e.path, size=e.size, sha256=e.lfs.sha256 if e.lfs else None)
            for e in entries
            if isinstance(e, RepoFile)
        ]

    files = with_retry(_list, f"Liệt kê file {spec.repo_id}/{spec.vi_dir}", max_retries)
    return sorted(files, key=lambda f: f.path)


def sha256_of(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Tính sha256 của một file trên đĩa (đọc từng khúc 8 MB, không tốn RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    spec: DatasetSpec,
    remote: RemoteFile,
    revision: str,
    local_dir: Path,
    token: str,
    verify_sha256: bool,
    max_retries: int,
) -> Path:
    """
    Tải một file về `local_dir`, giữ nguyên đường dẫn như trên HF, rồi kiểm tra.

    - hf_hub_download tự tải tiếp phần dở dang nếu lần trước bị ngắt, và bỏ qua
      nếu file trên đĩa đã đầy đủ.
    - Sau khi tải: kiểm tra kích thước (luôn luôn) và sha256 (nếu verify_sha256).
      Không khớp thì xoá file hỏng và tải lại (tính vào số lần retry).

    Returns:
        Đường dẫn file trên đĩa.
    """

    def _download_and_check() -> Path:
        local_path = Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=remote.path,
                repo_type="dataset",
                revision=revision,
                local_dir=local_dir,
                token=token,
            )
        )
        actual_size = local_path.stat().st_size
        if actual_size != remote.size:
            local_path.unlink(missing_ok=True)
            raise CorruptDownloadError(
                f"{remote.path}: kích thước {actual_size} khác trên HF {remote.size}"
            )
        if verify_sha256 and remote.sha256:
            actual_sha = sha256_of(local_path)
            if actual_sha != remote.sha256:
                local_path.unlink(missing_ok=True)
                raise CorruptDownloadError(f"{remote.path}: sha256 không khớp")
        return local_path

    return with_retry(_download_and_check, f"Tải {remote.path}", max_retries)
