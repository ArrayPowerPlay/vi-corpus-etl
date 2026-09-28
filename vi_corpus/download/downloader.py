"""
Vòng tải chính cho một bộ dữ liệu, và lệnh xem tiến độ.

Quy trình run_dataset():
    1. Đọc token, lấy mã commit mới nhất của repo, liệt kê file tiếng Việt.
    2. Lưu manifest (danh sách file) để --status dùng được khi không có mạng.
    3. Bỏ qua file đã xong (state "done" + file trên đĩa còn đủ kích thước).
    4. Tải song song các file còn lại; mỗi file xong thì ghi state ngay.
       Đơn vị checkpoint là từng file: file đang tải dở lúc crash sẽ được tải lại từ đầu
       (huggingface_hub không hỗ trợ tải tiếp), phần dở được dọn ở lần chạy sau.
    5. In tổng kết. File lỗi sẽ được thử lại khi chạy lại đúng lệnh cũ.

Cấu trúc thư mục dưới data_root:
    raw/<ds>/<đường dẫn gốc trên HF>   bản gốc
    state/<ds>/*.json                  checkpoint
    logs/<ds>_<thời-gian>.log          log
"""

import fcntl
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm

from .checkpoint import Checkpoint
from .datasets import DatasetSpec, get_spec
from .hub import RemoteFile, download_file, list_remote_files, load_token, resolve_revision

logger = logging.getLogger("vi_corpus")

GB = 1000**3  # dùng GB thập phân, khớp với số liệu trên HF


@dataclass
class DatasetPaths:
    """Các thư mục dùng cho một bộ dữ liệu dưới data_root."""

    raw: Path
    state: Path
    logs: Path

    @classmethod
    def for_dataset(cls, data_root: Path, key: str) -> "DatasetPaths":
        """Tính các thư mục raw/state/logs cho bộ dữ liệu `key`."""
        root = Path(data_root)
        return cls(raw=root / "raw" / key, state=root / "state" / key, logs=root / "logs")


@dataclass
class RunResult:
    """Kết quả một lần chạy: số file đã có sẵn, tải mới, và lỗi."""

    skipped: int = 0
    downloaded: int = 0
    failed: int = 0


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
    logger.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return log_file


def acquire_run_lock(state_dir: Path) -> TextIO:
    """
    Đảm bảo mỗi bộ dữ liệu chỉ có một tiến trình tải tại một thời điểm.

    Dùng flock trên state/<ds>/.run.lock; hệ điều hành tự nhả khoá khi tiến trình
    chết (kể cả kill -9), nên không bao giờ bị kẹt khoá sau crash.

    Returns:
        File đang giữ khoá. Phải giữ biến này sống suốt lần chạy.

    Raises:
        SystemExit: nếu đã có tiến trình khác đang tải bộ này.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_file = open(state_dir / ".run.lock", "w")  # noqa: SIM115 — giữ mở suốt lần chạy
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(
            f"Đang có một tiến trình khác tải bộ này ({state_dir.name}). "
            "Xem tiến độ bằng --status, hoặc dừng tiến trình kia trước."
        ) from None
    return lock_file


def cleanup_incomplete(raw_dir: Path) -> None:
    """
    Xoá các file tải dở (*.incomplete) còn sót lại sau crash.

    huggingface_hub không tải tiếp các file này (mỗi lần tải dùng một file tạm tên ngẫu nhiên),
    nên giữ lại chỉ tốn ổ đĩa. Chỉ gọi khi đang giữ khoá, để không xoá nhầm file đang tải.
    """
    download_cache = raw_dir / ".cache" / "huggingface" / "download"
    if not download_cache.exists():
        return
    leftovers = list(download_cache.rglob("*.incomplete"))
    freed = sum(p.stat().st_size for p in leftovers)
    for p in leftovers:
        p.unlink(missing_ok=True)
    if leftovers:
        logger.info("Đã dọn %d file tải dở từ lần trước (%.2f GB)", len(leftovers), freed / GB)


def _is_complete(ckpt: Checkpoint, raw_dir: Path, remote: RemoteFile) -> bool:
    """File coi là xong khi state ghi "done" và file trên đĩa vẫn còn, đúng kích thước."""
    if not ckpt.is_done(remote.path, remote.size, remote.sha256):
        return False
    local = raw_dir / remote.path
    return local.is_file() and local.stat().st_size == remote.size


def run_dataset(
    key: str,
    data_root: Path,
    workers: int = 8,
    limit_files: int | None = None,
    verify_sha256: bool = False,
    max_retries: int = 5,
) -> RunResult:
    """
    Tải toàn bộ phần tiếng Việt của một bộ dữ liệu, có checkpoint để chạy tiếp.

    Args:
        key:           Mã bộ dữ liệu (xem vi_corpus.download.datasets.DATASETS).
        data_root:     Thư mục gốc lưu dữ liệu.
        workers:       Số file tải cùng lúc.
        limit_files:   Chỉ xử lý N file đầu tiên (để chạy thử). None = tất cả.
        verify_sha256: Tính sha256 sau khi tải để chắc chắn file không hỏng (chậm hơn).
        max_retries:   Số lần thử lại mỗi file khi lỗi.

    Returns:
        RunResult với số file bỏ qua / tải mới / lỗi.
    """
    spec: DatasetSpec = get_spec(key)
    paths = DatasetPaths.for_dataset(data_root, key)
    lock = acquire_run_lock(paths.state)  # noqa: F841 — giữ khoá tới khi hàm kết thúc
    log_file = setup_logging(paths.logs, key)
    disable_progress_bars()  # tắt thanh tiến trình từng file của HF; dùng thanh tổng bên dưới

    logger.info("=== %s ===", spec.description)
    logger.info("Repo: %s | thư mục: %s | log: %s", spec.repo_id, spec.vi_dir, log_file)
    cleanup_incomplete(paths.raw)

    token = load_token()
    api = HfApi(token=token)
    revision = resolve_revision(api, spec.repo_id, max_retries)
    files = list_remote_files(api, spec, revision, max_retries)
    if limit_files is not None:
        files = files[:limit_files]

    ckpt = Checkpoint(paths.state)
    ckpt.write_manifest(
        {
            "repo_id": spec.repo_id,
            "revision": revision,
            "vi_dir": spec.vi_dir,
            "limit_files": limit_files,
            "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in files],
        }
    )

    pending = [f for f in files if not _is_complete(ckpt, paths.raw, f)]
    result = RunResult(skipped=len(files) - len(pending))
    total_gb = sum(f.size for f in files) / GB
    pending_gb = sum(f.size for f in pending) / GB
    logger.info(
        "Revision %s | %d file (%.1f GB) | đã xong %d | cần tải %d (%.1f GB)",
        revision[:10], len(files), total_gb, result.skipped, len(pending), pending_gb,
    )
    if not pending:
        logger.info("Không còn gì để tải.")
        return result

    def _work(remote: RemoteFile) -> RemoteFile:
        """Tải một file và ghi state "done" ngay khi xong."""
        started = time.time()
        download_file(spec, remote, revision, paths.raw, token, verify_sha256, max_retries)
        ckpt.mark_done(
            remote.path,
            {
                "size": remote.size,
                "sha256": remote.sha256,
                "sha256_verified": bool(verify_sha256 and remote.sha256),
                "revision": revision,
                "seconds": round(time.time() - started, 1),
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        return remote

    # mininterval lớn để log nohup không bị tràn bởi thanh tiến trình
    bar = tqdm(total=sum(f.size for f in pending), unit="B", unit_scale=True,
               desc=key, mininterval=30, file=sys.stdout)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(_work, f): f for f in pending}
        for fut in as_completed(futures):
            remote = futures[fut]
            try:
                fut.result()
                result.downloaded += 1
                logger.info("Xong %s (%.2f GB)", remote.path, remote.size / GB)
            except Exception as exc:  # noqa: BLE001 — một file lỗi không được làm dừng cả bộ
                result.failed += 1
                ckpt.mark_failed(remote.path, f"{type(exc).__name__}: {exc}")
                logger.error("LỖI %s: %s: %s", remote.path, type(exc).__name__, exc)
            bar.update(remote.size)
    except KeyboardInterrupt:
        # Ctrl+C: thoát ngay thay vì chờ các file đang tải dở. An toàn vì state được ghi
        # nguyên tử và phần đã tải dở sẽ được tải tiếp ở lần chạy sau.
        logger.warning("Đã dừng theo yêu cầu. Chạy lại đúng lệnh cũ để tải tiếp.")
        logging.shutdown()
        os._exit(130)
    finally:
        bar.close()
    pool.shutdown(wait=True)

    logger.info(
        "Tổng kết %s: có sẵn %d | tải mới %d | lỗi %d",
        key, result.skipped, result.downloaded, result.failed,
    )
    if result.failed:
        logger.warning("Có file lỗi. Chạy lại đúng lệnh cũ để thử lại các file đó.")
    return result


def print_status(key: str, data_root: Path) -> None:
    """
    In tiến độ của một bộ dữ liệu, chỉ đọc file trên đĩa (không cần mạng).

    Chạy được cả khi tiến trình tải đang chạy song song.
    """
    spec = get_spec(key)
    paths = DatasetPaths.for_dataset(data_root, key)
    manifest = Checkpoint(paths.state).read_manifest() if paths.state.exists() else None
    if not manifest:
        print(f"[{key}] Chưa chạy lần nào (không có {paths.state}/_manifest.json).")
        return

    ckpt = Checkpoint(paths.state)
    files = [RemoteFile(f["path"], f["size"], f["sha256"]) for f in manifest["files"]]
    done = [f for f in files if _is_complete(ckpt, paths.raw, f)]
    failed = [s for s in ckpt.all_states() if s.get("status") == "failed"]
    total_gb = sum(f.size for f in files) / GB
    done_gb = sum(f.size for f in done) / GB
    pct = 100 * done_gb / total_gb if total_gb else 100.0

    print(f"[{key}] {spec.description}")
    print(f"  Revision : {manifest['revision'][:10]}")
    print(f"  Xong     : {len(done)}/{len(files)} file | {done_gb:.1f}/{total_gb:.1f} GB ({pct:.1f}%)")
    print(f"  Lỗi      : {len(failed)} file")
    for s in failed[:5]:
        print(f"     - {s['path']}: {s.get('error', '')[:150]}")
    if len(failed) > 5:
        print(f"     ... và {len(failed) - 5} file khác")
