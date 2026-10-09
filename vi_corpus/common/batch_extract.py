"""
Khung xử lý thô theo lô file dùng chung cho các nguồn PDF (giáo trình, VJOL, ...).

Mỗi nguồn chỉ cung cấp hai thứ: hàm trích text một file và hàm lấy metadata từ đường dẫn. Phần còn lại
làm ở đây:
1. inventory: sha256 từng file; cùng sha256 ở nhiều đường dẫn thì chỉ xử lý một lần, giữ đủ đường dẫn.
2. run_batch: trích text từng file chưa làm, ghi checkpoint <data-root>/<text_dir>/<sha256>.json (nguyên tử,
   khoá sha256); chạy lại đúng lệnh cũ để làm tiếp. File lỗi được ghi lỗi và thử lại ở lần chạy sau.
3. make_ocr_getter: nạp lười bộ OCR (chỉ khi gặp trang scan).
4. status: tóm tắt tiến độ từ checkpoint.

Checkpoint có dạng:
    {"sha256", "size", "paths": [...], **metadata từ đường dẫn, "format", "pages": [{"text", "method"}],
     "error", "skipped", "parser_version", "created_at"}
"""

import hashlib
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from vi_corpus.common.state import read_json, write_json_atomic

logger = logging.getLogger("vi_corpus")


def checkpoint_path(data_root: Path, text_dir: str, sha256: str) -> Path:
    """Đường dẫn checkpoint (file kết quả trích text) của một file, trong thư mục `text_dir` của data_root."""
    return data_root / text_dir / f"{sha256}.json"


def file_sha256(path: Path) -> str:
    """sha256 của file, đọc từng khối 1 MiB để không nạp cả file vào RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def list_files(root: Path) -> list[Path]:
    """Mọi file trong `root`, theo thứ tự đường dẫn; bỏ file ẩn và thư mục ẩn (vd .cache của rclone)."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


def inventory(root: Path, limit_files: int | None = None) -> dict[str, list[Path]]:
    """
    Kiểm kê: gom file theo sha256 nội dung.

    Args:
        root:        Thư mục nguồn.
        limit_files: Chỉ tính N file đầu (chạy thử).

    Returns:
        {sha256: [đường dẫn các file trùng nội dung, theo thứ tự đường dẫn]}, theo thứ tự xuất hiện đầu tiên.
    """
    groups: dict[str, list[Path]] = {}
    # ponytail: băm lại mọi file mỗi lần chạy; thêm cache theo (đường dẫn, size, mtime) nếu kho lớn hàng chục GB.
    for path in tqdm(list_files(root)[:limit_files], desc="Kiểm kê (sha256)", unit="file"):
        groups.setdefault(file_sha256(path), []).append(path)
    return groups


def make_ocr_getter(device: str, enabled: bool = True) -> Callable[[], object | None]:
    """
    Tạo hàm lấy bộ OCR, nạp lười ở lần gọi đầu (chỉ khi gặp trang scan).

    Trả về None nếu `enabled=False` hoặc nhóm thư viện ocr chưa cài (trang thành "needs_ocr").
    Lỗi nạp khác (vd CUDA) được nhớ lại và ném lại mỗi lần gọi (file scan đó thành lỗi), không nạp lại.
    """
    cache: list = []

    def get_ocr() -> object | None:
        """Bộ OCR dùng chung cho cả lần chạy, hoặc None nếu không OCR được."""
        if not enabled:
            return None
        if not cache:
            try:
                from vi_corpus.common.ocr import PageOcr

                cache.append(PageOcr(device))
            except ImportError as exc:
                logger.warning("Không nạp được OCR (%s): trang scan để needs_ocr. Cài: uv sync --group ocr", exc)
                cache.append(None)
            except Exception as exc:  # noqa: BLE001 — nhớ lỗi để không nạp lại mô hình mỗi trang
                cache.append(exc)
        if isinstance(cache[0], Exception):
            raise cache[0]
        return cache[0]

    return get_ocr


def _needs_redo(done: dict | None, ocr_enabled: bool, retry_skipped: bool = False) -> bool:
    """
    Checkpoint cần làm lại: chưa có, lần trước lỗi, lần trước bị bỏ qua mà retry_skipped, hoặc còn trang needs_ocr mà
    lần này có OCR.
    """
    if done is None or done.get("error") or (retry_skipped and done.get("skipped")):
        return True
    return ocr_enabled and any(p["method"] == "needs_ocr" for p in done["pages"])


def run_batch(root: Path, data_root: Path, text_dir: str,
              extract_file: Callable[[Path, Callable[[], object | None]], dict],
              path_meta: Callable[[Path], dict],
              get_ocr: Callable[[], object | None], ocr_enabled: bool = True,
              limit_files: int | None = None, desc: str = "Trích text", retry_skipped: bool = False) -> int:
    """
    Kiểm kê rồi trích text mọi file chưa làm, ghi checkpoint từng file ngay khi xong.

    Đường dẫn trong checkpoint tương đối so với data_root (tuyệt đối nếu `root` nằm ngoài data_root).
    Checkpoint đã có mà danh sách đường dẫn trùng đổi thì cập nhật lại danh sách.

    Args:
        root:         Thư mục nguồn.
        data_root:    Thư mục gốc dữ liệu.
        text_dir:     Thư mục checkpoint, tương đối so với data_root (vd "interim/vjol_text").
        extract_file: Hàm (đường dẫn, get_ocr) -> {"format", "pages", "error", "skipped", ...}; không ném lỗi ra ngoài.
        path_meta:    Hàm đường dẫn file -> dict metadata suy ra từ đường dẫn (vd ngành, môn, năm), trộn vào checkpoint.
        get_ocr:      Hàm lấy bộ OCR (xem make_ocr_getter).
        ocr_enabled:  Có OCR hay không (dùng để quyết định làm lại file còn trang needs_ocr).
        limit_files:  Chỉ xét N file đầu (chạy thử).
        desc:         Nhãn thanh tiến độ.
        retry_skipped: Làm lại cả file từng bị bỏ qua (định dạng mới được hỗ trợ, vừa cài công cụ ngoài).

    Returns:
        Số file bị lỗi trong lần chạy này.
    """
    groups = inventory(root, limit_files)
    logger.info("Kiểm kê: %d file, %d nội dung duy nhất", sum(map(len, groups.values())), len(groups))
    failed = done_count = 0
    for sha, paths in tqdm(groups.items(), desc=desc, unit="file"):
        rels = [str(p.relative_to(data_root)) if p.is_relative_to(data_root) else str(p) for p in paths]
        ckpt = checkpoint_path(data_root, text_dir, sha)
        done = read_json(ckpt)
        if not _needs_redo(done, ocr_enabled, retry_skipped):
            done_count += 1
            if done["paths"] != rels:
                write_json_atomic(ckpt, {**done, "paths": rels})
            continue
        result = extract_file(paths[0], get_ocr)
        write_json_atomic(ckpt, {
            "sha256": sha, "size": paths[0].stat().st_size, "paths": rels,
            **path_meta(paths[0]), **result, "created_at": datetime.now(timezone.utc).isoformat(),
        })
        if result["error"]:
            failed += 1
            logger.warning("Lỗi %s: %s", rels[0], result["error"])
    logger.info("Xong: %d file làm mới hoặc làm lại, %d đã có checkpoint, %d lỗi",
                len(groups) - done_count, done_count, failed)
    return failed


def status(root: Path, data_root: Path, text_dir: str) -> dict:
    """
    Tóm tắt tiến độ (không băm file): số file trong raw, số checkpoint, và thống kê trang theo method.

    Returns:
        {"files", "checkpoints", "errors", "skipped", "pages": {method: số trang}}.
    """
    info = {"files": len(list_files(root)), "checkpoints": 0, "errors": 0, "skipped": 0, "pages": {}}
    for ckpt in sorted((data_root / text_dir).glob("*.json")):
        done = read_json(ckpt)
        if done is None:
            continue
        info["checkpoints"] += 1
        info["errors"] += bool(done.get("error"))
        info["skipped"] += bool(done.get("skipped"))
        for page in done["pages"]:
            info["pages"][page["method"]] = info["pages"].get(page["method"], 0) + 1
    return info
