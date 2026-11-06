"""
Kiểm tra bản chép VISTA trên server (bước V0, VS-02): so thư mục raw/vista với nhật ký tải pdf_download_log.jsonl, chạy trên CPU, không cần mạng.

Mỗi dòng nhật ký: {"publication_id", "status", "path", "sha256", "bytes"} (status: downloaded, already_on_disk, corrupt_pdf, http_404, server_error_502). Bài tải thành công
phải có file raw/vista/<năm>/<publication_id>.pdf đúng số byte (và đúng sha256 nếu bật --sha256, chậm vì đọc cả ~234 GB). Báo:
- thiếu (nhật ký ghi tải xong mà không có file) và sai kích thước (chép dở / hỏng), kèm vài mã bài ví dụ;
- thừa (file không có trong nhật ký, chưa tính unknown/ vốn không có trong nhật ký);
- số file theo thư mục năm (đối chiếu với nhật ký: năm lấy từ thư mục nơi file thật nằm; bài thiếu thì không biết năm nên đếm riêng).
Năm của bài thiếu không có trong nhật ký, nên kiểm tra theo năm chỉ so số file có mặt; muốn biết bài thiếu thuộc năm nào thì đối chiếu thêm vista_detail.jsonl (không làm ở đây).
"""

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vi_corpus.vjol.extract import YEAR_DIR

OK_STATUSES = frozenset({"downloaded", "already_on_disk"})
EXAMPLES = 10


def read_log(path: Path) -> dict[str, dict]:
    """Đọc nhật ký tải: {mã bài: dòng cuối cùng của bài đó}; dòng JSON hỏng bị bỏ."""
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("publication_id") is not None:
                rows[str(row["publication_id"])] = row
    return rows


def scan_raw(root: Path) -> tuple[dict[str, tuple[Path, int]], list[Path]]:
    """Liệt kê file PDF: ({mã bài: (đường dẫn, số byte)} của <năm>/<mã>.pdf, danh sách file trong unknown/ hoặc nơi khác không thuộc thư mục năm)."""
    by_id: dict[str, tuple[Path, int]] = {}
    other: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name == "metadata":
            continue
        for p in d.rglob("*.pdf"):
            if YEAR_DIR.match(d.name) and p.stem.isdigit():
                by_id[p.stem] = (p, p.stat().st_size)
            else:
                other.append(p)
    return by_id, other


def _sha256(path: Path) -> str:
    """sha256 nội dung một file, đọc theo khối 8 MB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def check_raw(root: Path, log_path: Path | None = None, verify_sha: bool = False, workers: int = 8) -> dict:
    """
    So thư mục raw/vista với nhật ký tải.

    Args:
        root:       Thư mục raw/vista (thư mục năm + unknown/ + metadata/).
        log_path:   File pdf_download_log.jsonl (mặc định <root>/metadata/pdf_download_log.jsonl).
        verify_sha: True thì băm lại mọi file có mặt và so với sha256 trong nhật ký (đọc cả kho, rất chậm).
        workers:    Số luồng băm.

    Returns:
        {"logged_ok", "on_disk", "unknown_dir", "missing", "size_mismatch", "sha_mismatch", "extra", "log_statuses", "per_year", "examples"}: các số đếm
        và vài mã bài ví dụ cho mỗi loại lỗi. Mọi số bằng 0 ở missing / size_mismatch / sha_mismatch nghĩa là bản chép khớp nhật ký.
    """
    log = read_log(log_path or root / "metadata" / "pdf_download_log.jsonl")
    by_id, other = scan_raw(root)
    ok = {pid: r for pid, r in log.items() if r.get("status") in OK_STATUSES}
    missing = sorted(pid for pid in ok if pid not in by_id)
    size_bad = sorted(pid for pid, r in ok.items() if pid in by_id and r.get("bytes") is not None and by_id[pid][1] != r["bytes"])
    extra = sorted(pid for pid in by_id if pid not in ok)
    sha_bad: list[str] = []
    if verify_sha:
        todo = [pid for pid, r in ok.items() if pid in by_id and r.get("sha256") and pid not in size_bad]
        with ThreadPoolExecutor(workers) as pool:
            for pid, digest in zip(todo, pool.map(lambda pid: _sha256(by_id[pid][0]), todo)):
                if digest != ok[pid]["sha256"]:
                    sha_bad.append(pid)
    return {
        "logged_ok": len(ok), "on_disk": len(by_id), "unknown_dir": len(other), "missing": len(missing), "size_mismatch": len(size_bad),
        "sha_mismatch": len(sha_bad) if verify_sha else None, "extra": len(extra), "log_statuses": dict(Counter(r.get("status") for r in log.values())),
        "per_year": dict(sorted(Counter(p.parent.name for p, _ in by_id.values()).items())),
        "examples": {"missing": missing[:EXAMPLES], "size_mismatch": size_bad[:EXAMPLES], "sha_mismatch": sha_bad[:EXAMPLES], "extra": extra[:EXAMPLES]},
    }
