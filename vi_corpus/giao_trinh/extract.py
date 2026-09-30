"""
Xử lý thô giáo trình (Google Drive "Tổng hợp giáo trình Đại học"): kiểm kê file và trích text từng trang.

Đọc <data-root>/raw/giao_trinh/<ngành>/<môn>/<file> (bản rclone, không sửa), ghi
interim/giao_trinh_text/<sha256>.json cho mỗi file duy nhất:

    {"sha256", "size", "paths": [đường dẫn, cùng nội dung], "nganh", "mon", "format",
     "pages": [{"text", "method"}], "error", "skipped", "created_at"}

Các bước (chiến lược ở docs/GIAO_TRINH.md, mục 2-3):
1. inventory: sha256 từng file; cùng sha256 ở nhiều đường dẫn thì chỉ xử lý một lần, giữ đủ đường dẫn.
2. extract: PDF qua vi_corpus.common.pdf_text.extract_pdf (trang scan thì OCR, mô hình nạp lười),
   pptx qua python-pptx (mỗi slide một trang), định dạng khác ghi skipped="unsupported_format".

Checkpoint theo file (khoá sha256), ghi nguyên tử; chạy lại đúng lệnh cũ để làm tiếp. Làm từng file một.
File lỗi (PDF hỏng, có mật khẩu) được ghi lỗi vào checkpoint và thử lại ở lần chạy sau.
"""

import hashlib
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from vi_corpus.common.pdf_text import extract_pdf
from vi_corpus.common.state import read_json, write_json_atomic

logger = logging.getLogger("vi_corpus")

TEXT_DIR = "interim/giao_trinh_text"
FORMATS = {".pdf": "pdf", ".pptx": "pptx"}


def checkpoint_path(data_root: Path, sha256: str) -> Path:
    """Đường dẫn checkpoint (file kết quả trích text) của một file giáo trình."""
    return data_root / TEXT_DIR / f"{sha256}.json"


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
        root:        Thư mục giáo trình.
        limit_files: Chỉ tính N file đầu (chạy thử).

    Returns:
        {sha256: [đường dẫn các file trùng nội dung, theo thứ tự đường dẫn]}, theo thứ tự xuất hiện đầu tiên.
    """
    groups: dict[str, list[Path]] = {}
    # ponytail: băm lại mọi file mỗi lần chạy; thêm cache theo (đường dẫn, size, mtime) nếu kho lớn hàng chục GB.
    for path in tqdm(list_files(root)[:limit_files], desc="Kiểm kê (sha256)", unit="file"):
        groups.setdefault(file_sha256(path), []).append(path)
    return groups


def extract_pptx(path: Path) -> list[dict]:
    """
    Trích text pptx: mỗi slide một trang gồm khung chữ (kể cả trong nhóm), bảng và ghi chú.

    Returns:
        [{"text", "method": "pptx"}] theo thứ tự slide.
    """
    from pptx import Presentation

    def shape_lines(shapes) -> list[str]:
        """Các dòng text của một danh sách shape (đệ quy vào nhóm shape)."""
        out = []
        for shape in shapes:
            if shape.shape_type == 6:  # MSO_SHAPE_TYPE.GROUP
                out += shape_lines(shape.shapes)
            elif shape.has_text_frame:
                out += [p.text for p in shape.text_frame.paragraphs]
            elif shape.has_table:
                out += [" | ".join(c.text for c in row.cells) for row in shape.table.rows]
        return out

    pages = []
    for slide in Presentation(path).slides:
        lines = shape_lines(slide.shapes)
        if slide.has_notes_slide:
            lines += [p.text for p in slide.notes_slide.notes_text_frame.paragraphs]
        pages.append({"text": "\n".join(ln for ln in lines if ln.strip()), "method": "pptx"})
    return pages


def extract_file(path: Path, get_ocr: Callable[[], object | None]) -> dict:
    """
    Trích text một file theo định dạng (đuôi file); không ném lỗi ra ngoài.

    Returns:
        {"format", "pages", "error", "skipped"}: định dạng không hỗ trợ thì skipped="unsupported_format";
        file hỏng / có mật khẩu thì error là thông báo lỗi.
    """
    fmt = FORMATS.get(path.suffix.lower())
    out = {"format": fmt or path.suffix.lower().lstrip(".") or "unknown", "pages": [], "error": None, "skipped": None}
    if fmt is None:
        out["skipped"] = "unsupported_format"
        return out
    try:
        out["pages"] = extract_pdf(path, get_ocr) if fmt == "pdf" else extract_pptx(path)
    except Exception as exc:  # noqa: BLE001 — một file lỗi không được dừng cả lô
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


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


def _needs_redo(done: dict | None, ocr_enabled: bool) -> bool:
    """Checkpoint cần làm lại: chưa có, lần trước lỗi, hoặc còn trang needs_ocr mà lần này có OCR."""
    if done is None or done.get("error"):
        return True
    return ocr_enabled and any(p["method"] == "needs_ocr" for p in done["pages"])


def run_extract(root: Path, data_root: Path, get_ocr: Callable[[], object | None],
                ocr_enabled: bool = True, limit_files: int | None = None) -> int:
    """
    Kiểm kê rồi trích text mọi file chưa làm, ghi checkpoint từng file ngay khi xong.

    Đường dẫn trong checkpoint tương đối so với data_root (tuyệt đối nếu `root` nằm ngoài data_root);
    ngành = thư mục cấp 1, môn = thư mục cấp 2 dưới `root`. Checkpoint đã có mà danh sách đường dẫn
    trùng đổi thì cập nhật lại danh sách.

    Args:
        root:        Thư mục giáo trình.
        data_root:   Thư mục gốc dữ liệu.
        get_ocr:     Hàm lấy bộ OCR (xem make_ocr_getter).
        ocr_enabled: Có OCR hay không (dùng để quyết định làm lại file còn trang needs_ocr).
        limit_files: Chỉ xét N file đầu (chạy thử).

    Returns:
        Số file bị lỗi trong lần chạy này.
    """
    groups = inventory(root, limit_files)
    logger.info("Kiểm kê: %d file, %d nội dung duy nhất", sum(map(len, groups.values())), len(groups))
    failed = done_count = 0
    for sha, paths in tqdm(groups.items(), desc="Trích text giáo trình", unit="file"):
        rels = [str(p.relative_to(data_root)) if p.is_relative_to(data_root) else str(p) for p in paths]
        ckpt = checkpoint_path(data_root, sha)
        done = read_json(ckpt)
        if not _needs_redo(done, ocr_enabled):
            done_count += 1
            if done["paths"] != rels:
                write_json_atomic(ckpt, {**done, "paths": rels})
            continue
        parts = paths[0].relative_to(root).parts
        result = extract_file(paths[0], get_ocr)
        write_json_atomic(ckpt, {
            "sha256": sha, "size": paths[0].stat().st_size, "paths": rels,
            "nganh": parts[0] if len(parts) > 1 else "", "mon": parts[1] if len(parts) > 2 else "",
            **result, "created_at": datetime.now(timezone.utc).isoformat(),
        })
        if result["error"]:
            failed += 1
            logger.warning("Lỗi %s: %s", rels[0], result["error"])
    logger.info("Xong: %d file làm mới hoặc làm lại, %d đã có checkpoint, %d lỗi",
                len(groups) - done_count, done_count, failed)
    return failed


def status(root: Path, data_root: Path) -> dict:
    """
    Tóm tắt tiến độ (không băm file): số file trong raw, số checkpoint, và thống kê trang theo method.

    Returns:
        {"files", "checkpoints", "errors", "skipped", "pages": {method: số trang}}.
    """
    info = {"files": len(list_files(root)), "checkpoints": 0, "errors": 0, "skipped": 0, "pages": {}}
    for ckpt in sorted((data_root / TEXT_DIR).glob("*.json")):
        done = read_json(ckpt)
        if done is None:
            continue
        info["checkpoints"] += 1
        info["errors"] += bool(done.get("error"))
        info["skipped"] += bool(done.get("skipped"))
        for page in done["pages"]:
            info["pages"][page["method"]] = info["pages"].get(page["method"], 0) + 1
    return info
