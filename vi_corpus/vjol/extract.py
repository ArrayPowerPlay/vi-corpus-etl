"""
Xử lý thô VJOL (bài báo khoa học vjol.info.vn, PDF): chọn mẫu theo năm và trích text từng trang.

Đọc <data-root>/raw/vjol/<năm>/<article_id>_<galley_id>.pdf (+ metadata/*.jsonl, không đọc ở bước này), ghi
interim/vjol_text/<sha256>.json cho mỗi PDF duy nhất (định dạng checkpoint và vòng lặp chạy lại nằm ở
vi_corpus.common.batch_extract, dùng chung với giáo trình):

    {"sha256", "size", "paths", "year_dir", "article_id", "galley_id", "format", "pages": [{"text", "method", ...}],
     "error", "skipped", "parser_version", "extract_version", "seconds", "created_at"}

Chiến lược (R-26, J-03, J-04): mỗi trang đọc tối đa 2 lần. Lần 1 là PyMuPDF cho mọi trang (text / tcvn3, vi_corpus.common.pdf_text);
lần 2 chỉ cho trang scan và trang lỗi font (J-10: lớp chữ toàn chữ có dấu nhưng sai, vi_corpus.common.garbled), bằng chuỗi engine
VLM (vi_corpus.common.ocr_cascade), method "ocr_vlm" kèm tier và status; trang lỗi font còn có reason và garbled_text.
- list_pdfs_by_year / sample_pdfs: kiểm kê PDF theo thư mục năm và chọn mẫu N bài chia đều theo năm. Mẫu là TIỀN TỐ của một
  thứ tự cố định theo seed, nên tăng N rồi chạy lại thì mẫu cũ nằm trong mẫu mới và chỉ phần mới được làm.
- extract_file: trích một PDF, không ném lỗi của file (PDF hỏng -> error); ServerDown / EngineUnavailable (lỗi hệ thống) thì
  ném lên để dừng cả lượt, không loại trang âm thầm (O-24).
- run_extract: chọn file rồi chạy run_batch với khoá theo file, nhiều tiến trình (mỗi GPU một tiến trình) chia nhau PDF.
- report: thống kê trang theo method / engine / trạng thái và thời gian, để báo lại sau mỗi lần chạy.
"""

import hashlib
import logging
import random
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from vi_corpus.common.batch_extract import checkpoint_path as _checkpoint_path
from vi_corpus.common.batch_extract import run_batch
from vi_corpus.common.parsers import PARSER_VERSION, detect_format
from vi_corpus.common.pdf_text import extract_pdf
from vi_corpus.common.state import read_json
from vi_corpus.ocr_bakeoff.cascade import EngineUnavailable, ServerDown

logger = logging.getLogger("vi_corpus")

TEXT_DIR = "interim/vjol_text"
# Phiên bản cách trích text VJOL, ghi vào checkpoint; đổi số này thì lần chạy sau làm lại mọi checkpoint cũ hơn. 1 = bản đầu (chưa có
# trường extract_version); 2 = J-10: trang lỗi font cũng đi OCR. Không dùng PARSER_VERSION vì nó dùng chung với giáo trình.
EXTRACT_VERSION = "2"
CLAIM_DIR = "state/vjol_extract/claims"
YEAR_DIR = re.compile(r"^\d{4}$")
PDF_STEM = re.compile(r"^(\d+)_(\d+)$")  # <article_id>_<galley_id>
SKIP_DIRS = frozenset({"metadata"})  # thư mục không chứa PDF bài báo


def checkpoint_path(data_root: Path, sha256: str) -> Path:
    """Đường dẫn checkpoint (file kết quả trích text) của một PDF VJOL."""
    return _checkpoint_path(data_root, TEXT_DIR, sha256)


def list_pdfs_by_year(root: Path) -> dict[str, list[Path]]:
    """
    Kiểm kê PDF trong `root` theo thư mục cấp 1 (thư mục năm "1970" ... "2026"); bỏ thư mục ẩn và metadata/.

    Returns:
        {tên thư mục: [đường dẫn PDF theo thứ tự tên]}; PDF nằm thẳng trong `root` có khoá "". Không thư mục nào rỗng.
    """
    by_year: dict[str, list[Path]] = {}
    direct = sorted(p for p in root.glob("*") if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith("."))
    if direct:
        by_year[""] = direct
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name in SKIP_DIRS:
            continue
        pdfs = sorted(p for p in d.rglob("*") if p.suffix.lower() == ".pdf" and not any(part.startswith(".") for part in p.relative_to(root).parts))
        if pdfs:
            by_year[d.name] = pdfs
    return by_year


def sample_order(by_year: dict[str, list[Path]], seed: int) -> list[Path]:
    """
    Thứ tự cố định theo seed để lấy mẫu chia đều theo năm: vòng d lấy bài thứ d của mỗi năm còn bài, thứ tự năm trong vòng bị xáo
    theo seed. Trong mỗi năm, bài được xếp theo băm (seed, tên file) nên không phụ thuộc thứ tự liệt kê của hệ điều hành.

    Returns:
        Mọi PDF, sao cho `order[:n]` là mẫu n bài chia đều theo năm (năm ít bài hơn phần chia thì nhường chỗ cho năm khác).
    """
    queues = {y: sorted(paths, key=lambda p: hashlib.sha256(f"{seed}:{y}/{p.name}".encode()).hexdigest())
              for y, paths in by_year.items()}
    order: list[Path] = []
    depth = 0
    while True:
        years = [y for y in sorted(queues) if depth < len(queues[y])]
        if not years:
            return order
        random.Random(f"{seed}:{depth}").shuffle(years)
        order.extend(queues[y][depth] for y in years)
        depth += 1


def sample_pdfs(by_year: dict[str, list[Path]], n: int, seed: int) -> list[Path]:
    """
    Mẫu n PDF chia đều theo năm (J-03). Mẫu của n nhỏ luôn nằm trong mẫu của n lớn hơn (cùng seed, cùng kho PDF), nên tăng n rồi
    chạy lại chỉ làm phần mới.
    """
    return sample_order(by_year, seed)[:n]


def select_files(root: Path, sample_docs: int | None = None, seed: int = 0, limit_files: int | None = None) -> list[Path]:
    """
    PDF cần xử lý: mẫu `sample_docs` bài chia đều theo năm, hoặc mọi PDF (theo năm rồi tên) nếu sample_docs là None.

    Args:
        limit_files: Chỉ lấy N file đầu của danh sách đã chọn (chạy thử).

    Returns:
        Danh sách đường dẫn, theo thứ tự năm rồi tên (ổn định giữa các tiến trình).
    """
    by_year = list_pdfs_by_year(root)
    if sample_docs is None:
        files = [p for y in sorted(by_year) for p in by_year[y]]
    else:
        files = sorted(sample_pdfs(by_year, sample_docs, seed), key=lambda p: (p.parent.name, p.name))
    return files[:limit_files]


def path_meta(path: Path) -> dict:
    """Metadata suy ra từ đường dẫn: năm (thư mục), article_id và galley_id (từ tên file <article_id>_<galley_id>.pdf)."""
    m = PDF_STEM.match(path.stem)
    year = path.parent.name
    return {"year_dir": int(year) if YEAR_DIR.match(year) else None,
            "article_id": m.group(1) if m else None, "galley_id": m.group(2) if m else None}


def extract_file(path: Path, get_ocr: Callable[[], object | None]) -> dict:
    """
    Trích text một PDF VJOL; không ném lỗi của file.

    Returns:
        {"format", "pages", "error", "skipped", "parser_version", "extract_version", "seconds"}: không phải PDF thì skipped="unsupported_format";
        PDF hỏng / có mật khẩu thì error là thông báo lỗi.

    Raises:
        ServerDown, EngineUnavailable: chuỗi OCR hỏng (lỗi hệ thống, không phải lỗi của file) — dừng cả lượt.
    """
    t0 = time.time()
    fmt = detect_format(path)
    out = {"format": fmt, "pages": [], "error": None, "skipped": None, "parser_version": PARSER_VERSION,
           "extract_version": EXTRACT_VERSION}
    if fmt != "pdf":
        out["skipped"] = "unsupported_format"
    else:
        try:
            out["pages"] = extract_pdf(path, get_ocr, detect_garbled=True)
        except (ServerDown, EngineUnavailable):
            raise
        except Exception as exc:  # noqa: BLE001 — một PDF lỗi không được dừng cả lô
            out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    out["seconds"] = round(time.time() - t0, 2)
    return out


def run_extract(root: Path, data_root: Path, get_ocr: Callable[[], object | None], ocr_enabled: bool = True,
                limit_files: int | None = None, retry_skipped: bool = False, sample_docs: int | None = None,
                seed: int = 0) -> int:
    """
    Chọn PDF (select_files) rồi trích text mọi file chưa làm (xem vi_corpus.common.batch_extract.run_batch).

    Chạy được nhiều tiến trình cùng lúc (khoá theo file ở CLAIM_DIR). Cùng sample_docs + seed ở mọi tiến trình.

    Returns:
        Số PDF bị lỗi trong lần chạy này.

    Raises:
        ServerDown, EngineUnavailable: chuỗi OCR hỏng; PDF đã xong trước đó vẫn nằm trong checkpoint.
    """
    files = select_files(root, sample_docs, seed, limit_files)
    logger.info("VJOL: %d PDF được chọn (sample_docs=%s, seed=%s)", len(files), sample_docs, seed)
    return run_batch(root, data_root, TEXT_DIR, extract_file, path_meta, get_ocr, ocr_enabled=ocr_enabled,
                     desc="Trích text VJOL", retry_skipped=retry_skipped, files=files, claim_dir=CLAIM_DIR,
                     extract_version=EXTRACT_VERSION)


def report(root: Path | None, data_root: Path) -> dict:
    """
    Thống kê từ checkpoint (không băm file): để biết tiến độ và báo lại sau mỗi lần chạy (J-05).

    Args:
        root: Thư mục raw/vjol để đếm số PDF có trong kho (None thì bỏ qua).

    Returns:
        {"pdfs_in_raw", "checkpoints", "errors", "skipped", "pages": {method: số trang}, "ocr_tier": {engine: số trang ok},
         "ocr_status": {ok | blank | flagged: số trang}, "flagged_rate": tỷ lệ trang flagged trên trang OCR VLM,
         "seconds": tổng giây trích text, "pages_per_doc": số trang trung bình mỗi bài, "old_version": số checkpoint ghi bởi bản
         trích text cũ (sẽ bị làm lại), "garbled_pages": {lý do: số trang lỗi font}, "garbled_docs": số bài có trang lỗi font}.
    """
    info = {"pdfs_in_raw": sum(len(v) for v in list_pdfs_by_year(root).values()) if root and root.is_dir() else None,
            "checkpoints": 0, "errors": 0, "skipped": 0, "seconds": 0.0}
    methods, tiers, statuses, garbled = Counter(), Counter(), Counter(), Counter()
    garbled_docs = old_version = 0
    for ckpt in sorted((data_root / TEXT_DIR).glob("*.json")):
        done = read_json(ckpt)
        if done is None:
            continue
        info["checkpoints"] += 1
        info["errors"] += bool(done.get("error"))
        info["skipped"] += bool(done.get("skipped"))
        info["seconds"] += done.get("seconds") or 0.0
        old_version += done.get("extract_version") != EXTRACT_VERSION
        reasons = [p["reason"] for p in done["pages"] if p.get("reason")]
        garbled.update(reasons)
        garbled_docs += bool(reasons)
        for page in done["pages"]:
            methods[page["method"]] += 1
            if page["method"] == "ocr_vlm":
                statuses[page["status"]] += 1
                if page["status"] == "ok":
                    tiers[page["tier"]] += 1
    n_vlm = sum(statuses.values())
    return {**info, "seconds": round(info["seconds"], 1), "pages": dict(methods), "ocr_tier": dict(tiers),
            "ocr_status": dict(statuses), "old_version": old_version, "garbled_pages": dict(garbled),
            "garbled_docs": garbled_docs, "flagged_rate": statuses["flagged"] / n_vlm if n_vlm else None,
            "pages_per_doc": sum(methods.values()) / info["checkpoints"] if info["checkpoints"] else None}
