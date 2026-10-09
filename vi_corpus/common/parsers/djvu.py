"""
Trích text DJVU bằng djvulibre: djvused đếm trang, djvutxt lấy lớp chữ từng trang; trang không có chữ thì ddjvu vẽ ra
ảnh rồi OCR (cùng bộ OCR với PDF scan). Cài trên server: docs/README.md Phần E (Q5).
"""

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from vi_corpus.common.parsers import MissingTool
from vi_corpus.common.pdf_text import MIN_TEXT_CHARS

TIMEOUT = 120  # giây cho một lệnh djvulibre


def _run(cmd: list[str]) -> str:
    """Chạy một lệnh djvulibre, trả stdout; lỗi thì RuntimeError kèm stderr."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} lỗi: {proc.stderr.strip()[:200]}")
    return proc.stdout


def extract_djvu(path: Path, get_ocr: Callable[[], object | None]) -> list[dict]:
    """
    Trích text từng trang của một file DJVU.

    Returns:
        [{"text", "method"}] theo thứ tự trang; method "djvu" (có lớp chữ), "ocr", "needs_ocr" hoặc "empty".

    Raises:
        MissingTool: chưa cài djvulibre (djvused, djvutxt, ddjvu).
    """
    if not all(shutil.which(t) for t in ("djvused", "djvutxt", "ddjvu")):
        raise MissingTool("needs_djvulibre")
    n_pages = int(_run(["djvused", str(path), "-e", "n"]).strip())
    pages = []
    for i in range(1, n_pages + 1):
        text = _run(["djvutxt", f"--page={i}", str(path)])
        if len("".join(text.split())) >= MIN_TEXT_CHARS:
            pages.append({"text": text, "method": "djvu"})
            continue
        ocr = get_ocr()
        if ocr is None:
            pages.append({"text": text, "method": "needs_ocr" if not text.strip() else "djvu"})
            continue
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="vi_corpus_djvu_") as tmp:
            img_path = Path(tmp) / "page.ppm"
            _run(["ddjvu", "-format=ppm", f"-page={i}", str(path), str(img_path)])
            with Image.open(img_path) as img:
                ocr_text = ocr.page_text(img.convert("RGB"))
        pages.append({"text": ocr_text, "method": "ocr" if ocr_text.strip() else "empty"})
    return pages
