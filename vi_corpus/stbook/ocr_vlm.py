"""
OCR sách stbook bằng chuỗi engine mô hình thị giác - ngôn ngữ (O-20): Qwen3-VL-8B (`qwen3vl_8b_mt4k`) -> chạy lại trang bị
bộ kiểm báo lỗi bằng `qwen3vl_8b_rp110` -> còn bị báo thì dùng OCR cũ (PaddleOCR detect + VietOCR). Xem vi_corpus.ocr_bakeoff.cascade.

Cùng thư mục kết quả với bộ OCR cũ (vi_corpus.stbook.ocr_books): interim/stbook_ocr/<danh-mục>/<product_id>.json, nên
`iter_records` đọc được không cần sửa. Khác với bộ cũ:
    * trường "pages" vẫn là danh sách text từng trang, nhưng trang bị loại (engine cuối vẫn bị báo) có text "" (không vào kho);
    * thêm "page_info" (mỗi trang: status "ok" | "blank" | "flagged", engine được chọn, finish_reason, mã lý do bộ kiểm, các
      lần thử; trang flagged giữ "rejected_text" để kiểm tay) và "stats" (số trang theo engine, số trang trắng / bị loại);
    * "ocr": {"engine": "vlm_cascade", ...}. Cuốn đã có kết quả của engine khác (bộ OCR cũ) bị coi là CHƯA làm, được OCR lại; file
      cũ được chép sang interim/stbook_ocr_baseline/ trước khi ghi đè.
Checkpoint theo cuốn (ghi nguyên tử, làm lại nếu kích thước PDF đổi). Nhiều tiến trình (mỗi GPU một server vLLM) có thể chạy cùng lúc:
mỗi cuốn được "nhận" bằng một khoá flock (state/stbook_ocr_vlm/claims/), cuốn nào đã có người làm thì bỏ qua, tiến trình chết
thì khoá tự nhả. Các trang của một cuốn được gửi tới server song song (`concurrency` yêu cầu cùng lúc).
"""

import fcntl
import json
import logging
import shutil
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from vi_corpus.common.state import read_json, write_json_atomic
from vi_corpus.ocr_bakeoff.cascade import CascadeOcr, Tier
from vi_corpus.stbook.ocr_books import OCR_DIR, _ocr_path, find_books

logger = logging.getLogger("vi_corpus")

ENGINE_TAG = "vlm_cascade"
BACKUP_DIR = "interim/stbook_ocr_baseline"  # bản kết quả của bộ OCR cũ, chép sang đây trước khi bị VLM ghi đè
CLAIM_DIR = "state/stbook_ocr_vlm/claims"


def is_done(done: dict | None, pdf: Path) -> bool:
    """Kết quả `done` (nội dung file JSON hoặc None) là của chuỗi VLM và còn khớp kích thước PDF."""
    return bool(done and done.get("pdf_size") == pdf.stat().st_size and (done.get("ocr") or {}).get("engine") == ENGINE_TAG)


def books_to_ocr_vlm(stbook_root: Path, data_root: Path, limit_books: int | None = None,
                     redo: bool = False) -> list[tuple[str, dict, Path]]:
    """
    Các cuốn cần OCR bằng chuỗi VLM: chưa có kết quả của chuỗi VLM (kể cả cuốn đã OCR bằng bộ cũ), hoặc kích thước PDF đã đổi.

    Args:
        redo: Coi mọi cuốn là chưa làm.

    Returns:
        Danh sách (slug danh mục, metadata sách, đường dẫn PDF), tối đa limit_books cuốn.
    """
    todo = [(slug, book, pdf) for slug, book, pdf in find_books(stbook_root)
            if redo or not is_done(read_json(_ocr_path(data_root, slug, book["product_id"])), pdf)]
    return todo[:limit_books]


def vlm_status(stbook_root: Path, data_root: Path) -> dict:
    """{"vlm": số cuốn xong bằng chuỗi VLM, "other": số cuốn xong bằng bộ OCR cũ (sẽ được làm lại), "total": số cuốn có PDF}."""
    out = {"vlm": 0, "other": 0, "total": 0}
    for slug, book, pdf in find_books(stbook_root):
        out["total"] += 1
        done = read_json(_ocr_path(data_root, slug, book["product_id"]))
        if is_done(done, pdf):
            out["vlm"] += 1
        elif done and done.get("pdf_size") == pdf.stat().st_size:
            out["other"] += 1
    return out


@contextmanager
def claim_book(data_root: Path, slug: str, product_id: str) -> Iterator[bool]:
    """
    Nhận một cuốn để làm: giữ khoá flock không chặn trên state/stbook_ocr_vlm/claims/<slug>__<product_id>.lock suốt khối `with`.

    Yields:
        True nếu nhận được, False nếu tiến trình khác đang làm cuốn này.
    """
    path = data_root / CLAIM_DIR / f"{slug}__{product_id}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def _run_pages(cascade: CascadeOcr, doc, concurrency: int) -> list[dict]:
    """
    OCR mọi trang của PDF `doc` (pymupdf.Document), tối đa `concurrency` trang gửi đồng thời; trả kết quả theo thứ tự trang.

    Chỉ giữ khoảng 2 x concurrency ảnh trong bộ nhớ (sách 900 trang không nạp hết một lúc). Một luồng ném lỗi (ServerDown) thì dừng
    cả cuốn và ném lên.
    """
    from vi_corpus.ocr_bakeoff.pages import page_image_bytes

    results: list[dict | None] = [None] * doc.page_count
    pending: dict = {}
    with ThreadPoolExecutor(concurrency) as pool:
        try:
            for i, page in enumerate(doc):
                data, ext = page_image_bytes(page)
                pending[pool.submit(cascade.page, data, ext)] = i
                while len(pending) >= 2 * concurrency:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        results[pending.pop(fut)] = fut.result()
            for fut in list(pending):
                results[pending.pop(fut)] = fut.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return results  # type: ignore[return-value]


def ocr_book_vlm(cascade: CascadeOcr, slug: str, book: dict, pdf: Path, data_root: Path, concurrency: int = 16) -> dict:
    """
    OCR một cuốn bằng chuỗi VLM và ghi kết quả nguyên tử (định dạng ở docstring module).

    Args:
        cascade:     Chuỗi engine đã dựng (vi_corpus.ocr_bakeoff.cascade.CascadeOcr).
        concurrency: Số trang gửi đồng thời cho server.

    Returns:
        Phần "stats" đã ghi.

    Raises:
        ValueError: PDF hỏng / bị cắt cụt (PyMuPDF phải sửa khi mở), giống bộ OCR cũ.
        ServerDown: server VLM không trả lời (cuốn không được ghi; chạy lại sau khi server sống lại).
    """
    import pymupdf

    t0 = time.time()
    with pymupdf.open(pdf) as doc:
        if doc.is_repaired:
            raise ValueError("PDF hỏng, cần xoá và tải lại bằng stbook-crawler")
        results = _run_pages(cascade, doc, concurrency)
    info = []
    for r in results:
        item = {k: r[k] for k in ("status", "tier", "finish_reason", "reasons") if k in r}
        item.update({k: r[k] for k in ("attempts", "rejected_text") if k in r})
        info.append(item)
    by_tier = Counter(r["tier"] for r in results if r["status"] == "ok")
    stats = {"pages": len(results), "ok_by_tier": dict(by_tier), "blank": sum(r["status"] == "blank" for r in results),
             "flagged": sum(r["status"] == "flagged" for r in results),
             "rerun": sum(len(r.get("attempts", ())) > 1 for r in results), "seconds": round(time.time() - t0, 1)}
    out = _ocr_path(data_root, slug, book["product_id"])
    old = read_json(out)
    if old and (old.get("ocr") or {}).get("engine") != ENGINE_TAG:  # giữ kết quả của bộ OCR cũ trước khi ghi đè
        backup = data_root / BACKUP_DIR / slug / out.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, backup)
    write_json_atomic(out, {
        "category_slug": slug,
        "book": book,
        "pdf_path": str(pdf.relative_to(data_root)) if pdf.is_relative_to(data_root) else str(pdf),
        "pdf_size": pdf.stat().st_size,
        "ocr": {"engine": ENGINE_TAG, "tiers": [_tier_info(t) for t in cascade.tiers]},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pages": [r["text"] for r in results],
        "page_info": info,
        "stats": stats,
    })
    return stats


def _tier_info(tier: Tier) -> dict:
    """Mô tả một engine của chuỗi để ghi vào kết quả (đủ để biết kết quả sinh ra bằng cấu hình nào)."""
    cfg = tier.cfg
    return {"name": tier.name, "type": cfg.get("type"), "model": cfg.get("model"), "max_tokens": cfg.get("max_tokens"),
            "extra_body": cfg.get("extra_body"), "prompt": cfg.get("prompt")}


def run_ocr_vlm(stbook_root: Path, data_root: Path, cascade: CascadeOcr, concurrency: int = 16,
                limit_books: int | None = None, redo: bool = False) -> int:
    """
    OCR mọi cuốn chưa làm bằng chuỗi VLM, ghi kết quả từng cuốn ngay khi xong (chạy lại để làm tiếp).

    Lỗi ở một cuốn (PDF hỏng, ...) chỉ ghi log rồi làm cuốn khác. ServerDown thì dừng cả lượt (server chết thì mọi cuốn sau cũng
    hỏng). Chạy được nhiều tiến trình cùng lúc (xem claim_book).

    Returns:
        Số cuốn bị lỗi.
    """
    from vi_corpus.ocr_bakeoff.cascade import ServerDown

    todo = books_to_ocr_vlm(stbook_root, data_root, limit_books, redo)
    logger.info("Cần OCR %d cuốn bằng chuỗi VLM (%s), %d trang đồng thời", len(todo),
                " -> ".join(t.name for t in cascade.tiers), concurrency)
    failed = done = 0
    started = datetime.now(timezone.utc).isoformat()
    for slug, book, pdf in todo:
        pid = str(book["product_id"])
        with claim_book(data_root, slug, pid) as got:
            if not got:
                continue  # tiến trình khác đang làm cuốn này
            now = read_json(_ocr_path(data_root, slug, pid))
            if is_done(now, pdf) and (not redo or now["created_at"] >= started):
                continue  # làm xong rồi (redo: chỉ tính bản làm sau khi lượt này bắt đầu)
            try:
                st = ocr_book_vlm(cascade, slug, book, pdf, data_root, concurrency)
            except ServerDown:
                logger.error("Server VLM không trả lời, dừng. Dựng lại server rồi chạy lại đúng lệnh cũ.")
                raise
            except Exception as exc:  # noqa: BLE001 — một cuốn lỗi không được dừng cả lượt
                failed += 1
                logger.warning("Lỗi OCR %s: %s", pdf, exc)
                continue
            done += 1
            logger.info("[%s/%s] %d trang, %.2f trang/s | chọn từ %s | chạy lại %d, trắng %d, bị loại %d", slug, pid, st["pages"],
                        st["pages"] / max(st["seconds"], 0.1), st["ok_by_tier"], st["rerun"], st["blank"], st["flagged"])
    logger.info("Xong: %d cuốn làm bởi tiến trình này, %d lỗi", done, failed)
    return failed


def flagged_report(data_root: Path) -> dict:
    """
    Tổng hợp trang bị loại / trắng / chạy lại trên mọi cuốn đã OCR bằng chuỗi VLM (đọc "stats" của từng file).

    Returns:
        {"books", "pages", "ok_by_tier", "blank", "flagged", "rerun", "worst": [(slug/product_id, số trang bị loại)] top 10}.
    """
    tot = Counter({"pages": 0, "blank": 0, "flagged": 0, "rerun": 0})
    ok_by_tier: Counter = Counter()
    worst = []
    books = 0
    for path in sorted((data_root / OCR_DIR).glob("*/*.json")):
        done = read_json(path)
        if not done or (done.get("ocr") or {}).get("engine") != ENGINE_TAG:
            continue
        st = done["stats"]
        books += 1
        tot.update({k: st[k] for k in ("pages", "blank", "flagged", "rerun")})
        ok_by_tier.update(st["ok_by_tier"])
        worst.append((f"{path.parent.name}/{path.stem}", st["flagged"]))
    worst.sort(key=lambda x: -x[1])
    return {"books": books, **tot, "ok_by_tier": dict(ok_by_tier), "worst": worst[:10]}


def dump_report(data_root: Path) -> str:
    """flagged_report dưới dạng JSON dễ đọc (cho --status)."""
    return json.dumps(flagged_report(data_root), ensure_ascii=False, indent=2)
