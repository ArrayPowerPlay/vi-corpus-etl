"""
Nối chuỗi engine OCR VLM (vi_corpus.ocr_bakeoff.cascade, O-20) vào các nguồn PDF: phần dùng chung giữa stbook, VJOL
và (sau này) giáo trình.

- run_pages: OCR nhiều trang PDF song song (tối đa `concurrency` trang gửi đồng thời), trả kết quả theo thứ tự trang; trang lỗi font
  (J-10) được vẽ lại cả trang thay vì lấy ảnh nhúng.
- CascadePageOcr: bộ OCR có đúng giao diện mà vi_corpus.common.pdf_text.extract_pdf cần (.ocr_pages): extract_pdf gom các
  trang scan của một PDF rồi gửi một lượt, thay vì OCR tuần tự từng trang (J-04).
- CascadeOcrGetter: hàm `get_ocr` nạp lười. Chỉ khi gặp trang scan đầu tiên mới dựng server vLLM (nếu --serve), chờ server
  sẵn sàng và dựng chuỗi engine; tài liệu toàn trang có chữ không tốn thời gian khởi động GPU. Lỗi dựng bị nhớ lại và ném lại
  dưới dạng EngineUnavailable (dừng cả lượt, không loại trang âm thầm, O-24).
"""

import logging
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from vi_corpus.ocr_bakeoff.cascade import CascadeOcr, EngineUnavailable, Tier

logger = logging.getLogger("vi_corpus")

# Định dạng ảnh vLLM chắc chắn đọc được; ảnh nhúng ở định dạng khác (tiff, jb2, ...) được render lại thành PNG.
SAFE_IMAGE_EXT = frozenset({"jpg", "png", "webp", "jp2"})


def page_image_for_vlm(page, render: bool = False) -> tuple[bytes, str]:
    """
    Ảnh của một trang PDF để gửi cho VLM: ảnh nhúng gốc (như stbook) nếu định dạng đọc được, không thì render 200 dpi PNG.

    Args:
        page:   Trang pymupdf.
        render: True thì luôn vẽ lại cả trang (200 dpi PNG), không lấy ảnh nhúng. Dùng cho trang có lớp chữ hỏng (J-10): trang chữ
                có hình minh hoạ sẽ bị lấy nhầm hình thay vì cả trang.

    Returns:
        (byte ảnh, phần mở rộng).
    """
    from vi_corpus.ocr_bakeoff.pages import page_image_bytes

    if render:
        return page.get_pixmap(dpi=200).tobytes("png"), "png"
    data, ext = page_image_bytes(page)
    if ext not in SAFE_IMAGE_EXT:
        return page.get_pixmap(dpi=200).tobytes("png"), "png"
    return data, ext


def run_pages(cascade: CascadeOcr, pages: Iterable, concurrency: int = 32, render: Iterable[bool] | None = None) -> list[dict]:
    """
    OCR các trang PDF (pymupdf.Page) bằng chuỗi engine, tối đa `concurrency` trang gửi đồng thời; trả kết quả theo thứ tự trang.

    `render` (cùng thứ tự với `pages`) cho biết trang nào phải vẽ lại cả trang thay vì lấy ảnh nhúng (xem page_image_for_vlm); None = không trang nào.

    Chỉ giữ khoảng 2 x concurrency ảnh trong bộ nhớ (sách 900 trang không nạp hết một lúc). Một luồng ném lỗi (ServerDown,
    EngineUnavailable) thì dừng cả lô và ném lên. `pages` phải còn sống (tài liệu chưa đóng) trong suốt lúc gọi.

    Returns:
        Danh sách kết quả CascadeOcr.page ({"text", "status", "tier", ...}), cùng thứ tự với `pages`.
    """
    flags = None if render is None else list(render)
    results: dict[int, dict] = {}
    pending: dict = {}
    count = 0
    with ThreadPoolExecutor(concurrency) as pool:
        try:
            for i, page in enumerate(pages):
                data, ext = page_image_for_vlm(page, bool(flags and flags[i]))
                pending[pool.submit(cascade.page, data, ext)] = i
                count = i + 1
                while len(pending) >= 2 * concurrency:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        results[pending.pop(fut)] = fut.result()
            for fut in list(pending):
                results[pending.pop(fut)] = fut.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return [results[i] for i in range(count)]


class CascadePageOcr:
    """Bộ OCR cho extract_pdf dựa trên chuỗi engine VLM: nhận cả lô trang của một PDF và OCR song song."""

    def __init__(self, cascade: CascadeOcr, concurrency: int = 32) -> None:
        """
        Args:
            cascade:     Chuỗi engine đã dựng (CascadeOcr).
            concurrency: Số trang gửi đồng thời cho server.
        """
        self.cascade = cascade
        self.concurrency = concurrency

    def ocr_pages(self, pages: Iterable, render: Iterable[bool] | None = None) -> list[dict]:
        """
        OCR các trang (pymupdf.Page) rồi đổi mỗi kết quả thành một mục trang của extract_pdf.

        `render`: cờ từng trang, True = vẽ lại cả trang thay vì lấy ảnh nhúng (trang lỗi font, J-10).

        Returns:
            Danh sách {"text", "method": "ocr_vlm", "tier", "status"} (+ "rejected_text" nếu trang bị loại), theo thứ tự trang.
            Trang flagged / blank có text rỗng.
        """
        out = []
        for r in run_pages(self.cascade, pages, self.concurrency, render):
            entry = {"text": r["text"], "method": "ocr_vlm", "tier": r["tier"], "status": r["status"]}
            if "rejected_text" in r:
                entry["rejected_text"] = r["rejected_text"]
            out.append(entry)
        return out

    def page_text(self, image) -> str:
        """OCR một ảnh PIL (giao diện giống PageOcr, dùng khi không có lô trang)."""
        return self.cascade.page_text(image)


class CascadeOcrGetter:
    """
    Hàm `get_ocr` cho vi_corpus.common.batch_extract / extract_pdf: dựng chuỗi engine ở lần gọi đầu và nhớ kết quả.

    Gọi `close()` ở cuối lần chạy để tắt server vLLM do chính nó dựng.
    """

    def __init__(self, tiers: list[Tier], concurrency: int = 32, enabled: bool = True, serve: bool = False,
                 gpu: str | None = None, vllm_bin: str | None = None, state_dir: Path | None = None,
                 server_timeout: float = 3600) -> None:
        """
        Args:
            tiers:          Chuỗi engine (vi_corpus.ocr_bakeoff.cascade.build_tiers).
            concurrency:    Số trang gửi đồng thời cho server.
            enabled:        False thì get_ocr luôn trả None (trang scan thành needs_ocr).
            serve:          Tự dựng server vLLM bằng lệnh "serve" của engine VLM đầu chuỗi.
            gpu:            CUDA_VISIBLE_DEVICES cho server.
            vllm_bin:       Đường dẫn lệnh vllm (môi trường riêng).
            state_dir:      Thư mục ghi server.log (cần khi serve).
            server_timeout: Thời gian chờ server sẵn sàng (giây).
        """
        if serve and state_dir is None:
            raise ValueError("serve=True cần state_dir (nơi ghi server.log)")
        self.tiers, self.concurrency, self.enabled = tiers, concurrency, enabled
        self.serve, self.gpu, self.vllm_bin, self.state_dir = serve, gpu, vllm_bin, state_dir
        self.server_timeout = server_timeout
        self._ocr: CascadePageOcr | None = None
        self._error: Exception | None = None
        self._proc = None

    def __call__(self) -> CascadePageOcr | None:
        """
        Bộ OCR dùng chung cho cả lần chạy, hoặc None nếu tắt OCR.

        Raises:
            EngineUnavailable: dựng server / chuỗi engine không được (lần đầu và mọi lần sau; không thử lại).
        """
        if not self.enabled:
            return None
        if self._error is not None:
            raise self._error
        if self._ocr is None:
            try:
                self._ocr = self._build()
            except Exception as exc:  # noqa: BLE001 — nhớ lỗi để không dựng lại server cho từng file
                self._error = exc if isinstance(exc, EngineUnavailable) else EngineUnavailable(
                    f"Không dựng được chuỗi OCR: {type(exc).__name__}: {exc}")
                raise self._error from exc
        return self._ocr

    def _build(self) -> CascadePageOcr:
        """Dựng server (nếu cần), chờ sẵn sàng rồi dựng CascadeOcr."""
        from vi_corpus.ocr_bakeoff.engines import start_server, wait_ready

        vlm = next((t for t in self.tiers if t.cfg.get("type") == "openai"), None)
        if vlm is not None:
            if self.serve:
                self.state_dir.mkdir(parents=True, exist_ok=True)
                self._proc = start_server(vlm.cfg, self.state_dir, self.gpu, self.vllm_bin)
            wait_ready(vlm.cfg["base_url"], vlm.cfg.get("health", "/models"), self.server_timeout, self._proc)
        logger.info("Chuỗi OCR sẵn sàng: %s, %d trang đồng thời", " -> ".join(t.name for t in self.tiers), self.concurrency)
        return CascadePageOcr(CascadeOcr(self.tiers), self.concurrency)

    def close(self) -> None:
        """Tắt server vLLM do getter này dựng (không làm gì nếu chưa dựng)."""
        from vi_corpus.ocr_bakeoff.engines import stop_server

        if self._proc is not None:
            stop_server(self._proc)
            self._proc = None
