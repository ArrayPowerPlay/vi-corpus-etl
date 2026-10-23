"""
Nối chuỗi engine OCR VLM (vi_corpus.ocr_bakeoff.cascade, O-20) vào các nguồn PDF: phần dùng chung giữa stbook, VJOL
và (sau này) giáo trình.

- run_pages: OCR nhiều trang PDF song song (tối đa `concurrency` trang gửi đồng thời), trả kết quả theo thứ tự trang; trang lỗi font
  (J-10) được vẽ lại cả trang thay vì lấy ảnh nhúng.
- CascadePageOcr: bộ OCR có đúng giao diện mà vi_corpus.common.pdf_text.extract_pdf cần (.ocr_pages): extract_pdf gom các
  trang scan của một PDF rồi gửi một lượt, thay vì OCR tuần tự từng trang (J-04).
- Trang xoay (J-24): trang scan bị xoay 180 độ (hoặc 90 / 270) làm cả chuỗi VLM bị báo rồi rơi xuống engine baseline, engine này đọc ra chữ
  ngược rác mà bộ kiểm trang không thấy (status "ok"). Khi CascadePageOcr có tập âm tiết (vocab), mỗi trang kết thúc ở engine baseline
  hoặc bị loại sẽ được xoay thử 180 / 90 / 270 độ và chạy lại chuỗi; bản xoay chỉ thắng khi tỷ lệ âm tiết hợp lệ cao hơn rõ rệt
  (ROTATE_MIN_GAIN, ROTATE_MIN_RATE). Không có vocab thì không xoay (giữ hành vi cũ cho stbook, giáo trình).
- CascadeOcrGetter: hàm `get_ocr` nạp lười. Chỉ khi gặp trang scan đầu tiên mới dựng server vLLM (nếu --serve), chờ server
  sẵn sàng và dựng chuỗi engine; tài liệu toàn trang có chữ không tốn thời gian khởi động GPU. Lỗi dựng bị nhớ lại và ném lại
  dưới dạng EngineUnavailable (dừng cả lượt, không loại trang âm thầm, O-24).
"""

import io
import logging
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from vi_corpus.common.garbled import unknown_syllable_rate
from vi_corpus.common.ocr_cache import ocr_signature
from vi_corpus.ocr_bakeoff.cascade import CascadeOcr, EngineUnavailable, Tier

logger = logging.getLogger("vi_corpus")

# Xoay trang (J-24). Các hằng số là điểm khởi đầu, chưa đo trên trang thật.
ROTATIONS = (180, 90, 270)  # góc thử, theo thứ tự thường gặp; dừng ngay khi một góc đạt ROTATE_GOOD_RATE
ROTATE_MIN_WORDS = 20  # văn bản ít từ Latinh hơn mức này thì không đủ để so, giữ bản gốc
ROTATE_GOOD_RATE = 0.85  # tỷ lệ âm tiết hợp lệ từ mức này là đọc ổn (cùng ngưỡng lọc trang baseline của stbook, O-30), dừng thử góc khác
ROTATE_MIN_GAIN = 0.20  # bản xoay phải hơn bản gốc ít nhất chừng này
ROTATE_MIN_RATE = 0.50  # và tự nó phải đạt ít nhất chừng này

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


def rotate_image(data: bytes, angle: int) -> tuple[bytes, str]:
    """Xoay ảnh `angle` độ (ngược chiều kim đồng hồ, mở rộng khung cho góc 90 / 270) rồi mã hoá PNG. Returns (byte ảnh, "png")."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        buf = io.BytesIO()
        img.convert("RGB").rotate(angle, expand=True).save(buf, format="PNG")
    return buf.getvalue(), "png"


def valid_rate(text: str | None, vocab: frozenset[str]) -> float | None:
    """Tỷ lệ từ Latinh là âm tiết hợp lệ (1 - tỷ lệ âm tiết lạ) của văn bản; None nếu dưới ROTATE_MIN_WORDS từ Latinh (không đủ để so)."""
    unknown, n = unknown_syllable_rate(text or "", vocab)
    return None if n < ROTATE_MIN_WORDS else 1.0 - unknown


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

    def __init__(self, cascade: CascadeOcr, concurrency: int = 32, vocab: frozenset[str] | None = None) -> None:
        """
        Args:
            cascade:     Chuỗi engine đã dựng (CascadeOcr).
            concurrency: Số trang gửi đồng thời cho server.
            vocab:       Tập âm tiết tham chiếu; có thì bật xoay thử trang (J-24), None thì không xoay.
        """
        self.cascade = cascade
        self.concurrency = concurrency
        self.vocab = vocab

    def _needs_rotation_check(self, r: dict) -> bool:
        """Trang bị loại, hoặc do engine tuần tự (baseline) đọc sau khi mọi engine VLM bị báo: nghi trang xoay."""
        return r["status"] == "flagged" or any(t.name == r["tier"] and t.serial for t in self.cascade.tiers)

    def _try_rotations(self, page, render: bool, first: dict) -> dict:
        """
        Xoay thử trang nghi ngờ (ROTATIONS) và chạy lại chuỗi engine; trả về kết quả tốt nhất.

        So bằng tỷ lệ âm tiết hợp lệ (valid_rate). Giữ `first` trừ khi bản xoay hơn ít nhất ROTATE_MIN_GAIN và đạt ROTATE_MIN_RATE; không đủ từ để
        so (văn bản gốc hoặc bản xoay) thì cũng giữ `first`. Bản được chọn có thêm khoá "rotation" (góc).
        """
        base = valid_rate(first["text"] or first.get("rejected_text"), self.vocab)
        data, ext = page_image_for_vlm(page, render)
        best, best_rate = first, base if base is not None else 0.0
        for angle in ROTATIONS:
            rot_data, rot_ext = rotate_image(data, angle)
            res = self.cascade.page(rot_data, rot_ext)
            rate = valid_rate(res["text"], self.vocab)
            if rate is not None and rate >= ROTATE_MIN_RATE and rate >= (base or 0.0) + ROTATE_MIN_GAIN and rate > best_rate:
                best, best_rate = {**res, "rotation": angle}, rate
                if rate >= ROTATE_GOOD_RATE:
                    break
        return best

    def ocr_pages(self, pages: Iterable, render: Iterable[bool] | None = None) -> list[dict]:
        """
        OCR các trang (pymupdf.Page) rồi đổi mỗi kết quả thành một mục trang của extract_pdf.

        `render`: cờ từng trang, True = vẽ lại cả trang thay vì lấy ảnh nhúng (trang lỗi font, J-10).

        Returns:
            Danh sách {"text", "method": "ocr_vlm", "tier", "status"} (+ "rotation" nếu phải xoay mới đọc được, + "rejected_text" nếu trang bị loại), theo thứ tự trang.
            Trang flagged / blank có text rỗng.
        """
        pages = list(pages)
        flags = [False] * len(pages) if render is None else list(render)
        results = run_pages(self.cascade, pages, self.concurrency, flags)
        if self.vocab is not None:
            todo = [i for i, r in enumerate(results) if self._needs_rotation_check(r)]
            if todo:
                with ThreadPoolExecutor(self.concurrency) as pool:
                    for i, fixed in zip(todo, pool.map(lambda i: self._try_rotations(pages[i], flags[i], results[i]), todo)):
                        results[i] = fixed
                logger.info("Xoay thử %d trang nghi ngờ, %d trang đọc được sau khi xoay", len(todo), sum("rotation" in results[i] for i in todo))
        out = []
        for r in results:
            entry = {"text": r["text"], "method": "ocr_vlm", "tier": r["tier"], "status": r["status"]}
            if "rotation" in r:
                entry["rotation"] = r["rotation"]
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
                 server_timeout: float = 3600, vocab: frozenset[str] | None = None) -> None:
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
            vocab:          Tập âm tiết; có thì CascadePageOcr xoay thử trang nghi ngờ (J-24).
        """
        if serve and state_dir is None:
            raise ValueError("serve=True cần state_dir (nơi ghi server.log)")
        self.tiers, self.concurrency, self.enabled = tiers, concurrency, enabled
        self.serve, self.gpu, self.vllm_bin, self.state_dir = serve, gpu, vllm_bin, state_dir
        self.server_timeout = server_timeout
        self.vocab = vocab
        self._ocr: CascadePageOcr | None = None
        self._error: Exception | None = None
        self._proc = None

    def signature(self) -> str:
        """Chữ ký bộ OCR cho bộ nhớ đệm OCR (vi_corpus.common.ocr_cache.ocr_signature): chuỗi engine + có xoay thử trang hay không."""
        return ocr_signature([t.name for t in self.tiers], self.vocab is not None)

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
        return CascadePageOcr(CascadeOcr(self.tiers), self.concurrency, self.vocab)

    def close(self) -> None:
        """Tắt server vLLM do getter này dựng (không làm gì nếu chưa dựng)."""
        from vi_corpus.ocr_bakeoff.engines import stop_server

        if self._proc is not None:
            stop_server(self._proc)
            self._proc = None
