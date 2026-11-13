"""
Chạy thử OCR nội bộ (VNPT SmartReader, API bất đồng bộ "gen-ai") trên mẫu PDF ngẫu nhiên của VJOL và VISTA.

Việc làm, theo thứ tự:
  1. Chọn mẫu: mỗi nguồn (vjol, vista) lấy ngẫu nhiên N file PDF (mặc định 30), chia đều hai loại PDF:
       - "1 lớp": PDF chỉ có ảnh scan, không có lớp chữ (layer 1);
       - "2 lớp": PDF có ảnh nền + lớp chữ (layer 2). Ưu tiên file mà lớp chữ nằm trên ảnh trang (đúng nghĩa 2 lớp: scan đã OCR),
         thiếu thì mới lấy file chỉ có chữ (born-digital) cho đủ số; mỗi file ghi rõ nó thuộc dạng nào ("layer2_kind").
     Thiếu một loại thì báo thiếu, không lấy loại kia bù vào (để tỷ lệ không bị lệch trong im lặng).
  2. Gọi API OCR nội bộ cho từng file: upload (/file-service/v1/addFile) -> tạo phiên (/rpa-service/aidigdoc/v1/integration/ocr/gen-ai)
     -> hỏi kết quả đến khi xong (.../gen-ai/result) -> tải kết quả (link hoặc trả thẳng trong phản hồi). Quá hạn hay Ctrl+C thì gọi
     .../gen-ai/cancel để không tốn thêm sản lượng. Mô tả API: file docx "Tài liệu mô tả API xử lý bất đồng bộ cho gen-ai" của VNPT AI.
  3. Ghi kết quả từng file (resume được, file đã xong thì bỏ qua) và báo cáo tổng hợp (số trang, thời gian, độ dài chữ; với file 2 lớp
     có đo độ trùng từ giữa OCR và lớp chữ có sẵn để biết OCR đọc đúng đến đâu).

Khoá riêng (không ghi vào code, đặt trong .env ở gốc repo hoặc biến môi trường):
    ACCESS_TOKEN   access token (có hay không có tiền tố "Bearer " đều được)
    TOKEN_ID       Token-id do VNPT cấp
    TOKEN_KEY      Token-key do VNPT cấp
(tên đầy đủ VNPT_OCR_ACCESS_TOKEN / VNPT_OCR_TOKEN_ID / VNPT_OCR_TOKEN_KEY cũng được và được ưu tiên hơn)
    VNPT_OCR_DOMAIN         (tuỳ chọn) mặc định https://api.idg.vnpt.vn
    VNPT_OCR_MAC_ADDRESS    (tuỳ chọn) header mac-address, mặc định "mac-address" như trong bộ Postman

Lưu ý sản lượng: mỗi trang OCR tính vào hạn mức của khoá. Mặc định chỉ lấy file không quá --max-pages (60) trang và --max-mb (30) MB;
--dry-run chỉ chọn mẫu và in danh sách, không gọi API; --limit-files N chỉ chạy N file đầu (xen kẽ nguồn và loại) để thử trước.

Lệnh:
    uv run python scripts/ocr_internal/trial_vjol_vista.py --data-root <root> --dry-run
    uv run python scripts/ocr_internal/trial_vjol_vista.py --data-root <root> --limit-files 4
    uv run python scripts/ocr_internal/trial_vjol_vista.py --data-root <root> [--per-source 30 --seed 42 --workers 3 --exporter json|md]
    uv run python scripts/ocr_internal/trial_vjol_vista.py --data-root <root> --report-only

Kết quả ở <data-root>/processed/ocr_internal_trial/<run-name>/: sample.json (mẫu đã chọn), results/<sha256>.json (chữ + trang + số liệu từng file),
raw/<sha256>.json|md (kết quả thô của API, giữ để đọc lại), report.md (kết luận + danh sách lỗi), report.csv, report.json. Mã thoát 1 nếu có file lỗi.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vi_corpus.common.garbled import control_char_ratio, garbled_ratio, load_vocab, unknown_syllable_rate  # noqa: E402  (cần sys.path ở trên)
from vi_corpus.vjol.font_canary import word_overlap  # noqa: E402

logger = logging.getLogger("ocr_internal_trial")

SOURCES = ("vjol", "vista")  # thư mục raw/<nguồn>/<năm>/*.pdf
DEFAULT_DOMAIN = "https://api.idg.vnpt.vn"
EP_UPLOAD = "/file-service/v1/addFile"
EP_SUBMIT = "/rpa-service/aidigdoc/v1/integration/ocr/gen-ai"
EP_RESULT = EP_SUBMIT + "/result"
EP_CANCEL = EP_SUBMIT + "/cancel"
SUCCESS_MESSAGE = "IDG-00000000"  # trường "message" khi thành công
PENDING_WARNING = "request_dang_trong_qua_trinh_xu_ly"  # cảnh báo khi phiên còn đang xử lý
MIN_TEXT_CHARS = 30  # trang có ít hơn số ký tự (không tính khoảng trắng) này coi là không có lớp chữ
PROBE_PAGES = 5  # số trang dò mỗi PDF để phân loại 1 lớp / 2 lớp
IMAGE_COVER = 0.6  # ảnh phủ >= tỉ lệ này diện tích trang thì coi là ảnh nền của trang
RETRY_STATUS = {408, 429, 500, 502, 503, 504}  # lỗi tạm thời: thử lại
FATAL_STATUS = {401, 403}  # sai khoá / hết sản lượng / không có quyền: dừng cả lần chạy
REQUEST_TIMEOUT = 120  # giây cho một request HTTP
# Ngưỡng thử để gắn cờ file OCR có vẻ kém và để ra kết luận chung; CHƯA chỉnh trên dữ liệu thật, đọc kèm số đo trong report.csv.
QUALITY_LIMITS = {"min_chars_per_page": 200, "min_page_coverage": 0.8, "max_unknown_rate": 0.15, "max_garbled": 0.05, "max_control": 0.01,
                  "max_dup_lines": 0.3, "min_overlap": 0.7, "max_error_rate": 0.05, "max_flagged_rate": 0.15}


class FatalApiError(RuntimeError):
    """Lỗi không thể thử lại và ảnh hưởng mọi file (sai khoá, hết sản lượng, không có quyền): dừng cả lần chạy."""


class ApiError(RuntimeError):
    """Lỗi của một file (API trả mã lỗi, quá hạn, kết quả không đọc được): ghi vào kết quả của file đó rồi chạy tiếp file khác."""


# --------------------------------------------------------------------------------------------
# Chọn mẫu
# --------------------------------------------------------------------------------------------

def classify_pdf(path: Path, probe_pages: int = PROBE_PAGES) -> dict | None:
    """
    Phân loại một PDF theo số lớp, dò tối đa `probe_pages` trang rải đều (không đọc hết file).

    Một trang "có chữ" khi lớp chữ có >= MIN_TEXT_CHARS ký tự; trang "có ảnh nền" khi có ảnh phủ >= IMAGE_COVER diện tích trang.
    Trả về:
        layer = 1 khi không trang dò nào có chữ (và có ảnh nền): PDF scan thuần;
        layer = 2 khi >= 60% trang dò có chữ; layer2_kind = "image_text" nếu >= 60% trang dò còn có ảnh nền (scan đã có lớp chữ),
        ngược lại "text_only" (PDF sinh từ phần mềm soạn thảo, không có ảnh nền);
        None cho file không đọc được, PDF bị cắt cụt (đã sửa), 0 trang, hoặc dạng lẫn lộn không thuộc loại nào.
    Dict trả về có thêm pages (tổng số trang) và probed (số trang đã dò).
    """
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001  - file hỏng thì bỏ qua, không dừng cả lần chọn
        logger.debug("không mở được %s: %s", path, exc)
        return None
    with doc:
        if doc.is_repaired or doc.page_count == 0 or doc.needs_pass:
            return None
        total = doc.page_count
        idx = sorted({round(i * (total - 1) / max(probe_pages - 1, 1)) for i in range(min(probe_pages, total))})
        text_pages = image_pages = 0
        for i in idx:
            try:
                page = doc[i]
                if len("".join(page.get_text("text").split())) >= MIN_TEXT_CHARS:
                    text_pages += 1
                area = abs(page.rect) or 1.0
                covered = max((abs(pymupdf.Rect(info["bbox"]) & page.rect) for info in page.get_image_info()), default=0.0)
                if covered / area >= IMAGE_COVER:
                    image_pages += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("lỗi trang %d của %s: %s", i, path, exc)
                return None
    n = len(idx)
    info = {"pages": total, "probed": n}
    if text_pages == 0 and image_pages > 0:
        return {**info, "layer": 1, "layer2_kind": None}
    if text_pages / n >= 0.6:
        return {**info, "layer": 2, "layer2_kind": "image_text" if image_pages / n >= 0.6 else "text_only"}
    return None


def sha256_of(path: Path) -> str:
    """sha256 nội dung file (đọc từng khối 1 MB), dùng làm khoá kết quả và để bỏ file trùng nội dung (VISTA có PDF dùng chung nhiều bài)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_pdfs(raw_dir: Path) -> list[Path]:
    """Mọi PDF dạng raw/<nguồn>/<thư mục>/<file>.pdf, bỏ thư mục metadata; sắp xếp để cùng seed cho cùng thứ tự."""
    return sorted(p for p in raw_dir.glob("*/*.pdf") if p.parent.name != "metadata")


def pick_source_sample(source: str, raw_dir: Path, n: int, seed: int, max_pages: int, max_mb: float, max_scan: int) -> tuple[list[dict], dict]:
    """
    Chọn mẫu ngẫu nhiên của một nguồn: n // 2 file 1 lớp và n - n // 2 file 2 lớp.

    Xáo thứ tự file theo seed rồi phân loại lần lượt cho đến khi đủ chỉ tiêu hoặc đã dò `max_scan` file. File 2 lớp "image_text" được
    ưu tiên, file "text_only" chỉ dùng để bù khi hết file dò được. File quá --max-pages / --max-mb hoặc trùng nội dung bị bỏ.
    Trả về (danh sách mục mẫu, thống kê gồm số file dò, số thiếu mỗi loại).
    """
    files = list_pdfs(raw_dir)
    random.Random(f"{seed}:{source}").shuffle(files)
    want = {1: n // 2, 2: n - n // 2}
    buckets: dict[str, list[dict]] = {"l1": [], "l2_image": [], "l2_text": []}
    seen: set[str] = set()
    scanned = 0
    for path in files:
        if len(buckets["l1"]) >= want[1] and len(buckets["l2_image"]) >= want[2]:
            break
        if scanned >= max_scan:
            break
        scanned += 1
        if path.stat().st_size > max_mb * 1024 * 1024:
            continue
        cls = classify_pdf(path)
        if cls is None or cls["pages"] > max_pages:
            continue
        key = "l1" if cls["layer"] == 1 else ("l2_image" if cls["layer2_kind"] == "image_text" else "l2_text")
        full = len(buckets[key]) >= (want[1] if key == "l1" else want[2])
        if full:
            continue
        digest = sha256_of(path)
        if digest in seen:
            continue
        seen.add(digest)
        buckets[key].append({"source": source, "path": str(path), "rel": str(path.relative_to(raw_dir)), "sha256": digest,
                             "size_mb": round(path.stat().st_size / 1048576, 2), **cls})
    layer2 = buckets["l2_image"] + buckets["l2_text"][: max(0, want[2] - len(buckets["l2_image"]))]
    sample = buckets["l1"] + layer2
    stats = {"files_in_raw": len(files), "scanned": scanned, "want_layer1": want[1], "got_layer1": len(buckets["l1"]),
             "want_layer2": want[2], "got_layer2": len(layer2), "layer2_image_text": len(buckets["l2_image"]),
             "layer2_text_only": len(layer2) - len(buckets["l2_image"])}
    return sample, stats


def interleave(sample: list[dict]) -> list[dict]:
    """Xếp xen kẽ theo (nguồn, loại) để --limit-files N lấy được mẫu đủ các nhóm thay vì N file đầu cùng một nhóm."""
    groups: dict[tuple, list[dict]] = {}
    for e in sample:
        groups.setdefault((e["source"], e["layer"]), []).append(e)
    out: list[dict] = []
    queues = list(groups.values())
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


# --------------------------------------------------------------------------------------------
# Client API OCR nội bộ
# --------------------------------------------------------------------------------------------

class OcrClient:
    """
    Gọi API OCR bất đồng bộ nội bộ (3 bước upload / tạo phiên / lấy kết quả, cộng huỷ phiên).

    Khoá lấy từ biến môi trường (xem docstring đầu file) và chỉ nằm trong header, không bao giờ ghi ra log.
    Lỗi 408/429/5xx và lỗi mạng được thử lại có chờ tăng dần; 401/403 ném FatalApiError (sai khoá hay hết sản lượng: chạy tiếp cũng vô ích).
    """

    def __init__(self, domain: str, access_token: str, token_id: str, token_key: str, mac_address: str = "mac-address",
                 poll_interval: float = 5.0, max_retries: int = 5):
        """Tạo client; access_token tự thêm tiền tố "Bearer " nếu chưa có (đúng theo tài liệu API)."""
        self.domain = domain.rstrip("/")
        token = access_token.strip()
        self.headers = {"Authorization": token if token.lower().startswith("bearer ") else f"Bearer {token}",
                        "Token-id": token_id.strip(), "Token-key": token_key.strip(), "mac-address": mac_address}
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self._local = threading.local()  # mỗi luồng một requests.Session (Session không đảm bảo an toàn đa luồng)
        self.active: set[str] = set()  # phiên đang chạy, để huỷ khi bị ngắt
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, **kwargs) -> "OcrClient":
        """Tạo client từ .env / biến môi trường; thiếu khoá thì báo rõ tên biến cần đặt (không in giá trị)."""
        load_dotenv(REPO_ROOT / ".env")

        def get(*names: str, default: str = "") -> str:
            """Giá trị của biến môi trường đầu tiên có đặt (tên đầy đủ VNPT_OCR_* trước, tên ngắn TOKEN_ID / TOKEN_KEY / ACCESS_TOKEN sau)."""
            return next((os.environ[n].strip() for n in names if os.environ.get(n, "").strip()), default)

        access = get("VNPT_OCR_ACCESS_TOKEN", "ACCESS_TOKEN")
        token_id = get("VNPT_OCR_TOKEN_ID", "TOKEN_ID")
        token_key = get("VNPT_OCR_TOKEN_KEY", "TOKEN_KEY")
        missing = [n for n, v in (("ACCESS_TOKEN", access), ("TOKEN_ID", token_id), ("TOKEN_KEY", token_key)) if not v]
        if missing:
            raise SystemExit("Thiếu khoá OCR nội bộ, hãy đặt trong .env hoặc biến môi trường: " + ", ".join(missing) + " (hoặc tên đầy đủ VNPT_OCR_<tên>)")
        return cls(get("VNPT_OCR_DOMAIN", "OCR_DOMAIN", default=DEFAULT_DOMAIN), access, token_id, token_key,
                   get("VNPT_OCR_MAC_ADDRESS", default="mac-address"), **kwargs)

    def _session(self) -> requests.Session:
        """requests.Session của luồng hiện tại (tạo khi cần)."""
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s

    def _call(self, endpoint: str, *, json_body: dict | None = None, files: dict | None = None, data: dict | None = None) -> dict:
        """
        POST một endpoint, trả về JSON phản hồi khi HTTP 200 và message = IDG-00000000.

        Upload dùng multipart (không đặt Content-Type để requests tự thêm boundary), hai bước còn lại gửi JSON. Thử lại lỗi tạm thời;
        ném FatalApiError với 401/403 và ApiError với mọi lỗi còn lại (kèm mã lỗi + message của API, không kèm header).
        """
        url = self.domain + endpoint
        last = ""
        for attempt in range(self.max_retries):
            try:
                resp = self._session().post(url, headers=self.headers, json=json_body, files=files, data=data, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                last = f"lỗi mạng: {type(exc).__name__}"
            else:
                if resp.status_code in FATAL_STATUS:
                    raise FatalApiError(f"{endpoint}: HTTP {resp.status_code} {_short(resp.text)} (kiểm tra khoá / sản lượng)")
                if resp.status_code in RETRY_STATUS:
                    last = f"HTTP {resp.status_code}"
                elif resp.status_code != 200:
                    raise ApiError(f"{endpoint}: HTTP {resp.status_code} {_short(resp.text)}")
                else:
                    try:
                        body = resp.json()
                    except ValueError as exc:
                        raise ApiError(f"{endpoint}: phản hồi không phải JSON") from exc
                    if body.get("message") != SUCCESS_MESSAGE:
                        raise ApiError(f"{endpoint}: message={body.get('message')!r} status={body.get('status')!r}")
                    return body
            if files is not None:  # luồng file đã đọc hết ở lần gửi trước, đưa về đầu để gửi lại
                for f in files.values():
                    f[1].seek(0)
            time.sleep(min(2 ** attempt * 2, 60))
        raise ApiError(f"{endpoint}: hết số lần thử ({last})")

    def upload(self, path: Path) -> str:
        """Upload file, trả về `hash` dùng cho bước tạo phiên."""
        with open(path, "rb") as f:
            body = self._call(EP_UPLOAD, files={"file": (path.name, f, "application/pdf")}, data={"title": path.stem, "description": path.stem})
        file_hash = (body.get("object") or {}).get("hash")
        if not file_hash:
            raise ApiError("addFile không trả hash")
        return file_hash

    def submit(self, file_hash: str, token: str, exporter: str, details: bool) -> str:
        """Tạo phiên OCR cho file đã upload, trả về session_id."""
        payload = {"file_hash": file_hash, "file_type": "pdf", "token": token, "client_session": f"00-14-22-01-23-45-{int(time.time() * 1000)}",
                   "details": details, "exporter": exporter}
        body = self._call(EP_SUBMIT, json_body=payload)
        session_id = (body.get("object") or {}).get("session_id")
        if not session_id:
            raise ApiError("gen-ai không trả session_id")
        return session_id

    def result(self, session_id: str) -> dict:
        """Hỏi kết quả một phiên; trả về `object` của phản hồi (còn xử lý thì có cảnh báo PENDING_WARNING và thiếu link)."""
        return self._call(EP_RESULT, json_body={"session_id": session_id}).get("object") or {}

    def cancel(self, session_id: str) -> None:
        """Huỷ phiên đang chạy; lỗi khi huỷ chỉ ghi log (đang dọn dẹp, không được làm hỏng luồng chính)."""
        try:
            self._call(EP_CANCEL, json_body={"session_id": session_id})
        except (ApiError, FatalApiError) as exc:
            logger.warning("huỷ phiên %s không được: %s", session_id, exc)

    def run_file(self, path: Path, token: str, exporter: str, details: bool, timeout: float) -> dict:
        """
        OCR một file từ đầu đến cuối, trả về {"session_id", "object", "link_payload", "elapsed_s"}.

        `object` là object của lần hỏi cuối (có link hoặc có kết quả trả thẳng); `link_payload` là nội dung tải từ link (nếu có).
        Quá `timeout` giây thì huỷ phiên rồi ném ApiError. Link kết quả là URL ký sẵn trên kho lưu trữ: tải KHÔNG kèm header khoá.
        """
        t0 = time.time()
        file_hash = self.upload(path)
        session_id = self.submit(file_hash, token, exporter, details)
        with self._lock:
            self.active.add(session_id)
        try:
            while True:
                obj = self.result(session_id)
                if obj.get("link") or obj.get("paragraphs") is not None:
                    break
                if time.time() - t0 > timeout:
                    self.cancel(session_id)
                    raise ApiError(f"quá hạn {timeout:.0f}s (đã xử lý {obj.get('num_of_processed_page')}/{obj.get('total_page_num')} trang)")
                time.sleep(self.poll_interval)
            payload = None
            if obj.get("link"):
                r = requests.get(obj["link"], timeout=REQUEST_TIMEOUT)
                if r.status_code != 200:
                    raise ApiError(f"tải link kết quả lỗi HTTP {r.status_code}")
                payload = r.text
        finally:
            with self._lock:
                self.active.discard(session_id)
        return {"session_id": session_id, "object": obj, "link_payload": payload, "elapsed_s": round(time.time() - t0, 1)}

    def cancel_all(self) -> None:
        """Huỷ mọi phiên đang chạy (gọi khi bị ngắt bằng Ctrl+C / SIGTERM)."""
        with self._lock:
            sessions = list(self.active)
        for s in sessions:
            self.cancel(s)


def _short(text: str, n: int = 200) -> str:
    """Cắt chuỗi phản hồi lỗi để ghi log gọn (và không kéo theo nội dung dài)."""
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "..."


# --------------------------------------------------------------------------------------------
# Đọc kết quả OCR
# --------------------------------------------------------------------------------------------

def parse_result(run: dict, exporter: str) -> tuple[str, dict[str, str], str]:
    """
    Đổi kết quả thô của API thành (text cả file, {số trang: text}, nội dung thô để lưu).

    - exporter "json": tìm `paragraphs` (trong nội dung tải từ link, hoặc trả thẳng trong `object`); mỗi cell có `text` và `page_id`,
      gộp theo trang, giữ thứ tự API trả; ô bảng (type Table) cũng lấy `text` của nó.
    - exporter "md": nội dung tải từ link là markdown của cả file, không tách được theo trang (pages rỗng).
    Cấu trúc kết quả lấy theo bộ Postman mẫu (server 1.2.32); nếu server mới đổi cấu trúc thì hàm rơi về nhặt mọi trường `text`
    và nội dung thô vẫn được lưu để đọc lại, nên không mất dữ liệu.
    """
    payload = run["link_payload"]
    if exporter == "md":
        text = payload if payload is not None else ""
        return text, {}, text
    data = json.loads(payload) if payload else run["object"]
    raw = json.dumps(data, ensure_ascii=False)
    root = data.get("object", data) if isinstance(data, dict) else {}
    paragraphs = root.get("paragraphs") if isinstance(root, dict) else None
    pages: dict[str, list[str]] = {}
    if isinstance(paragraphs, list):
        for block in paragraphs:
            for cell in (block.get("cells") or [block]) if isinstance(block, dict) else []:
                if isinstance(cell, dict) and cell.get("text"):
                    pages.setdefault(str(cell.get("page_id", 0)), []).append(cell["text"])
    else:  # cấu trúc lạ: nhặt mọi trường "text" theo thứ tự xuất hiện, không biết trang
        pages["0"] = [t for t in _walk_text(data)]
    page_text = {k: "\n".join(v) for k, v in sorted(pages.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)}
    return "\n\n".join(page_text.values()), page_text, raw


def _walk_text(node) -> list[str]:
    """Nhặt mọi giá trị chuỗi của khoá "text" trong cây JSON (đường lùi khi không có `paragraphs`)."""
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "text" and isinstance(v, str) and v:
                out.append(v)
            else:
                out.extend(_walk_text(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(_walk_text(v))
    return out


# --------------------------------------------------------------------------------------------
# Chạy một file, báo cáo
# --------------------------------------------------------------------------------------------

def write_atomic(path: Path, content: str) -> None:
    """Ghi file qua file tạm rồi đổi tên, để bị ngắt giữa chừng không để lại file dở."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def process_entry(entry: dict, client: OcrClient, out_dir: Path, exporter: str, details: bool, timeout: float) -> dict:
    """
    OCR một mục mẫu và ghi results/<sha256>.json (ok hoặc error). Mục đã có kết quả ok thì bỏ qua (resume).

    FatalApiError được ném tiếp lên để dừng cả lần chạy; ApiError của riêng file được ghi vào kết quả (status "error") để lần sau chạy lại.
    """
    res_path = out_dir / "results" / f"{entry['sha256']}.json"
    if res_path.exists():
        prev = json.loads(res_path.read_text(encoding="utf-8"))
        if prev.get("status") == "ok":
            return prev
    rec = {**entry, "exporter": exporter, "status": "error"}
    try:
        run = client.run_file(Path(entry["path"]), token=f"vi-corpus-{entry['sha256'][:12]}", exporter=exporter, details=details, timeout=timeout)
        text, pages, raw = parse_result(run, exporter)
        write_atomic(out_dir / "raw" / f"{entry['sha256']}.{'md' if exporter == 'md' else 'json'}", raw)
        obj = run["object"]
        rec.update(status="ok", session_id=run["session_id"], elapsed_s=run["elapsed_s"], billed_pages=obj.get("num_of_pages"),
                   total_pages=obj.get("total_page_num") or entry["pages"], warnings=obj.get("warnings") or obj.get("warning") or [],
                   ocr_chars=len(text), ocr_pages_with_text=len(pages), text=text, ocr_pages=pages)
    except FatalApiError:
        raise
    except (ApiError, ValueError) as exc:  # ValueError: JSON kết quả hỏng
        rec["error"] = str(exc)
    write_atomic(res_path, json.dumps(rec, ensure_ascii=False))
    return rec


def text_layer_of(path: str) -> str:
    """Toàn bộ lớp chữ có sẵn của PDF (PyMuPDF), dùng làm mốc so sánh cho file 2 lớp."""
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def file_quality(rec: dict, vocab: frozenset[str] | None) -> dict:
    """
    Đo chất lượng chữ OCR của một file đã xong và liệt kê các dấu hiệu lỗi (flags).

    Các số đo (dùng lại bộ đo của pipeline, vi_corpus.common.garbled): tỉ lệ trang có chữ trên tổng số trang (`page_coverage`, chỉ khi exporter json),
    tỉ lệ âm tiết lạ so với tập âm tiết tiếng Việt (`unknown_rate`, bỏ qua khi thiếu syllables.json), tỉ lệ từ mang dấu sai kiểu font hỏng (`garbled`),
    tỉ lệ ký tự điều khiển (`control`), tỉ lệ dòng lặp (`dup_lines`, dấu hiệu máy OCR bị lặp vòng) và, với file 2 lớp, độ trùng từ với lớp chữ có sẵn.
    Ngưỡng ở đầu file (QUALITY_LIMITS) là ngưỡng thử, chưa chỉnh trên dữ liệu thật; nhìn số đo chứ đừng chỉ nhìn cờ.
    """
    text = rec["text"]
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 15]
    q = {"page_coverage": round(rec["ocr_pages_with_text"] / rec["pages"], 3) if rec["exporter"] == "json" and rec["pages"] else None,
         "unknown_rate": round(unknown_syllable_rate(text, vocab)[0], 3) if vocab else None,
         "garbled": round(garbled_ratio(text)[0], 3), "control": round(control_char_ratio(text)[0], 4),
         "dup_lines": round(1 - len(set(lines)) / len(lines), 3) if lines else 0.0,
         "overlap_vs_text_layer": round(word_overlap(text, text_layer_of(rec["path"])), 3) if rec["layer"] == 2 else None}
    lim = QUALITY_LIMITS
    flags = []
    if rec["ocr_chars"] < lim["min_chars_per_page"] * rec["pages"]:
        flags.append("ít_chữ")
    if q["page_coverage"] is not None and q["page_coverage"] < lim["min_page_coverage"]:
        flags.append("thiếu_trang")
    if q["unknown_rate"] is not None and q["unknown_rate"] > lim["max_unknown_rate"]:
        flags.append("âm_tiết_lạ")
    if q["garbled"] > lim["max_garbled"]:
        flags.append("dấu_sai")
    if q["control"] > lim["max_control"]:
        flags.append("ký_tự_điều_khiển")
    if q["dup_lines"] > lim["max_dup_lines"]:
        flags.append("dòng_lặp")
    if q["overlap_vs_text_layer"] is not None and q["overlap_vs_text_layer"] < lim["min_overlap"]:
        flags.append("lệch_lớp_chữ")
    return {**q, "flags": flags}


def build_report(out_dir: Path, sample: list[dict], vocab: frozenset[str] | None = None) -> dict:
    """
    Gom kết quả thành report.csv (một dòng một file), report.json (số liệu + kết luận) và report.md (đọc nhanh, có danh sách lỗi để gửi lại bên cung cấp OCR).

    Kết luận (`verdict`) là gợi ý theo QUALITY_LIMITS: "OK" khi tỉ lệ file lỗi API <= max_error_rate và tỉ lệ file ok nhưng bị cờ chất lượng <= max_flagged_rate;
    ngược lại "KHÔNG ĐẠT" kèm lý do. File chưa chạy (not_run) không được tính vào kết luận; kết luận chỉ có nghĩa khi mọi file đã chạy.
    Cờ "lệch_lớp_chữ" chỉ là gợi ý: lớp chữ cũ của PDF 2 lớp có thể chính là bên sai, xem tay vài file trước khi kết luận OCR sai.
    """
    rows = []
    for e in sample:
        p = out_dir / "results" / f"{e['sha256']}.json"
        r = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {**e, "status": "not_run"}
        q = file_quality(r, vocab) if r["status"] == "ok" else {"flags": []}
        rows.append({"source": r["source"], "layer": r["layer"], "layer2_kind": r.get("layer2_kind") or "", "rel": r["rel"], "pages": r["pages"],
                     "size_mb": r["size_mb"], "status": r["status"], "elapsed_s": r.get("elapsed_s", ""), "billed_pages": r.get("billed_pages", ""),
                     "session_id": r.get("session_id", ""), "ocr_chars": r.get("ocr_chars", ""), "page_coverage": q.get("page_coverage"),
                     "unknown_rate": q.get("unknown_rate"), "garbled": q.get("garbled"), "control": q.get("control"), "dup_lines": q.get("dup_lines"),
                     "overlap_vs_text_layer": q.get("overlap_vs_text_layer"), "flags": " ".join(q["flags"]), "error": r.get("error", "")})
    with open(out_dir / "report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["source"])
        w.writeheader()
        w.writerows(rows)
    groups: dict[str, dict] = {}
    for r in rows:
        g = groups.setdefault(f"{r['source']}/layer{r['layer']}", {"files": 0, "ok": 0, "error": 0, "not_run": 0, "flagged": 0, "pages": 0, "seconds": 0.0,
                                                                   "chars": 0, "overlaps": []})
        g["files"] += 1
        g[r["status"]] += 1
        if r["status"] == "ok":
            g["flagged"] += bool(r["flags"])
            g["pages"] += r["pages"]
            g["seconds"] += r["elapsed_s"]
            g["chars"] += r["ocr_chars"]
            if r["overlap_vs_text_layer"] is not None:
                g["overlaps"].append(r["overlap_vs_text_layer"])
    for g in groups.values():
        ov = g.pop("overlaps")
        g["seconds_per_page"] = round(g["seconds"] / g["pages"], 2) if g["pages"] else None
        g["chars_per_page"] = round(g["chars"] / g["pages"]) if g["pages"] else None
        g["mean_overlap_vs_text_layer"] = round(sum(ov) / len(ov), 3) if ov else None
    ran = [r for r in rows if r["status"] != "not_run"]
    errors = [r for r in rows if r["status"] == "error"]
    flagged = [r for r in rows if r["status"] == "ok" and r["flags"]]
    reasons = []
    if ran and len(errors) / len(ran) > QUALITY_LIMITS["max_error_rate"]:
        reasons.append(f"{len(errors)}/{len(ran)} file lỗi API (ngưỡng {QUALITY_LIMITS['max_error_rate']:.0%})")
    if ran and len(flagged) / len(ran) > QUALITY_LIMITS["max_flagged_rate"]:
        reasons.append(f"{len(flagged)}/{len(ran)} file ok nhưng bị cờ chất lượng (ngưỡng {QUALITY_LIMITS['max_flagged_rate']:.0%})")
    verdict = "CHƯA CHẠY" if not ran else ("KHÔNG ĐẠT" if reasons else "OK")
    summary = {"verdict": verdict, "reasons": reasons, "files_run": len(ran), "files_total": len(rows), "errors": len(errors), "flagged": len(flagged),
               "limits": QUALITY_LIMITS, "groups": groups, "total_billed_pages": sum(r["billed_pages"] for r in rows if isinstance(r["billed_pages"], int))}
    write_atomic(out_dir / "report.json", json.dumps(summary, ensure_ascii=False, indent=1))
    write_atomic(out_dir / "report.md", render_markdown(summary, errors, flagged))
    return summary


def render_markdown(summary: dict, errors: list[dict], flagged: list[dict]) -> str:
    """Dựng report.md: kết luận, bảng theo nhóm nguồn x loại, danh sách file lỗi API (kèm session_id để bên VNPT tra log) và file bị cờ chất lượng."""
    L = [f"# Báo cáo chạy thử OCR nội bộ\n", f"**Kết luận: {summary['verdict']}**  ", f"Đã chạy {summary['files_run']}/{summary['files_total']} file, "
         f"{summary['errors']} lỗi API, {summary['flagged']} file bị cờ chất lượng, {summary['total_billed_pages']} trang tính sản lượng.\n"]
    L += [f"- {r}" for r in summary["reasons"]]
    L += ["", "## Theo nhóm", "", "| nhóm | file | ok | lỗi | cờ | trang | giây/trang | ký tự/trang | trùng lớp chữ |", "|---|---|---|---|---|---|---|---|---|"]
    for k, g in summary["groups"].items():
        L.append(f"| {k} | {g['files']} | {g['ok']} | {g['error']} | {g['flagged']} | {g['pages']} | {g['seconds_per_page']} | {g['chars_per_page']} | "
                 f"{g['mean_overlap_vs_text_layer']} |")
    L += ["", "## File lỗi API (gửi bên cung cấp OCR kèm session_id)", ""]
    L += [f"- `{r['source']}/{r['rel']}` ({r['pages']} trang, {r['size_mb']} MB): {r['error']}" for r in errors] or ["- không có"]
    L += ["", "## File ok nhưng bị cờ chất lượng", ""]
    L += [f"- `{r['source']}/{r['rel']}` layer{r['layer']} [{r['flags']}] session `{r['session_id']}`: trang có chữ {r['page_coverage']}, âm tiết lạ {r['unknown_rate']}, "
          f"dấu sai {r['garbled']}, dòng lặp {r['dup_lines']}, trùng lớp chữ {r['overlap_vs_text_layer']}" for r in flagged] or ["- không có"]
    L += ["", "Ngưỡng cờ là ngưỡng thử chưa chỉnh trên dữ liệu thật (xem QUALITY_LIMITS trong script). Chữ OCR đầy đủ ở `results/<sha256>.json`, kết quả thô ở `raw/`.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------------------------
# Chương trình chính
# --------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh (xem docstring đầu file)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", default=os.environ.get("SEA_DATA_ROOT", "./data"), help="gốc dữ liệu (có raw/vjol, raw/vista)")
    ap.add_argument("--per-source", type=int, default=30, help="số PDF mỗi nguồn, chia đều 1 lớp / 2 lớp")
    ap.add_argument("--seed", type=int, default=42, help="seed chọn ngẫu nhiên (cùng seed cùng mẫu)")
    ap.add_argument("--run-name", default="trial", help="tên thư mục kết quả trong processed/ocr_internal_trial/")
    ap.add_argument("--max-pages", type=int, default=60, help="bỏ PDF dài hơn số trang này (tiết kiệm sản lượng)")
    ap.add_argument("--max-mb", type=float, default=30.0, help="bỏ PDF nặng hơn số MB này")
    ap.add_argument("--max-scan", type=int, default=3000, help="số file tối đa dò mỗi nguồn khi chọn mẫu")
    ap.add_argument("--workers", type=int, default=3, help="số file OCR song song (API báo 408 'hệ thống bận' thì giảm)")
    ap.add_argument("--exporter", choices=["json", "md"], default="json", help="định dạng kết quả của API; json có số trang, md thì không")
    ap.add_argument("--no-details", action="store_true", help="gửi details=false (mặc định true như ví dụ trong tài liệu)")
    ap.add_argument("--timeout", type=float, default=900.0, help="giây tối đa chờ một file, quá thì huỷ phiên")
    ap.add_argument("--poll-interval", type=float, default=5.0, help="giây giữa hai lần hỏi kết quả")
    ap.add_argument("--limit-files", type=int, default=0, help="chỉ chạy N file đầu của mẫu (xen kẽ nguồn / loại); 0 = hết")
    ap.add_argument("--syllables", default="", help="tập âm tiết tiếng Việt (mặc định <data-root>/processed/ocr_bakeoff/syllables.json)")
    ap.add_argument("--dry-run", action="store_true", help="chỉ chọn mẫu và in danh sách, không gọi API")
    ap.add_argument("--report-only", action="store_true", help="không gọi API, chỉ dựng lại báo cáo từ kết quả đã có")
    ap.add_argument("--reselect", action="store_true", help="chọn lại mẫu dù đã có sample.json (mặc định dùng lại để resume đúng mẫu)")
    return ap.parse_args(argv)


def load_or_pick_sample(args: argparse.Namespace, root: Path, out_dir: Path) -> list[dict]:
    """Dùng lại sample.json nếu có (giữ đúng mẫu giữa các lần chạy), không thì chọn mới và ghi lại cùng thống kê thiếu/đủ từng loại."""
    sample_path = out_dir / "sample.json"
    if sample_path.exists() and not args.reselect:
        return json.loads(sample_path.read_text(encoding="utf-8"))["sample"]
    sample, stats = [], {}
    for source in SOURCES:
        raw_dir = root / "raw" / source
        if not raw_dir.is_dir():
            raise SystemExit(f"Không thấy {raw_dir}; cần có raw/vjol và raw/vista trong --data-root")
        s, st = pick_source_sample(source, raw_dir, args.per_source, args.seed, args.max_pages, args.max_mb, args.max_scan)
        sample += s
        stats[source] = st
        logger.info("%s: dò %d/%d file -> 1 lớp %d/%d, 2 lớp %d/%d (image_text %d, text_only %d)", source, st["scanned"], st["files_in_raw"],
                    st["got_layer1"], st["want_layer1"], st["got_layer2"], st["want_layer2"], st["layer2_image_text"], st["layer2_text_only"])
        for kind in ("layer1", "layer2"):
            if st[f"got_{kind}"] < st[f"want_{kind}"]:
                logger.warning("%s: THIẾU file %s (%d/%d) - tỷ lệ 1 lớp / 2 lớp không còn đều", source, kind, st[f"got_{kind}"], st[f"want_{kind}"])
    write_atomic(sample_path, json.dumps({"seed": args.seed, "stats": stats, "sample": interleave(sample)}, ensure_ascii=False, indent=1))
    return interleave(sample)


def main(argv: list[str] | None = None) -> int:
    """Chọn mẫu, (tuỳ chọn) chạy OCR nội bộ, dựng báo cáo. Trả mã thoát: 0 ổn, 1 có file lỗi, 2 sai khoá / hết sản lượng."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    root = Path(args.data_root).resolve()
    out_dir = root / "processed" / "ocr_internal_trial" / args.run_name
    sample = load_or_pick_sample(args, root, out_dir)
    if args.limit_files:
        sample = sample[: args.limit_files]

    if args.dry_run:
        for e in sample:
            print(f"{e['source']:5} layer{e['layer']} {e['layer2_kind'] or '-':10} {e['pages']:3d} trang {e['size_mb']:6.2f} MB  {e['rel']}")
        print(f"Tổng {len(sample)} file, {sum(e['pages'] for e in sample)} trang (số trang sẽ tính vào sản lượng). Mẫu ghi ở {out_dir / 'sample.json'}")
        return 0

    failed = 0
    if not args.report_only:
        client = OcrClient.from_env(poll_interval=args.poll_interval)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        pool = ThreadPoolExecutor(max_workers=args.workers)
        futures = {pool.submit(process_entry, e, client, out_dir, args.exporter, not args.no_details, args.timeout): e for e in sample}
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                e = futures[fut]
                rec = fut.result()
                failed += rec["status"] != "ok"
                logger.info("[%d/%d] %s/%s -> %s %s", i, len(futures), e["source"], e["rel"], rec["status"],
                            f"{rec.get('elapsed_s')}s {rec.get('ocr_chars')} ký tự" if rec["status"] == "ok" else rec.get("error", ""))
        except FatalApiError as exc:
            logger.error("Dừng cả lần chạy: %s", exc)
            pool.shutdown(wait=False, cancel_futures=True)
            client.cancel_all()
            return 2
        except KeyboardInterrupt:
            logger.error("Bị ngắt, huỷ các phiên đang chạy...")
            pool.shutdown(wait=False, cancel_futures=True)
            client.cancel_all()
            return 130
        pool.shutdown()

    vocab_path = Path(args.syllables) if args.syllables else root / "processed" / "ocr_bakeoff" / "syllables.json"
    vocab = load_vocab(vocab_path)
    if vocab is None:
        logger.warning("Không có %s: bỏ số đo âm tiết lạ (unknown_rate); các số đo khác vẫn có", vocab_path)
    summary = build_report(out_dir, sample, vocab)
    print((out_dir / "report.md").read_text(encoding="utf-8"))
    print(f"Báo cáo: {out_dir / 'report.md'}, report.csv, report.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
