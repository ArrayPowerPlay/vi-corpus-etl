"""
Chạy một engine OCR trên bộ trang của đợt so sánh (F-11), lưu đầu ra thô theo trang để chấm lại không cần chạy lại.

Cấu hình engine ở configs/ocr_bakeoff/engines.json, mỗi engine một mục:
    type        : "baseline" (Paddle detect + VietOCR, vi_corpus.common.ocr.PageOcr, chạy tuần tự),
                  "openai"   (mô hình thị giác - ngôn ngữ phục vụ bằng vLLM, API tương thích OpenAI, gửi song song),
                  "paddleocr_vl" (pipeline PaddleOCRVL của paddleocr: dò bố cục + VLM nhận dạng từng vùng).
    output      : định dạng đầu ra cho adapter textnorm.to_plain ("plain" | "markdown" | "dots_layout_json").
    serve       : (tuỳ chọn) lệnh dựng server, dạng list; run_engine --serve tự chạy, chờ sẵn sàng, chạy xong thì tắt.
    base_url    : địa chỉ API ".../v1"; health: đường dẫn kiểm sẵn sàng (mặc định "/models").
    model, served_name, prompt, text_prefix, max_tokens, extra_body: tham số gọi API (engine "openai").
    pipeline_kwargs: tham số khởi tạo PaddleOCRVL (engine "paddleocr_vl").
    license, note: ghi chú, in vào báo cáo.
Mọi engine giải mã tham lam (temperature 0). Đầu ra: <bakeoff>/runs/<engine>/outputs.jsonl, mỗi trang một dòng
{"page_id", "raw", "text", "finish_reason", "prompt_tokens", "completion_tokens", "t_start", "t_end", "error",
"session"}; chạy lại thì bỏ qua trang đã có kết quả không lỗi (--redo để chạy lại từ đầu). run_info.json lưu cấu hình,
GPU, phiên bản và từng phiên chạy (bắt đầu, kết thúc, số giây, số giây chờ server tải mô hình). Tốc độ được tính ở bước chấm từ t_end của từng trang trong cùng một phiên (bỏ các trang khởi động).
"""

import base64
import json
import logging
import os
import platform
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vi_corpus.ocr_bakeoff.pages import read_pages
from vi_corpus.ocr_bakeoff.textnorm import to_plain

logger = logging.getLogger("vi_corpus")

DEFAULT_ENGINES = Path(__file__).resolve().parents[2] / "configs/ocr_bakeoff/engines.json"
_MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "jp2": "image/jp2", "webp": "image/webp"}


def load_engines(path: Path = DEFAULT_ENGINES) -> dict[str, dict]:
    """Đọc cấu hình engine (bỏ các khoá bắt đầu bằng "_", dùng làm chú thích trong JSON)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


class BaselineEngine:
    """Engine hiện tại: PaddleOCR detect + VietOCR (giống run_ocr của stbook)."""

    def __init__(self, cfg: dict) -> None:
        """Nạp mô hình (cần `uv sync --group ocr`)."""
        from vi_corpus.common.ocr import PageOcr

        self.ocr = PageOcr(device=cfg.get("device", "cuda"))

    def run(self, image_path: Path) -> dict:
        """OCR một ảnh trang."""
        from PIL import Image

        with Image.open(image_path) as im:
            return {"raw": self.ocr.page_text(im.convert("RGB")), "finish_reason": None}


class OpenAIEngine:
    """Mô hình thị giác - ngôn ngữ gọi qua API chat completions tương thích OpenAI (vLLM)."""

    def __init__(self, cfg: dict) -> None:
        """Lưu tham số gọi API."""
        import requests

        self.session = requests.Session()
        self.cfg = cfg
        self.url = cfg["base_url"].rstrip("/") + "/chat/completions"

    def payload(self, image_path: Path) -> dict:
        """Thân yêu cầu: ảnh (data URL base64, byte gốc) rồi lời nhắc; giải mã tham lam."""
        mime = _MIME.get(image_path.suffix.lower().lstrip("."), "image/png")
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                   {"type": "text", "text": self.cfg.get("text_prefix", "") + self.cfg["prompt"]}]
        return {"model": self.cfg.get("served_name") or self.cfg["model"],
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0, "top_p": 1.0, "max_tokens": self.cfg.get("max_tokens", 8192),
                **self.cfg.get("extra_body", {})}

    def run(self, image_path: Path) -> dict:
        """Gửi một trang, trả nội dung và lý do dừng (length = bị cắt vì hết max_tokens)."""
        r = self.session.post(self.url, json=self.payload(image_path), timeout=self.cfg.get("timeout", 900))
        r.raise_for_status()
        body = r.json()
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        return {"raw": choice["message"].get("content") or "", "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


class PaddleVLEngine:
    """Pipeline PaddleOCR-VL (dò bố cục PP-DocLayout + VLM 0.9B nhận dạng), xuất markdown."""

    def __init__(self, cfg: dict) -> None:
        """Khởi tạo pipeline (cần paddleocr[doc-parser] và paddlepaddle-gpu trên server)."""
        from paddleocr import PaddleOCRVL

        self.pipe = PaddleOCRVL(**cfg.get("pipeline_kwargs", {}))

    def run(self, image_path: Path) -> dict:
        """OCR một trang: lưu markdown / JSON của từng kết quả vào thư mục tạm rồi đọc lại."""
        md, js = [], []
        with tempfile.TemporaryDirectory() as tmp:
            for res in self.pipe.predict(str(image_path)):
                res.save_to_markdown(save_path=tmp)
                res.save_to_json(save_path=tmp)
            for f in sorted(Path(tmp).rglob("*.md")):
                md.append(f.read_text(encoding="utf-8"))
            for f in sorted(Path(tmp).rglob("*.json")):
                js.append(json.loads(f.read_text(encoding="utf-8")))
        return {"raw": "\n\n".join(md), "finish_reason": None, "layout": js}


ENGINE_TYPES = {"baseline": BaselineEngine, "openai": OpenAIEngine, "paddleocr_vl": PaddleVLEngine}


def _page_id(line: str) -> str | None:
    """page_id của một dòng outputs.jsonl (None nếu dòng hỏng)."""
    try:
        return json.loads(line)["page_id"]
    except (json.JSONDecodeError, KeyError):
        return None


def _done(out_path: Path) -> set[str]:
    """Các page_id đã có kết quả không lỗi trong outputs.jsonl."""
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # dòng cuối ghi dở khi bị kill
                continue
            if not rec.get("error"):
                done.add(rec["page_id"])
    return done


def pending_pages(bakeoff_dir: Path, name: str, only_set: str | None = None, limit: int | None = None) -> list[dict]:
    """Các trang engine `name` chưa có kết quả không lỗi (lọc theo bộ, cắt còn `limit` trang)."""
    done = _done(bakeoff_dir / "runs" / name / "outputs.jsonl")
    pages = [p for p in read_pages(bakeoff_dir) if (only_set is None or p["set"] == only_set)
             and p["page_id"] not in done]
    return pages[:limit]


def wait_ready(base_url: str, health: str = "/models", timeout: float = 3600, proc=None) -> None:
    """
    Chờ server trả 200 ở base_url + health (lần đầu có thể phải tải trọng số mô hình).

    Raises:
        RuntimeError: server tắt giữa chừng hoặc quá thời gian.
    """
    import requests

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Server đã tắt (mã {proc.returncode}), xem server.log")
        try:
            if requests.get(base_url.rstrip("/") + health, timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(f"Server {base_url} chưa sẵn sàng sau {timeout:.0f}s")


def start_server(cfg: dict, run_dir: Path, gpu: str | None, vllm_bin: str | None) -> subprocess.Popen:
    """Chạy lệnh cfg["serve"] (đổi "vllm" đầu lệnh thành vllm_bin nếu có), log ra run_dir/server.log."""
    cmd = list(cfg["serve"])
    if vllm_bin and cmd and cmd[0] == "vllm":
        cmd[0] = vllm_bin
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    log = open(run_dir / "server.log", "ab")  # noqa: SIM115 - đóng khi tiến trình server kết thúc (stop_server)
    logger.info("Dựng server: %s (CUDA_VISIBLE_DEVICES=%s)", " ".join(cmd), env.get("CUDA_VISIBLE_DEVICES"))
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def stop_server(proc: subprocess.Popen) -> None:
    """Tắt server (SIGTERM, sau 60 s thì SIGKILL)."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _gpu_info() -> str | None:
    """Tên GPU theo nvidia-smi (None nếu không có)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _update_session(info_path: Path, session: str, **fields) -> None:
    """Ghi thêm trường vào phiên `session` trong run_info.json."""
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for sess in info["sessions"]:
        if sess["session"] == session:
            sess.update(fields)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def run_engine(bakeoff_dir: Path, name: str, cfg: dict, concurrency: int = 1, limit: int | None = None,
               redo: bool = False, only_set: str | None = None, server_wait_s: float | None = None) -> dict:
    """
    Chạy engine `name` trên các trang chưa có kết quả, ghi thêm vào runs/<name>/outputs.jsonl.

    Args:
        concurrency: Số trang gửi đồng thời (chỉ engine "openai"; engine khác chạy tuần tự trong tiến trình này).
        limit:       Chỉ chạy chừng này trang (thử nhanh).
        redo:        Xoá kết quả cũ của các trang thuộc bộ đang chạy (only_set, hoặc mọi trang) rồi chạy lại.
        only_set:    "A" / "B": chỉ chạy một bộ.
        server_wait_s: Số giây chờ server tải mô hình (do run_engine.py đo), lưu vào run_info.json.

    Returns:
        {"engine", "session", "pages", "errors", "seconds"}.

    Raises:
        ValueError: type của engine lạ.
    """
    if cfg.get("type") not in ENGINE_TYPES:
        raise ValueError(f"Engine {name}: type {cfg.get('type')!r} không hỗ trợ ({', '.join(ENGINE_TYPES)})")
    run_dir = bakeoff_dir / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "outputs.jsonl"
    if redo and out_path.exists():  # chỉ xoá kết quả của các trang thuộc bộ đang chạy lại
        redo_ids = {p["page_id"] for p in read_pages(bakeoff_dir) if only_set is None or p["set"] == only_set}
        with open(out_path, encoding="utf-8") as f:
            keep = [ln for ln in f if ln.strip() and _page_id(ln) not in redo_ids]
        out_path.write_text("".join(keep), encoding="utf-8")
    todo = pending_pages(bakeoff_dir, name, only_set, limit)
    session = time.strftime("%Y%m%dT%H%M%S")
    logger.info("[%s] %d trang cần chạy", name, len(todo))
    if not todo:
        return {"engine": name, "session": session, "pages": 0, "errors": 0, "seconds": 0.0}
    info_path = run_dir / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {"sessions": []}
    info.update({"engine": name, "config": cfg, "gpu": _gpu_info(), "host": platform.node()})
    info["sessions"].append({"session": session, "pages": len(todo), "concurrency": concurrency,
                             "started": time.time(), "server_wait_s": server_wait_s})
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    engine = ENGINE_TYPES[cfg["type"]](cfg)
    lock = threading.Lock()
    errors = 0
    t_begin = time.time()

    def one(page: dict) -> dict:
        """Chạy một trang, bắt lỗi thành bản ghi có trường error."""
        t0 = time.time()
        rec = {"page_id": page["page_id"], "session": session, "t_start": t0}
        try:
            res = engine.run(bakeoff_dir / page["image"])
            rec.update(res)
            rec["text"] = to_plain(res["raw"], cfg.get("output", "plain"))
            rec["error"] = None
        except Exception as e:  # noqa: BLE001 - một trang lỗi (timeout, OOM...) không dừng cả lượt; chạy lại sẽ làm lại
            rec.update({"raw": None, "text": None, "error": f"{type(e).__name__}: {e}"})
        rec["t_end"] = time.time()
        return rec

    with open(out_path, "a", encoding="utf-8") as f:

        def write(rec: dict) -> None:
            """Ghi một dòng kết quả (khoá vì nhiều luồng)."""
            nonlocal errors
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                errors += bool(rec["error"])

        workers = concurrency if cfg["type"] == "openai" else 1
        if workers == 1:
            for i, page in enumerate(todo, 1):
                write(one(page))
                if i % 20 == 0:
                    logger.info("[%s] %d/%d trang (%.2f trang/s)", name, i, len(todo), i / (time.time() - t_begin))
        else:
            with ThreadPoolExecutor(workers) as pool:
                futures = [pool.submit(one, p) for p in todo]
                for i, fut in enumerate(as_completed(futures), 1):
                    write(fut.result())
                    if i % 20 == 0:
                        logger.info("[%s] %d/%d trang (%.2f trang/s)", name, i, len(todo),
                                    i / (time.time() - t_begin))
    seconds = time.time() - t_begin
    _update_session(info_path, session, ended=time.time(), seconds=round(seconds, 1), errors=errors)
    logger.info("[%s] xong %d trang, %d lỗi, %.1fs", name, len(todo), errors, seconds)
    return {"engine": name, "session": session, "pages": len(todo), "errors": errors, "seconds": round(seconds, 1)}
