"""
Chuỗi engine OCR (O-20): engine chính chạy trước, trang bị bộ kiểm (guard.py) báo lỗi thì chạy lại bằng engine kế tiếp.

Chuỗi đã chốt: `qwen3vl_8b_mt4k` (chính) -> `qwen3vl_8b_rp110` (chạy lại, phạt lặp mạnh) -> `baseline` (Paddle detect +
VietOCR, dự phòng). Engine cuối nhận trang bất kể kết quả; nếu nó vẫn bị báo thì trang được đánh dấu "flagged" (loại hoặc
kiểm tay, không đưa vào kho). Một quy tắc chọn duy nhất (pick_tier) dùng cho cả ba nơi:
    * chạy thật  : CascadeOcr (bên dưới) gọi engine theo từng trang, chỉ gọi engine kế tiếp khi cần (lười);
    * mô phỏng   : merge_cascade ghép đầu ra có sẵn của các engine trong đợt so sánh (eval_guard.py --cascade, score.py --cascade);
    * giờ chạy   : cascade_hours ước tính số giờ cả kho từ tốc độ từng engine và tỷ lệ trang phải chạy lại.
Các engine VLM trong chuỗi chỉ khác nhau ở tham số lấy mẫu (extra_body) nên dùng chung MỘT server vLLM; mỗi trang gửi tới server
với tham số của engine tương ứng (xem build_tiers). Engine baseline chỉ nạp khi có trang cần tới nó.
"""

import copy
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from vi_corpus.ocr_bakeoff.engines import ENGINE_TYPES
from vi_corpus.ocr_bakeoff.guard import guard_page
from vi_corpus.ocr_bakeoff.textnorm import to_plain

logger = logging.getLogger("vi_corpus")

DEFAULT_TIERS = ("qwen3vl_8b_mt4k", "qwen3vl_8b_rp110", "baseline")  # chuỗi chốt ở O-20
RETRY_STATUS = (408, 429, 500, 502, 503, 504)  # mã HTTP đáng thử lại; 4xx khác (vd 400 ảnh hỏng) thì không


class ServerDown(RuntimeError):
    """Server VLM không trả lời liên tiếp quá nhiều lần: dừng cả lượt thay vì lặng lẽ đẩy mọi trang xuống engine dự phòng."""


def pick_tier(tiers: list[str], get_rec: Callable[[str], dict | None],
              max_tokens: dict[str, int | None]) -> tuple[str, dict | None, list[str]]:
    """
    Chọn đầu ra cho một trang: của engine đầu tiên không bị bộ kiểm báo; engine cuối nhận bất kể.

    Args:
        tiers:      Tên engine theo thứ tự chạy.
        get_rec:    Hàm tên engine -> bản ghi trang ({"raw", "text", "finish_reason", ...}) hoặc None nếu engine không có kết
                    quả cho trang (chưa chạy / lỗi). Gọi lười: chỉ gọi engine kế tiếp khi engine trước bị báo (chạy thật
                    nhờ vậy không tốn lượt gọi engine sau).
        max_tokens: max_tokens của từng engine, để bộ kiểm phát hiện chạm trần token.

    Returns:
        (tên engine được chọn, bản ghi, mã lý do bộ kiểm báo trên bản ghi đó). Không engine nào có kết quả: (engine cuối, None, []).
    """
    for name in tiers:
        rec = get_rec(name)
        if rec is None:
            continue
        reasons = guard_page(rec, max_tokens.get(name))
        if not reasons or name == tiers[-1]:
            return name, rec, reasons
    return tiers[-1], None, []


def merge_cascade(runs: dict[str, dict[str, dict]], tiers: list[str], max_tokens: dict[str, int | None],
                  page_ids: list[str]) -> tuple[dict[str, dict], Counter]:
    """
    Ghép đầu ra có sẵn của các engine thành một "engine chuỗi" (mô phỏng, dùng lại đầu ra của đợt so sánh nên là cận gần đúng).

    Args:
        runs:       {engine: {page_id: bản ghi}} (perplexity.latest_outputs).
        tiers:      Thứ tự chuỗi.
        max_tokens: max_tokens từng engine (xem pick_tier).
        page_ids:   Các trang cần ghép.

    Returns:
        ({page_id: bản ghi đã chọn thêm "tier" và "guard" (mã lý do còn lại)}, số trang lấy từ mỗi engine). Trang không engine
        nào có kết quả thì vắng mặt trong kết quả.
    """
    merged: dict[str, dict] = {}
    used: Counter = Counter({t: 0 for t in tiers})
    for pid in page_ids:
        name, rec, reasons = pick_tier(tiers, lambda t, pid=pid: runs[t].get(pid), max_tokens)
        if rec is not None:
            merged[pid] = {**rec, "tier": name, "guard": reasons}
            used[name] += 1
    return merged, used


def cascade_hours(hours: dict[str, dict], tiers: list[str], used: Counter, num_gpus: int) -> dict:
    """
    Số giờ chạy cả kho của chuỗi: engine thứ i chỉ chạy trên phần trang đã bị các engine trước báo lỗi.

    Số giờ = tổng_i (tỷ lệ trang tới được engine i) x (số giờ cả kho nếu engine i chạy mọi trang). Tỷ lệ trang tới được engine i
    = số trang cuối cùng lấy từ engine i trở đi / tổng số trang (đo trên bộ trang của đợt so sánh, nên là ước tính thô).

    Args:
        hours:    {engine: run_hours()} của từng engine trong chuỗi.
        used:     Số trang lấy từ mỗi engine (merge_cascade).
        num_gpus: Số GPU quy ra giờ GPU.

    Returns:
        Cùng khoá với scoring.run_hours; "pages_per_s" là tốc độ hiệu dụng của cả chuỗi trên 1 GPU, "reach" là tỷ lệ trang tới
        từng engine. "corpus_hours" là None nếu có engine chưa đo được tốc độ.
    """
    total = max(sum(used.values()), 1)
    reach = {t: sum(used[u] for u in tiers[i:]) / total for i, t in enumerate(tiers)}
    per_tier = [hours[t]["corpus_hours"] for t in tiers]
    ok = all(h is not None for h in per_tier)
    corpus_hours = sum(reach[t] * h for t, h in zip(tiers, per_tier)) if ok else None
    corpus_pages = hours[tiers[0]]["corpus_pages"]
    by_source = {}
    for src in hours[tiers[0]]["corpus_hours_by_source"]:
        vals = [hours[t]["corpus_hours_by_source"][src] for t in tiers]
        by_source[src] = sum(reach[t] * v for t, v in zip(tiers, vals)) if all(v is not None for v in vals) else None
    return {"bakeoff_pages": sum(hours[t]["bakeoff_pages"] for t in tiers),
            "bakeoff_run_hours": sum(hours[t]["bakeoff_run_hours"] for t in tiers),
            "bakeoff_startup_hours": sum(hours[t]["bakeoff_startup_hours"] for t in tiers),
            "pages_per_s": corpus_pages / (corpus_hours * 3600 * num_gpus) if corpus_hours else None,
            "steady": all(hours[t]["steady"] for t in tiers), "corpus_pages": corpus_pages, "num_gpus": num_gpus,
            "corpus_hours": corpus_hours, "corpus_gpu_hours": corpus_hours * num_gpus if corpus_hours else None,
            "corpus_hours_by_source": by_source, "reach": reach}


# ----------------------------------------------------------------------------------------------------------------------
# Chạy thật
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class Tier:
    """Một engine của chuỗi: cấu hình, hàm dựng engine (gọi lười, một lần) và khoá nếu engine không an toàn đa luồng."""

    name: str
    cfg: dict
    factory: Callable[[], object]
    serial: bool = False  # True: chỉ một luồng gọi engine cùng lúc (baseline Paddle + VietOCR)
    _engine: object = field(default=None, repr=False)
    _build_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _call_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def max_tokens(self) -> int | None:
        """max_tokens khai báo trong cấu hình (None với engine không sinh token)."""
        return self.cfg.get("max_tokens")

    def engine(self):
        """Engine, dựng ở lần gọi đầu (an toàn đa luồng)."""
        with self._build_lock:
            if self._engine is None:
                self._engine = self.factory()
            return self._engine

    def run(self, image: bytes, ext: str) -> dict:
        """Chạy engine trên một ảnh (tuần tự hoá nếu serial)."""
        engine = self.engine()
        if self.serial:
            with self._call_lock:
                return engine.run_bytes(image, ext)
        return engine.run_bytes(image, ext)


def retarget(cfg: dict, base_url: str | None = None, port: int | None = None, gpu_mem: float | None = None) -> dict:
    """
    Bản sao cấu hình engine VLM trỏ tới server khác: đổi `base_url`, hoặc đổi cổng trong `base_url` và trong lệnh `serve`.

    Dùng để chạy nhiều server vLLM cùng máy (mỗi GPU một cổng) từ một cấu hình, và để các engine của chuỗi dùng chung server.
    `gpu_mem` đổi `--gpu-memory-utilization` trong lệnh `serve` (chừa chỗ cho engine baseline chạy cùng GPU).
    """
    out = copy.deepcopy(cfg)
    serve = out.get("serve")
    if gpu_mem is not None and serve and "--gpu-memory-utilization" in serve:
        serve[serve.index("--gpu-memory-utilization") + 1] = str(gpu_mem)
    if port is not None:
        host = out["base_url"].rsplit(":", 1)[0]
        out["base_url"] = f"{host}:{port}/v1"
        if serve and "--port" in serve:
            serve[serve.index("--port") + 1] = str(port)
    if base_url is not None:
        out["base_url"] = base_url
    return out


def build_tiers(names: list[str] | tuple[str, ...], engines: dict[str, dict], port: int | None = None,
                device: str = "cuda", gpu_mem: float | None = None) -> list[Tier]:
    """
    Dựng chuỗi engine từ cấu hình (configs/ocr_bakeoff/engines.json).

    Các engine VLM ("openai") phải cùng một mô hình và dùng chung server của engine VLM đầu tiên (cổng `port`, bộ nhớ GPU
    `gpu_mem` nếu cho). Engine "baseline" nạp lười trên `device`.

    Raises:
        ValueError: chuỗi rỗng, thiếu engine trong cấu hình, type không hỗ trợ, hoặc hai engine VLM khác mô hình.
    """
    if not names:
        raise ValueError("Chuỗi engine rỗng")
    unknown = [n for n in names if n not in engines]
    if unknown:
        raise ValueError(f"Engine không có trong cấu hình: {unknown} (có: {', '.join(engines)})")
    first_vlm: dict | None = None
    tiers = []
    for name in names:
        cfg = engines[name]
        kind = cfg.get("type")
        if kind not in ("openai", "baseline"):
            raise ValueError(f"Engine {name}: type {kind!r} không dùng được trong chuỗi (chỉ openai / baseline)")
        if kind == "openai":
            cfg = retarget(cfg, port=port, gpu_mem=gpu_mem) if first_vlm is None else retarget(cfg, base_url=first_vlm["base_url"])
            if first_vlm is None:
                first_vlm = cfg
            elif cfg.get("model") != first_vlm.get("model"):
                raise ValueError(f"Engine {name} dùng mô hình {cfg.get('model')!r}, khác {first_vlm.get('model')!r} của engine đầu: "
                                 "các engine VLM trong chuỗi phải dùng chung một server")
        else:
            cfg = {**cfg, "device": device}
        tiers.append(Tier(name, cfg, factory=lambda c=cfg: ENGINE_TYPES[c["type"]](c), serial=kind == "baseline"))
    return tiers


class CascadeOcr:
    """OCR một trang qua chuỗi engine: chạy engine đầu, bị báo thì engine sau, theo pick_tier. An toàn đa luồng."""

    def __init__(self, tiers: list[Tier], retries: int = 2, backoff_s: float = 3.0, max_conn_errors: int = 20) -> None:
        """
        Args:
            tiers:           Chuỗi engine (build_tiers).
            retries:         Số lần thử lại một lượt gọi khi lỗi tạm thời (server bận, quá thời gian, 5xx).
            backoff_s:       Giây chờ trước lần thử lại đầu, nhân đôi mỗi lần.
            max_conn_errors: Số lượt gọi liên tiếp không kết nối được server thì ném ServerDown.
        """
        self.tiers = tiers
        self.retries, self.backoff_s, self.max_conn_errors = retries, backoff_s, max_conn_errors
        self._conn_errors: dict[str, int] = {}  # theo server (base_url), engine VLM cùng server dùng chung bộ đếm
        self._lock = threading.Lock()
        self._max_tokens = {t.name: t.max_tokens for t in tiers}

    def _call(self, tier: Tier, image: bytes, ext: str) -> dict:
        """
        Gọi một engine trên một ảnh, thử lại lỗi tạm thời. Lỗi cuối cùng trả về dưới dạng bản ghi có "error" (bộ kiểm báo "error"
        và trang rơi xuống engine sau). Không kết nối được server nhiều lần liên tiếp thì ném ServerDown.
        """
        import requests

        delay = self.backoff_s
        server = tier.cfg.get("base_url") or tier.name
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                res = tier.run(image, ext)
                with self._lock:
                    self._conn_errors[server] = 0
                res["text"] = to_plain(res["raw"], tier.cfg.get("output", "plain"))
                res["error"] = None
                res["seconds"] = round(time.time() - t0, 2)
                return res
            except requests.ConnectionError as exc:
                with self._lock:
                    self._conn_errors[server] = self._conn_errors.get(server, 0) + 1
                    down = self._conn_errors[server] >= self.max_conn_errors
                if down:
                    raise ServerDown(f"Không kết nối được server của {tier.name} {self.max_conn_errors} lượt liên tiếp: {exc}") from exc
                err, retry = f"{type(exc).__name__}: {exc}", True
            except requests.Timeout as exc:
                err, retry = f"{type(exc).__name__}: {exc}", True
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                err, retry = f"{type(exc).__name__}: {exc}", status in RETRY_STATUS
            except Exception as exc:  # noqa: BLE001 - trang lạ làm engine lỗi (ảnh hỏng, OOM...) không được dừng cả sách
                err, retry = f"{type(exc).__name__}: {exc}", False
            if not retry or attempt == self.retries:
                return {"raw": None, "text": None, "finish_reason": None, "error": err[:500]}
            time.sleep(delay)
            delay *= 2
        raise AssertionError("không tới được")  # pragma: no cover

    def page(self, image: bytes, ext: str = "jpg") -> dict:
        """
        OCR một trang.

        Returns:
            {"text": văn bản được nhận ("" nếu trang trắng hoặc bị loại), "status": "ok" | "blank" | "flagged", "tier": engine
             được chọn, "finish_reason", "reasons": mã bộ kiểm còn lại trên bản chọn, "attempts": [{"tier", "reasons", "error"?}]
             (chỉ khi phải chạy lại), "rejected_text": văn bản bị loại (chỉ khi flagged, để kiểm tay)}.
            "blank" = mọi engine đều trả trang rỗng (trang trắng thật, không phải lỗi). "flagged" = engine cuối vẫn bị báo.
        """
        attempts: list[dict] = []

        def get_rec(name: str) -> dict:
            """Gọi engine `name` và ghi lại lần thử."""
            tier = next(t for t in self.tiers if t.name == name)
            rec = self._call(tier, image, ext)
            reasons = guard_page(rec, tier.max_tokens)
            attempts.append({"tier": name, "reasons": reasons, **({"error": rec["error"]} if rec.get("error") else {})})
            return rec

        name, rec, reasons = pick_tier([t.name for t in self.tiers], get_rec, self._max_tokens)
        text = (rec or {}).get("text") or ""
        if not reasons:
            status = "ok"
        elif all(a["reasons"] == ["empty"] for a in attempts):
            status = "blank"
        else:
            status = "flagged"
        out = {"text": text if status == "ok" else "", "status": status, "tier": name,
               "finish_reason": (rec or {}).get("finish_reason"), "reasons": reasons}
        if len(attempts) > 1 or reasons:
            out["attempts"] = attempts
        if status == "flagged":
            out["rejected_text"] = text
        return out

    def page_text(self, image) -> str:
        """Giao diện giống vi_corpus.common.ocr.PageOcr: nhận ảnh PIL, trả văn bản ("" nếu trang trắng hoặc bị loại)."""
        import io

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        return self.page(buf.getvalue(), "jpg")["text"]
