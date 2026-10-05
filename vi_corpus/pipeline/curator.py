"""
Adapter NeMo Curator + Ray: chạy các stage nặng của pipeline SONG SONG trên nhiều CPU / nhiều GPU.

Cùng bộ khung với ViLA (packages/pipeline/executors.py, packages/embedder/stage.py): mỗi stage là một
`ProcessingStage[DocumentBatch, DocumentBatch]` khai báo `Resources(cpus, gpus)`, executor Ray (Xenna / ray_actor_pool /
ray_data) tạo nhiều actor và phân các phân vùng (DocumentBatch) cho chúng. Stage nặng GPU nạp mô hình trong `setup()`
nên mỗi actor giữ một bản mô hình trên GPU riêng của nó (Ray đặt CUDA_VISIBLE_DEVICES), 4 GPU = 4 actor chạy cùng lúc.

Không cần viết lại logic: các hàm thuần của pipeline (annotate_language, annotate_quality, embed_rows, ocr_one_book)
được bọc bằng RowsStage. Hàng được chuyển giữa tiến trình dưới dạng JSON trong một cột chuỗi ("payload") để mọi executor
(kể cả Ray Data / Arrow) giữ nguyên kiểu dữ liệu (số nguyên, None, list) — không bị pandas tự đổi int thành float.

Module này import nemo_curator ngay đầu file, nên chỉ import khi người dùng bật --executor (cài bằng `uv sync --group curator`).
"""

import functools
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from vi_corpus.pipeline import embed as embed_mod
from vi_corpus.pipeline.config import RunConfig
from vi_corpus.pipeline.language import annotate_language
from vi_corpus.pipeline.quality import annotate_quality

logger = logging.getLogger("vi_corpus")

EXECUTOR_CHOICES = ("xenna", "ray_actor_pool", "ray_data")

RowsFn = Callable[[list[dict]], list[dict]]


def encode_rows(rows: list[dict]) -> pd.DataFrame:
    """Gói các bản ghi thành DataFrame hai cột (doc_id, payload JSON) để đi qua executor mà không đổi kiểu dữ liệu."""
    return pd.DataFrame({"doc_id": [r["doc_id"] for r in rows],
                         "payload": [json.dumps(r, ensure_ascii=False) for r in rows]})


def decode_rows(df: pd.DataFrame) -> list[dict]:
    """Ngược lại của encode_rows."""
    return [json.loads(p) for p in df["payload"]]


@dataclass
class RowsStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Stage Curator bọc một hàm `rows -> rows`: giải mã phân vùng, gọi hàm, mã hóa kết quả.

    `factory` được gọi MỘT lần trong `setup()` trên actor (không phải trên driver) để tạo hàm; nhờ đó stage GPU nạp mô hình
    lên đúng GPU của actor và không phải pickle mô hình. Factory và hàm phải pickle được (hàm cấp module, functools.partial).
    """

    factory: Callable[[], RowsFn] = None  # type: ignore[assignment]
    name: str = "rows_stage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        """Cột đầu vào bắt buộc."""
        return (["data"], ["payload"])

    def outputs(self) -> tuple[list[str], list[str]]:
        """Cột đầu ra."""
        return (["data"], ["payload"])

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Tạo hàm xử lý trên actor (nạp mô hình lên GPU của actor nếu là stage GPU)."""
        self._fn = self.factory()

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Xử lý một phân vùng: giải mã, gọi hàm, mã hóa lại."""
        if getattr(self, "_fn", None) is None:
            self.setup(None)
        out = self._fn(decode_rows(task.to_pandas()))
        return DocumentBatch(dataset_name=task.dataset_name, data=encode_rows(out),
                             _metadata=task._metadata, _stage_perf=task._stage_perf)


# ----- factory cấp module (pickle được) cho từng stage -----

def _language_fn() -> RowsFn:
    """Factory stage language."""
    return annotate_language


def _quality_fn(cfg: RunConfig) -> RowsFn:
    """Factory stage quality."""
    return functools.partial(annotate_quality, cfg=cfg)


def _embed_fn(cfg: RunConfig) -> RowsFn:
    """Factory stage embed: nạp mô hình HF lên GPU của actor một lần, trả hàm nhúng từng phân vùng."""
    spec = cfg.embed_spec or ""
    if not spec.startswith("hf:"):
        raise ValueError("Chạy song song chỉ hỗ trợ embedder 'hf:<model_id>' (tfidf phải fit trên toàn corpus, chạy tuần tự)")
    model = embed_mod.HfEmbedder(spec[3:], "auto", cfg.embed_max_seq, cfg.embed_batch_size, cfg.embed_prompt)

    def run(rows: list[dict]) -> list[dict]:
        """Nhúng các bản ghi của một phân vùng."""
        mat = model.embed([r["text"] for r in rows])
        return [{"doc_id": r["doc_id"], "embedding": v.tolist()} for r, v in zip(rows, mat)]
    return run


# ----- Ray + executor (theo packages/pipeline/executors.py của ViLA) -----

_PINNED: dict[str, str] = {}  # RAY_ADDRESS do chính module này đặt (để phân biệt với địa chỉ người dùng tự đặt)


def init_ray(num_gpus: int | None = None, num_cpus: int | None = None, address: str | None = None) -> bool:
    """
    Khởi động (hoặc nối vào) Ray; trả True nếu vừa khởi tạo.

    Args:
        num_gpus: Số GPU cho Ray cục bộ (None = tự phát hiện). Bị bỏ qua khi có address.
        num_cpus: Số CPU cho Ray cục bộ (None = tự phát hiện).
        address:  Địa chỉ cụm Ray có sẵn ("auto", "ray://host:10001", ...); None = chạy Ray cục bộ trong máy này.
    """
    import ray

    if ray.is_initialized():
        return False
    if not address and os.environ.get("RAY_ADDRESS") == _PINNED.get("addr"):
        os.environ.pop("RAY_ADDRESS", None)  # địa chỉ của lần Ray trước đã bị executor tắt; không được nối vào đó
    kwargs: dict = {"ignore_reinit_error": True}
    if address:
        kwargs["address"] = address
    else:
        if num_cpus is not None:
            kwargs["num_cpus"] = num_cpus
        if num_gpus is not None:
            kwargs["num_gpus"] = num_gpus
    ray.init(**kwargs)
    try:  # ghim địa chỉ GCS: cosmos-xenna báo lỗi "multiple Ray instances" nếu máy có nhiều cụm
        gcs = ray.get_runtime_context().gcs_address
        if gcs:
            os.environ["RAY_ADDRESS"] = _PINNED["addr"] = gcs
    except Exception as exc:  # noqa: BLE001 — chỉ là cố gắng tốt nhất
        logger.debug("không ghim được RAY_ADDRESS: %s", exc)
    return True


def build_executor(name: str):
    """
    Tạo executor Curator theo tên: "xenna" (mặc định ở ViLA, autoscale streaming), "ray_actor_pool", "ray_data".

    Raises:
        ValueError: nếu name không thuộc EXECUTOR_CHOICES.
    """
    if name == "xenna":
        from nemo_curator.backends.xenna import XennaExecutor
        return XennaExecutor(config={"execution_mode": "streaming", "logging_interval": 30, "autoscale_interval_s": 60,
                                     "cpu_allocation_percentage": 0.95, "ignore_failures": False},
                             ignore_head_node=False)
    if name == "ray_actor_pool":
        from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
        return RayActorPoolExecutor(config={}, ignore_head_node=False)
    if name == "ray_data":
        from nemo_curator.backends.ray_data import RayDataExecutor
        return RayDataExecutor(config={}, ignore_head_node=False)
    raise ValueError(f"executor không hợp lệ: {name!r}; chọn một trong {EXECUTOR_CHOICES}")


class CuratorBackend:
    """
    Bộ chạy song song cho runner (language / quality / embed) và cho OCR nhiều GPU.

    Dùng như context manager để Ray được tắt sạch: `with CuratorBackend("xenna", num_gpus=4) as backend: ...`.

    Attributes:
        executor_name:    Một trong EXECUTOR_CHOICES.
        num_gpus:         Số GPU Ray được dùng (None = tự phát hiện).
        num_cpus:         Số CPU Ray được dùng (None = tự phát hiện).
        address:          Địa chỉ cụm Ray có sẵn (None = Ray cục bộ).
        rows_per_task:    Số bản ghi mỗi phân vùng (DocumentBatch). Nhỏ hơn = cân tải tốt hơn nhưng nhiều overhead.
        gpus_per_worker:  Phần GPU mỗi actor GPU giữ (1.0 = một actor / GPU; 0.5 = hai actor / GPU nếu mô hình nhỏ).
        cpu_workers_cpus: Số CPU mỗi actor của stage CPU (language, quality).
    """

    def __init__(self, executor_name: str = "xenna", num_gpus: int | None = None, num_cpus: int | None = None,
                 address: str | None = None, rows_per_task: int = 500, gpus_per_worker: float = 1.0,
                 cpu_workers_cpus: float = 1.0) -> None:
        """Lưu tham số; Ray chỉ được khởi động khi vào context (`with`)."""
        if executor_name not in EXECUTOR_CHOICES:
            raise ValueError(f"executor không hợp lệ: {executor_name!r}; chọn một trong {EXECUTOR_CHOICES}")
        self.executor_name, self.num_gpus, self.num_cpus, self.address = executor_name, num_gpus, num_cpus, address
        self.rows_per_task, self.gpus_per_worker, self.cpu_workers_cpus = rows_per_task, gpus_per_worker, cpu_workers_cpus
        self._owns_ray = False

    def __enter__(self) -> "CuratorBackend":
        """Khởi động Ray."""
        self._owns_ray = init_ray(self.num_gpus, self.num_cpus, self.address)
        return self

    def __exit__(self, *exc) -> None:
        """Tắt Ray nếu chính backend này đã khởi động nó."""
        if self._owns_ray:
            import ray
            ray.shutdown()

    def run(self, name: str, rows: list[dict], factory: Callable[[], RowsFn], resources: Resources) -> list[dict]:
        """
        Chạy một stage RowsStage trên mọi bản ghi, song song theo phân vùng, trả kết quả theo thứ tự đầu vào.

        Thứ tự đầu ra của executor không được đảm bảo, nên kết quả được sắp lại theo doc_id của `rows`
        (mọi hàm stage đều giữ nguyên tập doc_id).
        """
        if not rows:
            return []
        # Executor của Curator gọi ray.shutdown() sau MỖI lần chạy, nên phải khởi động lại Ray (đúng số GPU / CPU đã chọn) trước mỗi stage
        self._owns_ray = init_ray(self.num_gpus, self.num_cpus, self.address) or self._owns_ray
        tasks = [DocumentBatch(dataset_name="vi_corpus", data=encode_rows(rows[i:i + self.rows_per_task]))
                 for i in range(0, len(rows), self.rows_per_task)]
        pipeline = Pipeline(name=f"vi-corpus-{name}", description=f"vi_corpus stage {name}",
                            stages=[RowsStage(factory=factory, name=name, resources=resources)])
        logger.info("[%s] %d phân vùng x ~%d bản ghi, executor=%s, resources=%s", name, len(tasks), self.rows_per_task,
                    self.executor_name, resources)
        results = pipeline.run(executor=build_executor(self.executor_name), initial_tasks=tasks) or []
        out = [row for batch in results for row in decode_rows(batch.to_pandas())]
        order = {r["doc_id"]: i for i, r in enumerate(rows)}
        out.sort(key=lambda r: order[r["doc_id"]])
        if len(out) != len(rows):
            raise RuntimeError(f"Stage {name}: vào {len(rows)} bản ghi nhưng ra {len(out)} (executor làm rơi phân vùng?)")
        return out

    def language(self, rows: list[dict]) -> list[dict]:
        """Stage language song song nhiều CPU."""
        return self.run("language", rows, _language_fn, Resources(cpus=self.cpu_workers_cpus))

    def quality(self, rows: list[dict], cfg: RunConfig) -> list[dict]:
        """Stage quality song song nhiều CPU."""
        return self.run("quality", rows, functools.partial(_quality_fn, cfg), Resources(cpus=self.cpu_workers_cpus))

    def embed(self, rows: list[dict], cfg: RunConfig) -> list[dict]:
        """Stage embed song song nhiều GPU: mỗi actor giữ một bản mô hình trên một GPU."""
        return self.run("embed", rows, functools.partial(_embed_fn, cfg),
                        Resources(cpus=2.0, gpus=self.gpus_per_worker))
