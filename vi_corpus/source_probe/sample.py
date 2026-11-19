"""
Lấy mẫu nhỏ từ xa cho một nguồn ứng viên, không tải cả bộ dữ liệu.

- hf_parquet: chọn ngẫu nhiên (theo seed) các row group trong các file parquet trên Hugging Face, chỉ đọc các cột cần, mỗi
  row group lấy vài dòng ngẫu nhiên. HfFileSystem đọc theo dải byte nên chỉ tải đúng row group đã chọn.
- hf_rows: API datasets-server (/size để biết số dòng, /rows để lấy một khúc tại vị trí ngẫu nhiên), cho bộ lưu dạng JSON lớn.
- hplt: đọc tuần tự phần ĐẦU shard jsonl.zst của HPLT 3.0 (file 10-26 GB, không đọc ngẫu nhiên được), lấy một kho đệm rồi chọn
  ngẫu nhiên trong đó. Mẫu này lệch về đầu shard, báo cáo phải ghi rõ.

Mỗi bản ghi mẫu là dict: id, text (tiếng Việt), en (bản gốc tiếng Anh nếu có, không thì ""), extra (cột số đo sẵn có).
"""

import io
import json
import math
import random
import time

import requests

from vi_corpus.source_probe.specs import ProbeSpec

DATASETS_SERVER = "https://datasets-server.huggingface.co"
ROWS_PER_GROUP = 10  # số dòng lấy trong mỗi row group / mỗi khúc /rows: rải mẫu ra nhiều chỗ mà không tốn quá nhiều request
HPLT_POOL_FACTOR = 5  # HPLT: đọc ngần này lần n bản ghi đầu shard rồi chọn ngẫu nhiên n bản
HTTP_TIMEOUT = 120
ROWS_PAUSE = 1.0  # giây nghỉ giữa hai request /rows, tránh 429 của datasets-server


def _fields(spec_field: str) -> list[str]:
    """Tách trường dạng "a+b" thành danh sách tên cột."""
    return [f for f in spec_field.split("+") if f]


def _join(row: dict, spec_field: str) -> str:
    """Nối các trường (hỏi + đáp) thành một văn bản, bỏ trường rỗng."""
    return "\n\n".join(str(row[f]) for f in _fields(spec_field) if row.get(f))


def _to_jsonable(value):
    """Đổi giá trị cột (numpy / list / bytes) về kiểu ghi được ra JSON."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return str(value)


def make_record(spec: ProbeSpec, row: dict, rid: str) -> dict:
    """Dựng bản ghi mẫu chuẩn từ một dòng thô của nguồn."""
    return {"id": str(row.get("id") or rid), "text": _join(row, spec.text_field),
            "en": _join(row, spec.en_field) if spec.en_field else "",
            "extra": {k: _to_jsonable(row.get(k)) for k in spec.extra if k in row}}


def pick_groups(group_counts: list[int], n_groups: int, rng: random.Random) -> list[tuple[int, int]]:
    """
    Chọn n_groups cặp (chỉ số file, chỉ số row group) phân biệt, ngẫu nhiên theo rng.

    Args:
        group_counts: số row group của từng file.
        n_groups:     số cặp cần chọn; lớn hơn tổng số row group thì lấy hết.
    """
    all_pairs = [(fi, gi) for fi, cnt in enumerate(group_counts) for gi in range(cnt)]
    if n_groups >= len(all_pairs):
        return all_pairs
    return sorted(rng.sample(all_pairs, n_groups))


def sample_hf_parquet(spec: ProbeSpec, n: int, seed: int, fs=None) -> list[dict]:
    """
    Lấy khoảng n bản ghi từ các file parquet của spec.location.

    fs: hệ thống file kiểu fsspec (có find / open); None = HfFileSystem (spec.location là đường dẫn "datasets/<repo>/...").

    Đọc footer của tối đa max(số row group cần, 8) file chọn ngẫu nhiên, để bộ có hàng trăm file nhỏ không tốn hàng trăm request.
    """
    import pyarrow.parquet as pq

    if fs is None:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
    rng = random.Random(seed)
    files = sorted(f for f in fs.find(spec.location) if f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"không thấy file parquet ở {spec.location}")
    n_groups = math.ceil(n / ROWS_PER_GROUP)
    files = rng.sample(files, min(len(files), max(n_groups, 8)))
    handles = [pq.ParquetFile(fs.open(f, "rb")) for f in files]
    wanted = set(_fields(spec.text_field)) | set(_fields(spec.en_field)) | set(spec.extra) | {"id"}
    out: list[dict] = []
    for fi, gi in pick_groups([h.metadata.num_row_groups for h in handles], n_groups, rng):
        cols = [c for c in handles[fi].schema_arrow.names if c in wanted]
        rows = handles[fi].read_row_group(gi, columns=cols).to_pylist()
        for ri in sorted(rng.sample(range(len(rows)), min(ROWS_PER_GROUP, len(rows)))):
            out.append(make_record(spec, rows[ri], f"{files[fi]}#{gi}:{ri}"))
    return out[:n]


def _get_json(url: str, params: dict, tries: int = 6) -> dict:
    """GET một endpoint JSON; lỗi 5xx / 429 / mạng thì thử lại (lùi 2, 4, 8... giây), lỗi 4xx khác hoặc hết lượt thì báo lỗi."""
    for attempt in range(tries):
        try:
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if resp.status_code < 500 and resp.status_code != 429:
                resp.raise_for_status()
                return resp.json()
            err: Exception = requests.HTTPError(f"{resp.status_code} cho {resp.url}")
        except (requests.ConnectionError, requests.Timeout) as exc:
            err = exc
        if attempt == tries - 1:
            raise err
        time.sleep(2 ** (attempt + 1))
    raise AssertionError("không tới được đây")


def sample_hf_rows(spec: ProbeSpec, n: int, seed: int) -> list[dict]:
    """Lấy khoảng n bản ghi qua API datasets-server, mỗi request một khúc ROWS_PER_GROUP dòng tại vị trí ngẫu nhiên."""
    rng = random.Random(seed)
    size = _get_json(f"{DATASETS_SERVER}/size", {"dataset": spec.location})
    total = sum(s["num_rows"] for s in size["size"]["splits"] if s["split"] == "train")
    out: list[dict] = []
    for _ in range(math.ceil(n / ROWS_PER_GROUP)):
        offset = rng.randrange(max(1, total - ROWS_PER_GROUP))
        data = _get_json(f"{DATASETS_SERVER}/rows", {"dataset": spec.location, "config": "default", "split": "train",
                                                    "offset": offset, "length": ROWS_PER_GROUP})
        time.sleep(ROWS_PAUSE)
        for item in data["rows"]:
            out.append(make_record(spec, item["row"], f"row{item['row_idx']}"))
    return out[:n]


def sample_hplt(spec: ProbeSpec, n: int, seed: int) -> list[dict]:
    """
    Đọc n * HPLT_POOL_FACTOR bản ghi đầu mỗi shard (giải nén zstd theo luồng) rồi chọn ngẫu nhiên n bản trên tổng kho đệm.

    Cần gói zstandard (chạy bằng `uv run --with zstandard ...`, không có trong uv.lock).
    """
    import zstandard

    rng = random.Random(seed)
    pool: list[dict] = []
    per_shard = math.ceil(n * HPLT_POOL_FACTOR / len(spec.location))
    for url in spec.location:
        with requests.get(url, stream=True, timeout=HTTP_TIMEOUT) as resp:
            resp.raise_for_status()
            reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(resp.raw), encoding="utf-8")
            for i, line in enumerate(reader):
                if i >= per_shard:
                    break
                row = json.loads(line)
                pool.append(make_record(spec, row, f"{url}:{i}"))
    return rng.sample(pool, min(n, len(pool)))


def sample_source(spec: ProbeSpec, n: int, seed: int) -> list[dict]:
    """Lấy mẫu một nguồn theo spec.kind."""
    if spec.kind == "hf_parquet":
        return sample_hf_parquet(spec, n, seed)
    if spec.kind == "hf_rows":
        return sample_hf_rows(spec, n, seed)
    if spec.kind == "hplt":
        return sample_hplt(spec, n, seed)
    raise ValueError(f"kind không hỗ trợ: {spec.kind}")
