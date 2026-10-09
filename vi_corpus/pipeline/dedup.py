"""
Stage dedup + rights gate: loại trùng lặp chính xác và gần trùng, gắn cổng quyền, chốt trạng thái từng bản ghi.

Đây là vòng 1 của dedup 2 vòng (D-09): bỏ trùng trong phạm vi lần chạy và ghi kho dấu vân tay của các văn bản còn
sống, cùng bia mộ cho văn bản bị loại (index_rows, write_index); vòng 2 (vi_corpus.pipeline.dedup_global, scripts/dedup_global.py) gộp kho của mọi nguồn.

- Chỉ bản ghi đã qua quality mới vào dedup. Hai nhóm tách biệt (Q4): "cpt" (văn bản thường, dedup giữa mọi nguồn) và
  "sft" (hỏi - đáp SEA-Instruct, chỉ dedup trong nhóm của mình), theo SourceProfile.dedup_group.
- Đơn vị dedup là VĂN BẢN (R-19): các đoạn của một sách / bài dài được gom theo parent_doc_id. Chữ ký MinHash và sha256
  của cả văn bản gốc được tính ở stage prepare (doc_minhash, doc_sha256); bản ghi thiếu thì tính từ text của chính nó.
  * exact: cùng sha256 văn bản.
  * fuzzy: MinHash trên CẢ văn bản (128 hoán vị, shingle 5 từ), LSH 16 băng x 8 hàng, xác nhận bằng Jaccard ước lượng >= ngưỡng
    (cấu hình A, giữ tới khi có kết quả đo S-11, R-34). Thuần numpy, vector hoá theo băng.
  Văn bản được giữ làm đại diện của họ trùng: ưu tiên nguồn trước (R-15, cfg.source_priority), cùng mức thì điểm chất
  lượng rule trung bình cao hơn (R-19), rồi doc_id. Mọi đoạn của văn bản bị trùng bị loại.
- L1 trùng bao hàm (D-03 L1, G-05 bước 3): MinHash cả văn bản không bắt được văn bản nằm trong một văn bản lớn hơn
  (chương chép vào sách khác, bài báo nằm trong tuyển tập; đo ở tests/test_dedup_recall.py). Mỗi văn bản còn sống được
  băm theo đoạn văn (ngăn bởi dòng trống, >= cfg.para_min_words từ, so sau khi gộp khoảng trắng); văn bản có >=
  cfg.contained_min_paras đoạn văn dài mà >= cfg.contained_frac số đó nằm trong MỘT văn bản lớn hơn (nhiều đoạn văn dài
  hơn) thì bị loại, dup_kind "contained", dup_of = văn bản chứa nó, chung họ trùng với văn bản đó. Giữ bản lớn vì nó
  chứa toàn bộ nội dung bản nhỏ (không xét ưu tiên nguồn ở bước này). Hash đoạn văn có ở > L1_MAX_DOCS văn bản là câu
  khuôn mẫu, bỏ qua (dòng lặp liên văn bản đã được D-04 C xử lý). Chỉ trong phạm vi lần chạy: kho dấu vân tay vòng 2
  chưa chứa hash đoạn văn.
- Sau đó, trong mỗi nhóm, đoạn có text giống hệt một đoạn tốt hơn ở văn bản khác cũng bị loại (exact theo đoạn).
- Họ trùng: dedup_family_id là mã băm của văn bản đại diện, chung cho mọi đoạn của văn bản. Chia train/val/test sau này
  luôn chia theo mã này để không rò rỉ giữa các bản gần trùng.
- Cổng quyền: rights_status không thuộc OPEN_RIGHTS thì rights_gate = "quarantine"; chế độ "enforce" loại luôn
  (R-01: hiện chỉ gắn nhãn).
"""

import hashlib
import zlib
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vi_corpus.common.schema import text_sha256
from vi_corpus.pipeline.config import OPEN_RIGHTS, RunConfig

DEDUP_VERSION = "3"  # 1 = theo bản ghi, một nhóm chung; 2 = theo văn bản, nhóm cpt / sft, ưu tiên nguồn; 3 = + L1 bao hàm
NUM_PERM, BANDS, ROWS = 128, 16, 8  # BANDS * ROWS == NUM_PERM; ngưỡng LSH ~ (1/16)^(1/8) ~ 0.71
SHINGLE = 5
MINHASH_VERSION = "2"  # 1 = chỉ 20.000 từ đầu; 2 = cả văn bản (sách dài chung phần mở đầu không còn bị coi là trùng)
SHINGLE_BLOCK = 8192  # số shingle mỗi khối khi lấy min, giữ bộ nhớ tạm ~ NUM_PERM x 8192 x 8 byte (~8 MB)
MIN_FUZZY_WORDS = 20
MAX_BUCKET_PAIRS = 50  # bucket LSH lớn hơn: chỉ so mỗi phần tử với phần tử đầu (tránh bùng nổ số cặp)
L1_MAX_DOCS = 50  # hash đoạn văn có ở nhiều văn bản hơn mức này là câu khuôn mẫu, không dùng để xét bao hàm
_MASK = np.uint64(0xFFFFFFFF)
_rng = np.random.default_rng(12345)  # hằng số cố định: chữ ký MinHash tái lập được giữa các lần chạy
_A = (_rng.integers(1, 1 << 31, NUM_PERM, dtype=np.uint64) * 2 + 1).astype(np.uint64)
_B = _rng.integers(0, 1 << 32, NUM_PERM, dtype=np.uint64)

INDEX_SCHEMA = pa.schema([
    ("doc_id", pa.string()),  # doc_id văn bản (parent_doc_id nếu là sách đã cắt đoạn)
    ("source_key", pa.string()),
    ("dedup_group", pa.string()),  # "cpt" | "sft"
    ("doc_sha256", pa.string()),
    ("minhash", pa.binary()),  # NUM_PERM giá trị uint32 (512 byte); null nếu văn bản quá ngắn
    ("quality_score", pa.float64()),  # điểm rule trung bình các đoạn còn sống
    ("priority", pa.int64()),
    ("n_chunks", pa.int64()),
    ("token_count", pa.int64()),
    ("rights_status", pa.string()),  # để vòng 2 bật được rights gate "enforce" (R-01)
    ("alive", pa.bool_()),  # False = văn bản đã bị loại ở lần chạy này (bia mộ: che dòng cũ của lần chạy trước)
    ("run", pa.string()),  # tên lần chạy ghi ra dòng này
    ("written_at", pa.string()),  # thời điểm ghi (UTC, ISO 8601): vòng 2 lấy dòng mới nhất của mỗi văn bản
])


def minhash(text: str) -> np.ndarray | None:
    """
    Chữ ký MinHash (NUM_PERM giá trị uint64, mỗi giá trị < 2^32) của text, theo shingle SHINGLE từ.

    Returns:
        None nếu text quá ngắn (< MIN_FUZZY_WORDS từ) để so gần trùng.
    """
    words = text.lower().split()
    if len(words) < MIN_FUZZY_WORDS:
        return None
    n = max(len(words) - SHINGLE + 1, 1)
    x = np.fromiter((zlib.crc32(" ".join(words[i:i + SHINGLE]).encode("utf-8")) for i in range(n)),
                    dtype=np.uint64, count=n)
    sig = np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, n, SHINGLE_BLOCK):  # cả văn bản, theo khối để không phình bộ nhớ với sách dài
        block = x[start:start + SHINGLE_BLOCK]
        sig = np.minimum(sig, ((_A[:, None] * block[None, :] + _B[:, None]) & _MASK).min(axis=1))
    return sig


def minhash_bytes(text: str) -> bytes | None:
    """Chữ ký MinHash dạng bytes (uint32, 512 byte) để lưu trong Parquet; None nếu text quá ngắn."""
    sig = minhash(text)
    return None if sig is None else sig.astype(np.uint32).tobytes()


def sig_from_bytes(raw: bytes | None) -> np.ndarray | None:
    """Ngược lại của minhash_bytes."""
    return None if raw is None else np.frombuffer(raw, dtype=np.uint32)


class UnionFind:
    """Union-find trên chỉ số nguyên; gốc của mỗi họ luôn là chỉ số NHỎ NHẤT (= bản tốt nhất khi đã sắp xếp)."""

    def __init__(self, n: int) -> None:
        """Tạo n phần tử rời nhau."""
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        """Gốc của phần tử i (nén đường đi)."""
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        """Gộp hai họ; gốc mới là gốc nhỏ hơn."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def lsh_pairs(sigs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Các cặp chỉ số (i < j) gần trùng: cùng một băng LSH và Jaccard ước lượng >= threshold.

    Args:
        sigs: Ma trận (n, NUM_PERM) chữ ký MinHash.

    Returns:
        Mảng (k, 2) int64, đã sắp xếp. Bucket có hơn MAX_BUCKET_PAIRS phần tử chỉ so với phần tử đầu (đủ để nối họ nếu
        cả bucket thật sự trùng; có thể bỏ sót cặp hiếm ở bucket rất lớn).
    """
    n = len(sigs)
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    sigs64 = sigs.astype(np.uint64)
    cand: list[np.ndarray] = []
    for b in range(BANDS):
        band = sigs64[:, b * ROWS:(b + 1) * ROWS]
        key = np.zeros(n, dtype=np.uint64)
        for c in range(ROWS):  # băm băng thành một số; va chạm hiếm và bị loại ở bước xác nhận Jaccard
            key = key * np.uint64(1_000_003) ^ band[:, c]
        order = np.argsort(key, kind="stable")
        sk = key[order]
        bounds = np.flatnonzero(np.diff(sk)) + 1
        for grp in np.split(order, bounds):
            if len(grp) < 2:
                continue
            grp = np.sort(grp)
            if len(grp) <= MAX_BUCKET_PAIRS:
                ii, jj = np.triu_indices(len(grp), k=1)
                cand.append(np.stack([grp[ii], grp[jj]], axis=1))
            else:
                cand.append(np.stack([np.full(len(grp) - 1, grp[0]), grp[1:]], axis=1))
    if not cand:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.unique(np.concatenate(cand).astype(np.int64), axis=0)
    sim = (sigs[pairs[:, 0]] == sigs[pairs[:, 1]]).mean(axis=1)
    return pairs[sim >= threshold]


def family_id(rep_doc_id: str) -> str:
    """Mã họ trùng: băm ngắn của doc_id văn bản đại diện (ổn định giữa các lần chạy)."""
    return "fam_" + hashlib.sha1(rep_doc_id.encode("utf-8")).hexdigest()[:12]


def doc_key(row: dict) -> str:
    """doc_id của văn bản chứa bản ghi (parent_doc_id nếu bản ghi là một đoạn)."""
    return row.get("parent_doc_id") or row["doc_id"]


def group_docs(alive: list[dict], cfg: RunConfig) -> dict[str, list[dict]]:
    """
    Gom các đoạn còn sống thành văn bản, theo nhóm dedup.

    Returns:
        {nhóm: [văn bản]}; mỗi văn bản là {"doc_id", "source_key", "group", "sha", "sig", "score", "priority",
        "rows"}, đã sắp theo (ưu tiên nguồn, -điểm trung bình, doc_id) nên chỉ số nhỏ = bản được giữ.
    """
    docs: dict[tuple[str, str], dict] = {}
    for row in alive:
        group = cfg.profile(row["source_key"]).dedup_group
        d = docs.get((group, doc_key(row)))
        if d is None:
            sha = row.get("doc_sha256") or text_sha256(row["text"] or "")
            raw = row.get("doc_minhash") if row.get("doc_sha256") else minhash_bytes(row["text"] or "")
            d = docs[(group, doc_key(row))] = {"doc_id": doc_key(row), "source_key": row["source_key"], "group": group,
                                                "sha": sha, "sig": sig_from_bytes(raw),
                                                "priority": cfg.priority(row["source_key"]), "rows": []}
        d["rows"].append(row)
    out: dict[str, list[dict]] = {}
    for d in docs.values():
        d["score"] = sum(r["quality_score"] for r in d["rows"]) / len(d["rows"])
        out.setdefault(d["group"], []).append(d)
    for members in out.values():
        members.sort(key=lambda d: (d["priority"], -d["score"], d["doc_id"]))
    return out


def dedup_docs(docs: list[dict], threshold: float) -> tuple[list[int], dict[int, str]]:
    """
    Gộp các văn bản (đã sắp xếp, xem group_docs) thành họ trùng.

    Returns:
        (rep: chỉ số văn bản đại diện của từng văn bản, kinds: {chỉ số văn bản bị trùng: "exact" | "fuzzy"}).
    """
    uf = UnionFind(len(docs))
    kinds: dict[int, str] = {}
    first_by_hash: dict[str, int] = {}
    for i, d in enumerate(docs):
        if d["sha"] in first_by_hash:
            uf.union(first_by_hash[d["sha"]], i)
            kinds[i] = "exact"
        else:
            first_by_hash[d["sha"]] = i
    with_sig = [i for i, d in enumerate(docs) if d["sig"] is not None]
    if len(with_sig) > 1:
        mat = np.stack([docs[i]["sig"] for i in with_sig])
        for a, b in lsh_pairs(mat, threshold):
            i, j = with_sig[a], with_sig[b]
            uf.union(i, j)
            kinds.setdefault(j, "fuzzy")
    return [uf.find(i) for i in range(len(docs))], kinds


def paragraph_hashes(rows: list[dict], min_words: int) -> set[bytes]:
    """Tập hash (blake2b 8 byte) các đoạn văn dài của một văn bản (mọi đoạn cắt của nó), sau khi gộp khoảng trắng."""
    out: set[bytes] = set()
    for row in rows:
        for para in (row["text"] or "").split("\n\n"):
            words = para.split()
            if len(words) >= min_words:
                out.add(hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8).digest())
    return out


def find_contained(docs: list[dict], alive: list[int], cfg: RunConfig) -> dict[int, int]:
    """
    Tìm văn bản nằm trong văn bản lớn hơn (L1, xem docstring module).

    Args:
        docs:  Văn bản của một nhóm (group_docs).
        alive: Chỉ số các văn bản còn sống sau exact + fuzzy theo văn bản.

    Returns:
        {chỉ số văn bản bị bao hàm: chỉ số văn bản chứa nó}. Xét theo thứ tự lớn trước, nên văn bản chứa luôn đã được
        xét trước văn bản bị chứa (để nối họ trùng theo chuỗi).
    """
    paras = {i: paragraph_hashes(docs[i]["rows"], cfg.para_min_words) for i in alive}
    owners: dict[bytes, list[int]] = {}
    for i, hs in paras.items():
        for h in hs:
            owners.setdefault(h, []).append(i)
    order = sorted(alive, key=lambda i: (-len(paras[i]), i))
    rank = {i: k for k, i in enumerate(order)}
    out: dict[int, int] = {}
    for i in order:
        if len(paras[i]) < cfg.contained_min_paras:
            continue
        overlap: dict[int, int] = {}
        for h in paras[i]:
            docs_h = owners[h]
            if len(docs_h) > L1_MAX_DOCS:
                continue
            for j in docs_h:
                if rank[j] < rank[i]:  # chỉ văn bản lớn hơn (hoặc bằng cỡ nhưng đứng trước) mới được chứa i
                    overlap[j] = overlap.get(j, 0) + 1
        if overlap:
            j, n = max(overlap.items(), key=lambda kv: (kv[1], -rank[kv[0]]))
            if n >= cfg.contained_frac * len(paras[i]):
                out[i] = j
    return out


def dedup_rows(rows: list[dict], cfg: RunConfig) -> tuple[list[dict], dict]:
    """
    Chốt cột status, rights_gate, dedup_family_id, dup_kind, dup_of cho mọi bản ghi (sửa tại chỗ).

    status: "rejected:quality" (band ngoài cfg.keep_bands), "rejected:rights" (chỉ khi rights_gate="enforce"),
    "rejected:duplicate" (văn bản / đoạn kém hơn trong họ trùng), còn lại "kept".

    Returns:
        (rows, thống kê: số bản ghi theo status, số đoạn bị loại theo loại trùng, số họ trùng có > 1 văn bản, theo nhóm).
    """
    for row in rows:
        row["rights_gate"] = "pass" if row["rights_status"] in OPEN_RIGHTS else "quarantine"
        row["dup_kind"] = row["dup_of"] = row["dedup_family_id"] = None
        if row["quality_band"] not in cfg.keep_bands:
            row["status"] = "rejected:quality"
        elif cfg.rights_gate == "enforce" and row["rights_gate"] == "quarantine":
            row["status"] = "rejected:rights"
        else:
            row["status"] = "kept"

    alive = [r for r in rows if r["status"] == "kept"]
    groups: dict[str, dict] = {}
    for group, docs in group_docs(alive, cfg).items():
        rep, kinds = dedup_docs(docs, cfg.fuzzy_threshold)
        g = groups[group] = {"docs": len(docs), "exact": 0, "fuzzy": 0, "contained": 0, "chunk_exact": 0}
        for i, d in enumerate(docs):
            fam = family_id(docs[rep[i]]["doc_id"])
            for row in d["rows"]:
                row["dedup_family_id"] = fam
                if rep[i] != i:
                    row.update(status="rejected:duplicate", dup_kind=kinds.get(i, "fuzzy"), dup_of=docs[rep[i]]["doc_id"])
                    g[row["dup_kind"]] += 1
        container = find_contained(docs, [i for i in range(len(docs)) if rep[i] == i], cfg)
        family_root = {i: rep[i] for i in range(len(docs))}
        for i, j in container.items():  # thứ tự lớn trước: họ của j đã chốt khi xét i
            family_root[i] = family_root[j]
        family_root = {i: family_root[rep[i]] for i in range(len(docs))}  # bản trùng của văn bản bị chứa cũng chuyển họ
        for i in range(len(docs)):
            if family_root[i] == rep[i]:
                continue
            fam = family_id(docs[family_root[i]]["doc_id"])
            for row in docs[i]["rows"]:
                row["dedup_family_id"] = fam
                if i in container and row["status"] == "kept":
                    row.update(status="rejected:duplicate", dup_kind="contained", dup_of=docs[container[i]]["doc_id"])
                    g["contained"] += 1
        g["families_multi"] = len({family_root[i] for i in range(len(docs)) if family_root[i] != i})
        sizes: dict[int, int] = {}
        for root in family_root.values():
            sizes[root] = sizes.get(root, 0) + 1
        g["largest_family"] = max(sizes.values(), default=0)  # số văn bản của họ trùng lớn nhất (R-19 mục 4)
        first_chunk: dict[str, str] = {}  # sha256 đoạn -> doc_id đoạn đầu tiên (thuộc văn bản tốt hơn)
        for d in docs:
            for row in d["rows"]:
                if row["status"] != "kept" or not row.get("parent_doc_id"):
                    continue
                h = text_sha256(row["text"] or "")
                if h in first_chunk:
                    row.update(status="rejected:duplicate", dup_kind="exact", dup_of=first_chunk[h])
                    g["chunk_exact"] += 1
                else:
                    first_chunk[h] = row["doc_id"]
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    return rows, {"version": DEDUP_VERSION, "statuses": statuses, "groups": groups,
                  "exact_dups": sum(g["exact"] + g["chunk_exact"] for g in groups.values()),
                  "fuzzy_dups": sum(g["fuzzy"] for g in groups.values()),
                  "contained_dups": sum(g["contained"] for g in groups.values()),
                  "families_multi": sum(g["families_multi"] for g in groups.values()),
                  "largest_family": max((g["largest_family"] for g in groups.values()), default=0),
                  "quarantined": sum(r["rights_gate"] == "quarantine" for r in rows)}


def index_rows(rows: list[dict], cfg: RunConfig, run_name: str) -> list[dict]:
    """
    Dấu vân tay (INDEX_SCHEMA) của mọi văn bản trong lần chạy: một dòng mỗi văn bản, không có text (~0,5 KB).

    Văn bản còn sống (có ít nhất một đoạn status "kept") mang đủ sha256, MinHash, điểm, ưu tiên. Văn bản bị loại ở lần
    chạy này (chất lượng, trùng, quyền) ghi thành bia mộ (alive=False, không có chữ ký) để vòng 2 không dùng lại dòng cũ
    của nó từ lần chạy trước.
    """
    out = []
    alive_ids: set[tuple[str, str]] = set()
    for group, docs in group_docs([r for r in rows if r["status"] == "kept"], cfg).items():
        for d in docs:
            alive_ids.add((group, d["doc_id"]))
            out.append({"doc_id": d["doc_id"], "source_key": d["source_key"], "dedup_group": group,
                        "doc_sha256": d["sha"], "minhash": None if d["sig"] is None else d["sig"].astype(np.uint32).tobytes(),
                        "quality_score": round(d["score"], 2), "priority": d["priority"], "n_chunks": len(d["rows"]),
                        "token_count": sum(r["token_count"] or 0 for r in d["rows"]),
                        "rights_status": d["rows"][0].get("rights_status"), "alive": True, "run": run_name})
    for row in rows:
        key = (cfg.profile(row["source_key"]).dedup_group, doc_key(row))
        if key in alive_ids:
            continue
        alive_ids.add(key)
        out.append({"doc_id": key[1], "source_key": row["source_key"], "dedup_group": key[0], "alive": False,
                    "rights_status": row.get("rights_status"), "run": run_name})
    return out


def write_index(data_root: Path, items: list[dict], run_name: str, sources: Iterable[str] = ()) -> dict[str, int]:
    """
    Ghi kho dấu vân tay vòng 1 (D-09): <data_root>/state/dedup_index/<nguồn>/<run_name>.parquet, nguyên tử.

    Mỗi lần chạy là nguồn sự thật cho chính nó: ghi đè file của nó cho mọi nguồn trong `sources` (nguồn không còn văn
    bản nào thì ghi file rỗng), rồi xoá file cùng tên lần chạy của nguồn không còn trong lần chạy (đổi mix). Mọi dòng
    mang written_at của lần ghi này.

    Returns:
        {nguồn: số dòng đã ghi (kể cả bia mộ)}.
    """
    stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    by_source: dict[str, list[dict]] = {k: [] for k in sources}
    for item in items:
        by_source.setdefault(item["source_key"], []).append({**item, "written_at": stamp})
    index_dir = data_root / "state" / "dedup_index"
    stale = [p for p in _index_files(data_root, run_name) if p.parent.name not in by_source]
    for key, part in by_source.items():
        path = index_dir / key / f"{run_name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(pa.Table.from_pylist(part, schema=INDEX_SCHEMA), tmp)
        tmp.replace(path)
    for old in stale:
        old.unlink(missing_ok=True)
    return {k: len(v) for k, v in by_source.items()}


def _index_files(data_root: Path, run_name: str) -> list[Path]:
    """
    Các file kho của lần chạy này (mọi nguồn). Ghép đường dẫn chính xác, không dùng glob: tên lần chạy do người dùng đặt
    (--run-name) có thể chứa '[', '*', '?' và khớp nhầm file của lần chạy khác.
    """
    index_dir = data_root / "state" / "dedup_index"
    if not index_dir.is_dir():
        return []
    return [p for d in sorted(index_dir.iterdir()) if d.is_dir() and (p := d / f"{run_name}.parquet").is_file()]


def has_index(data_root: Path, run_name: str) -> bool:
    """Lần chạy này đã ghi kho dấu vân tay chưa (ít nhất một file)."""
    return bool(_index_files(data_root, run_name))


def remove_index(data_root: Path, run_name: str) -> int:
    """Xoá mọi file kho của lần chạy này (khi 05_dedup.parquet không còn hoặc chạy lại dedup không ghi kho); trả số file."""
    files = _index_files(data_root, run_name)
    for p in files:
        p.unlink(missing_ok=True)
    return len(files)
