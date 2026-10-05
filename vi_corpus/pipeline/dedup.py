"""
Stage 6 (dedup + rights gate): loại trùng lặp chính xác và gần trùng, gắn cổng quyền, chốt trạng thái từng bản ghi.

Thứ tự (docs/PIPELINE.md): chỉ bản ghi đã qua quality mới vào dedup, bản có điểm chất lượng cao nhất được giữ làm đại diện.
- exact: cùng sha256 của text đã chuẩn hóa.
- fuzzy: MinHash (128 hoán vị, shingle 5 từ), LSH 16 băng x 8 hàng, xác nhận bằng Jaccard ước lượng >= ngưỡng.
  Thuần numpy, chạy CPU; quy mô lớn thì thay bằng adapter NeMo Curator (cùng đầu vào / đầu ra).
- Mỗi bản ghi còn sống thuộc một họ trùng; dedup_family_id là mã băm của doc_id đại diện. Chia train/val/test sau này
  luôn chia theo mã này để không rò rỉ giữa các bản gần trùng.
- Cổng quyền: rights_status không thuộc OPEN_RIGHTS thì rights_gate = "quarantine"; chế độ "enforce" loại luôn.
"""

import hashlib
import zlib

import numpy as np

from vi_corpus.common.schema import text_sha256
from vi_corpus.pipeline.config import OPEN_RIGHTS, RunConfig

NUM_PERM, BANDS, ROWS = 128, 16, 8  # BANDS * ROWS == NUM_PERM; ngưỡng LSH ~ (1/16)^(1/8) ~ 0.71
SHINGLE = 5
MAX_WORDS = 20_000  # chỉ băm phần đầu văn bản rất dài
MIN_FUZZY_WORDS = 20
_MASK = np.uint64(0xFFFFFFFF)
_rng = np.random.default_rng(12345)  # hằng số cố định: chữ ký MinHash tái lập được giữa các lần chạy
_A = (_rng.integers(1, 1 << 31, NUM_PERM, dtype=np.uint64) * 2 + 1).astype(np.uint64)
_B = _rng.integers(0, 1 << 32, NUM_PERM, dtype=np.uint64)


def minhash(text: str) -> np.ndarray | None:
    """
    Chữ ký MinHash (NUM_PERM giá trị uint64) của text, theo shingle SHINGLE từ.

    Returns:
        None nếu text quá ngắn (< MIN_FUZZY_WORDS từ) để so gần trùng.
    """
    words = text.lower().split()[:MAX_WORDS]
    if len(words) < MIN_FUZZY_WORDS:
        return None
    n = max(len(words) - SHINGLE + 1, 1)
    x = np.fromiter((zlib.crc32(" ".join(words[i:i + SHINGLE]).encode("utf-8")) for i in range(n)),
                    dtype=np.uint64, count=n)
    return ((_A[:, None] * x[None, :] + _B[:, None]) & _MASK).min(axis=1)


class _UnionFind:
    """Union-find tối giản trên chỉ số nguyên, dùng để gộp các cặp gần trùng thành họ."""

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
        """Gộp hai phần tử vào cùng một họ."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def find_fuzzy_pairs(sigs: list[np.ndarray | None], threshold: float) -> list[tuple[int, int]]:
    """
    Tìm các cặp chỉ số (i, j) gần trùng: cùng rơi vào một bucket LSH và Jaccard ước lượng >= threshold.

    Chữ ký None (văn bản quá ngắn) bị bỏ qua.
    """
    buckets: dict[tuple[int, bytes], list[int]] = {}
    for i, sig in enumerate(sigs):
        if sig is None:
            continue
        for b in range(BANDS):
            buckets.setdefault((b, sig[b * ROWS:(b + 1) * ROWS].tobytes()), []).append(i)
    seen: set[tuple[int, int]] = set()
    for members in buckets.values():
        for k, i in enumerate(members):
            for j in members[k + 1:]:
                if (i, j) not in seen:
                    seen.add((i, j))
    return [(i, j) for i, j in seen if float(np.mean(sigs[i] == sigs[j])) >= threshold]


def family_id(rep_doc_id: str) -> str:
    """Mã họ trùng: băm ngắn của doc_id đại diện (ổn định giữa các lần chạy)."""
    return "fam_" + hashlib.sha1(rep_doc_id.encode("utf-8")).hexdigest()[:12]


def dedup_rows(rows: list[dict], cfg: RunConfig) -> tuple[list[dict], dict]:
    """
    Chốt cột status, rights_gate, dedup_family_id, dup_kind, dup_of cho mọi bản ghi (sửa tại chỗ).

    status: "rejected:quality" (band ngoài cfg.keep_bands), "rejected:rights" (chỉ khi rights_gate="enforce"),
    "rejected:duplicate" (bản kém hơn trong họ trùng), còn lại "kept".

    Returns:
        (rows, thống kê: số bản ghi theo status, số bản trùng chính xác / gần trùng, số họ trùng có > 1 thành viên).
    """
    for row in rows:
        row["rights_gate"] = "pass" if row["rights_status"] in OPEN_RIGHTS else "quarantine"
        if row["quality_band"] not in cfg.keep_bands:
            row["status"] = "rejected:quality"
        elif cfg.rights_gate == "enforce" and row["rights_gate"] == "quarantine":
            row["status"] = "rejected:rights"
        else:
            row["status"] = "kept"

    alive = sorted((r for r in rows if r["status"] == "kept"), key=lambda r: (-r["quality_score"], r["doc_id"]))
    uf = _UnionFind(len(alive))
    kinds: dict[int, str] = {}
    first_by_hash: dict[str, int] = {}
    for i, row in enumerate(alive):
        h = text_sha256(row["text"])
        if h in first_by_hash:
            uf.union(first_by_hash[h], i)
            kinds[i] = "exact"
        else:
            first_by_hash[h] = i
    sigs = [minhash(r["text"]) for r in alive]
    for i, j in sorted(find_fuzzy_pairs(sigs, cfg.fuzzy_threshold)):
        if uf.find(i) != uf.find(j):
            kinds.setdefault(max(i, j), "fuzzy")
        uf.union(min(i, j), max(i, j))  # chỉ số nhỏ = điểm cao hơn; gốc luôn là bản tốt nhất vì union(min, max)

    reps = {i: uf.find(i) for i in range(len(alive))}
    n_exact = n_fuzzy = 0
    for i, row in enumerate(alive):
        rep = alive[reps[i]]
        row["dedup_family_id"] = family_id(rep["doc_id"])
        if reps[i] != i:
            row["status"] = "rejected:duplicate"
            row["dup_kind"] = kinds.get(i, "fuzzy")
            row["dup_of"] = rep["doc_id"]
            n_exact += row["dup_kind"] == "exact"
            n_fuzzy += row["dup_kind"] == "fuzzy"
    families = {}
    for i in reps:
        families[reps[i]] = families.get(reps[i], 0) + 1
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    return rows, {"statuses": statuses, "exact_dups": n_exact, "fuzzy_dups": n_fuzzy,
                  "families_multi": sum(v > 1 for v in families.values()),
                  "quarantined": sum(r["rights_gate"] == "quarantine" for r in rows)}
