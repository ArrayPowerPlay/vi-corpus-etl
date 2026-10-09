"""
Dedup vòng 2 (D-09): gộp kho dấu vân tay của MỌI nguồn, tìm họ trùng toàn cục, ghi verdict.parquet.

Vòng 1 (stage dedup của mỗi lần chạy, vi_corpus.pipeline.dedup) đã bỏ trùng trong lần chạy và ghi dấu vân tay mọi văn
bản của lần chạy vào <data_root>/state/dedup_index/<nguồn>/<run>.parquet (doc_id, sha256, MinHash, điểm, quyền; văn bản
bị loại ghi thành bia mộ alive=False; không có text). Vòng 2 chỉ đọc kho đó nên rẻ; thêm nguồn mới thì chạy vòng 1 cho
nguồn đó rồi chạy lại vòng 2.

- Nhóm "cpt" dedup toàn cục giữa mọi nguồn; nhóm "sft" chỉ trong nhóm của mình (Q4).
- Cùng văn bản xuất hiện ở nhiều lần chạy thì lấy dòng ghi MỚI NHẤT (cột written_at, không theo tên file); dòng mới nhất
  là bia mộ thì văn bản không vào vòng 2 (lần chạy gần nhất đã loại nó).
- Bản giữ: ưu tiên nguồn (R-15) -> điểm rule -> doc_id, giống vòng 1. Ưu tiên tính LẠI ở vòng 2 từ source_key theo
  bảng ưu tiên truyền vào (mặc định SOURCE_PRIORITY hiện tại), không dùng số đã ghi lúc chạy vòng 1.
- Họ trùng nối từ các cặp chắc: cùng sha256, hoặc Jaccard MinHash >= ngưỡng (R-19 mục 4).
- Chỉ ĐÁNH DẤU (R-01): verdict.parquet ghi status "kept" / "duplicate" / "rights" cho mọi văn bản, không xoá text ở
  interim/ hay ở kết quả của các lần chạy. rights_gate="enforce" thì văn bản có rights_status ngoài OPEN_RIGHTS bị đánh
  "rights" và không được chọn làm bản giữ; mặc định "tag" (R-01). Đổi ưu tiên / bật rights gate thì chỉ cần chạy lại
  vòng 2 rồi áp lại verdict.

Áp verdict vào bản ghi của một lần chạy: apply_verdict (runner gọi khi có RunConfig.global_verdict, trước khi sinh
knowledge unit, vì knowledge unit và chia train/test làm sau vòng 2).

Giới hạn: toàn bộ chữ ký được nạp vào RAM (512 byte / văn bản: 10 triệu văn bản ~ 5 GB); khi kho lớn hơn thì cần chia
theo băng LSH trên đĩa (D-06, chưa làm).
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vi_corpus.pipeline.config import OPEN_RIGHTS
from vi_corpus.pipeline.dedup import dedup_docs, family_id, sig_from_bytes

logger = logging.getLogger("vi_corpus")

VERDICT_SCHEMA = pa.schema([
    ("doc_id", pa.string()),
    ("source_key", pa.string()),
    ("dedup_group", pa.string()),
    ("status", pa.string()),  # "kept" | "duplicate" | "rights" (rights_gate="enforce") | "dropped" (lần chạy mới nhất đã loại)
    ("dup_kind", pa.string()),  # "exact" | "fuzzy" | null
    ("dup_of", pa.string()),  # doc_id văn bản đại diện của họ (null nếu chính nó là đại diện)
    ("dedup_family_id", pa.string()),
    ("family_size", pa.int64()),
])


def read_index(index_dir: Path, with_dropped: bool = False) -> list[dict]:
    """
    Đọc mọi file kho <index_dir>/<nguồn>/*.parquet; mỗi (nhóm, doc_id) lấy dòng có written_at mới nhất (bằng nhau thì
    file đọc sau, theo tên). Văn bản mà dòng mới nhất là bia mộ (alive=False) bị bỏ, trừ khi with_dropped (để verdict
    ghi chúng là "dropped").
    """
    best: dict[tuple[str, str], tuple[tuple, dict]] = {}
    for order, path in enumerate(sorted(index_dir.glob("*/*.parquet"))):
        for row in pq.read_table(path).to_pylist():
            key = (row["dedup_group"], row["doc_id"])
            rank = (row.get("written_at") or "", order)
            if key not in best or rank >= best[key][0]:
                best[key] = (rank, row)
    return [row for _, row in best.values() if with_dropped or row.get("alive", True)]


def global_verdict(items: list[dict], threshold: float, priority_of: Callable[[str], int] | None = None,
                   rights_gate: str = "tag") -> tuple[list[dict], dict]:
    """
    Tính verdict toàn cục từ các dòng kho dấu vân tay.

    Args:
        items:       Dòng kho (read_index); dòng bia mộ (alive=False) ra status "dropped", không vào dedup.
        threshold:   Ngưỡng Jaccard MinHash.
        priority_of: Hàm source_key -> mức ưu tiên (nhỏ = giữ trước); None thì dùng số đã ghi trong kho.
        rights_gate: "tag" (chỉ gắn nhãn, R-01) hoặc "enforce" (văn bản quyền chưa rõ không vào dedup, status "rights").

    Returns:
        (các dòng VERDICT_SCHEMA, thống kê theo nhóm: số văn bản, số bị trùng theo loại, cỡ họ lớn nhất, số bị chặn quyền).
    """
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(it["dedup_group"], []).append(it)
    verdict: list[dict] = []
    stats: dict[str, dict] = {}
    for group, members in groups.items():
        dropped = [m for m in members if m.get("alive", True) is False]
        members = [m for m in members if m.get("alive", True) is not False]
        blocked = [m for m in members if rights_gate == "enforce" and m.get("rights_status") not in OPEN_RIGHTS]
        for status, part in (("dropped", dropped), ("rights", blocked)):
            for m in part:
                verdict.append({"doc_id": m["doc_id"], "source_key": m["source_key"], "dedup_group": group,
                                "status": status, "dup_kind": None, "dup_of": None,
                                "dedup_family_id": family_id(m["doc_id"]), "family_size": 1})
        blocked_ids = {m["doc_id"] for m in blocked}
        docs = sorted(({"doc_id": m["doc_id"], "source_key": m["source_key"], "sha": m["doc_sha256"],
                        "sig": sig_from_bytes(m["minhash"]),
                        "priority": priority_of(m["source_key"]) if priority_of else m["priority"],
                        "score": m["quality_score"]}
                       for m in members if m["doc_id"] not in blocked_ids),
                      key=lambda d: (d["priority"], -d["score"], d["doc_id"]))
        rep, kinds = dedup_docs(docs, threshold)
        sizes = np.bincount(np.array(rep, dtype=np.int64), minlength=len(docs)) if docs else np.zeros(0, dtype=np.int64)
        for i, d in enumerate(docs):
            dup = rep[i] != i
            verdict.append({"doc_id": d["doc_id"], "source_key": d["source_key"], "dedup_group": group,
                            "status": "duplicate" if dup else "kept", "dup_kind": kinds.get(i, "fuzzy") if dup else None,
                            "dup_of": docs[rep[i]]["doc_id"] if dup else None,
                            "dedup_family_id": family_id(docs[rep[i]]["doc_id"]), "family_size": int(sizes[rep[i]])})
        dup_rows = [v for v in verdict if v["dedup_group"] == group and v["status"] == "duplicate"]
        by_source: dict[str, int] = {}
        for v in dup_rows:
            by_source[v["source_key"]] = by_source.get(v["source_key"], 0) + 1
        stats[group] = {"docs": len(members), "exact": sum(v["dup_kind"] == "exact" for v in dup_rows),
                        "fuzzy": sum(v["dup_kind"] == "fuzzy" for v in dup_rows), "rights_blocked": len(blocked), "dropped": len(dropped),
                        "duplicates_by_source": by_source, "largest_family": int(sizes.max()) if len(sizes) else 0}
    return verdict, stats


def write_verdict(path: Path, verdict: list[dict]) -> int:
    """Ghi verdict.parquet nguyên tử. Trả về số dòng."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(verdict, schema=VERDICT_SCHEMA), tmp)
    tmp.replace(path)
    return len(verdict)


def apply_verdict(rows: list[dict], verdict_path: Path) -> dict:
    """
    Áp verdict vòng 2 vào bản ghi của một lần chạy (sửa tại chỗ): đoạn đang "kept" thuộc văn bản bị trùng toàn cục thành
    "rejected:duplicate" (dup_kind "global_exact" / "global_fuzzy"), thuộc văn bản bị chặn quyền (status "rights") thành
    "rejected:rights"; mọi đoạn có trong verdict nhận dedup_family_id toàn cục (để chia train/test theo họ trùng giữa các
    nguồn). Văn bản "dropped" (lần chạy mới hơn đã loại nó) giữ nguyên và được đếm riêng: chỉ nên finalize lần chạy mới
    nhất của mỗi nguồn với --global-verdict.

    Returns:
        Thống kê: số đoạn bị loại thêm, số đoạn đổi họ, số đoạn không có trong verdict (chưa chạy lại vòng 2), số đoạn
        thuộc văn bản mà lần chạy mới hơn đã loại.
    """
    table = pq.read_table(verdict_path, columns=["doc_id", "status", "dup_kind", "dup_of", "dedup_family_id"])
    by_id = {r["doc_id"]: r for r in table.to_pylist()}
    rejected = refamily = missing = superseded = 0
    for row in rows:
        if row["status"] != "kept":
            continue
        v = by_id.get(row.get("parent_doc_id") or row["doc_id"])
        if v is None:
            missing += 1
            continue
        if v["status"] == "dropped":
            superseded += 1
            continue
        if row["dedup_family_id"] != v["dedup_family_id"]:
            row["dedup_family_id"] = v["dedup_family_id"]
            refamily += 1
        if v["status"] == "duplicate":
            row.update(status="rejected:duplicate", dup_kind=f"global_{v['dup_kind']}", dup_of=v["dup_of"])
            rejected += 1
        elif v["status"] == "rights":
            row["status"] = "rejected:rights"
            rejected += 1
    if missing:
        logger.warning("apply_verdict: %d đoạn còn sống không có trong verdict (chạy lại scripts/dedup_global.py?)",
                       missing)
    if superseded:
        logger.warning("apply_verdict: %d đoạn còn sống ở đây nhưng lần chạy mới hơn đã loại văn bản của chúng (lần chạy "
                       "này cũ hơn; nên finalize lần chạy mới nhất của nguồn)", superseded)
    return {"rejected": rejected, "refamily": refamily, "missing": missing, "superseded": superseded}
