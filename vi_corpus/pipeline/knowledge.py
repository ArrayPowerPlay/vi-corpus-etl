"""
Stage 7 (knowledge unit): từ clean corpus sinh các đơn vị tri thức cho CPT / SFT.

- CPT: mỗi bản ghi giữ lại (status = kept) là một đơn vị văn bản thuần (SEA-Pile, đoạn sách).
- SFT: mỗi cuộc hội thoại SEA-Instruct có >= 1 cặp (human, assistant) cho các đơn vị (instruction, response), mỗi cặp liền
  kề là một đơn vị; vai trò "gpt" / "assistant" / "model" là phía trả lời.
- Hybrid: ghép từ hai loại trên ở bước huấn luyện (theo tỷ lệ), nên không sinh riêng ở đây.
Mọi đơn vị mang dedup_family_id của bản ghi gốc; split_bucket (0..99) là băm của dedup_family_id: cắt train / val / test
sau này chỉ cần so split_bucket với ngưỡng, nên các bản gần trùng luôn rơi cùng một phía (không chia ở tầng corpus).
"""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

KU_SCHEMA = pa.schema([
    ("unit_id", pa.string()),
    ("doc_id", pa.string()),
    ("source_key", pa.string()),
    ("dedup_family_id", pa.string()),
    ("split_bucket", pa.int64()),
    ("unit_type", pa.string()),  # "cpt" | "sft"
    ("text", pa.string()),  # cpt: văn bản; sft: instruction
    ("response", pa.string()),  # sft: câu trả lời; cpt: null
    ("token_count", pa.int64()),
])
_ASSISTANT_ROLES = frozenset({"gpt", "assistant", "model", "bot"})


def split_bucket(family: str) -> int:
    """Số 0..99 ổn định từ dedup_family_id, để chia train / val / test theo họ trùng."""
    return int(hashlib.sha1(family.encode("utf-8")).hexdigest()[:8], 16) % 100


def sft_pairs(turns: list[dict]) -> list[tuple[str, str]]:
    """Các cặp (lời hỏi, lời đáp) từ các lượt thoại liền kề: một lượt không phải trợ lý, theo sau là một lượt trợ lý."""
    pairs = []
    for prev, cur in zip(turns, turns[1:]):
        if cur["role"].lower() in _ASSISTANT_ROLES and prev["role"].lower() not in _ASSISTANT_ROLES:
            pairs.append((prev["content"], cur["content"]))
    return pairs


def build_units(rows: list[dict]) -> list[dict]:
    """
    Sinh các knowledge unit từ các bản ghi có status = "kept".

    Hội thoại không có cặp hợp lệ nào thì được dùng như văn bản CPT.
    """
    units = []
    for row in rows:
        if row["status"] != "kept":
            continue
        base = {"doc_id": row["doc_id"], "source_key": row["source_key"], "dedup_family_id": row["dedup_family_id"],
                "split_bucket": split_bucket(row["dedup_family_id"])}
        turns = (json.loads(row["meta"]).get("turns") or []) if row["source_key"] == "sea_instruct_2602" and row["meta"] else []
        pairs = sft_pairs(turns)
        if pairs:
            for i, (q, a) in enumerate(pairs):
                units.append({**base, "unit_id": f"{row['doc_id']}:sft{i}", "unit_type": "sft", "text": q,
                              "response": a, "token_count": len((q + " " + a).split())})
        else:
            units.append({**base, "unit_id": f"{row['doc_id']}:cpt", "unit_type": "cpt", "text": row["text"],
                          "response": None, "token_count": row["token_count"]})
    return units


def write_units(path: Path, units: list[dict]) -> int:
    """Ghi các knowledge unit ra Parquet (KU_SCHEMA), nguyên tử. Trả về số đơn vị."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(units, schema=KU_SCHEMA), tmp)
    tmp.replace(path)
    return len(units)
