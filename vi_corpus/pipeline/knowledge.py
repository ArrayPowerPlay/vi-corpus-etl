"""
Stage finalize (knowledge unit): từ clean corpus sinh các đơn vị tri thức cho CPT / SFT.

Thuật ngữ (R-09): KU loại "cpt" = một candidate (một đoạn CPT còn sống sau mọi bước lọc và dedup, có đủ provenance; các
chỉ tiêu "≥100k / ≥500k / ≥1,2M candidates" đếm đơn vị này); KU loại "sft" = một instruction (cặp hỏi - đáp).

- CPT: mỗi bản ghi giữ lại (status = kept) là một đơn vị văn bản thuần (SEA-Pile, đoạn sách).
- SFT: mỗi cuộc hội thoại SEA-Instruct có >= 1 cặp (human, assistant) cho các đơn vị (instruction, response), mỗi cặp liền
  kề là một đơn vị; vai trò "gpt" / "assistant" / "model" là phía trả lời.
- Hybrid: ghép từ hai loại trên ở bước huấn luyện (theo tỷ lệ), nên không sinh riêng ở đây.
Mọi đơn vị mang dedup_family_id của bản ghi gốc; split_bucket (0..99) là băm của dedup_family_id: cắt train / val / test
sau này chỉ cần so split_bucket với ngưỡng, nên các bản gần trùng luôn rơi cùng một phía (không chia ở tầng corpus).
Theo D-09, knowledge unit chính thức sinh SAU dedup vòng 2 (RunConfig.global_verdict); không có verdict thì họ trùng
chỉ là của lần chạy này.

Khối CPT (D-10, pack_cpt_blocks): khi xuất dữ liệu huấn luyện CPT, các đoạn của cùng một văn bản được nhóm theo
parent_doc_id, sắp theo chunk_index, rồi nối các đoạn LIỀN NHAU còn giữ cho tới độ dài ngữ cảnh; gặp đoạn bị loại (hoặc
đoạn không có trong lần chạy vì lấy mẫu) thì ngắt khối.
"""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vi_corpus.pipeline.tokens import TokenCounter

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
BLOCK_SCHEMA = pa.schema([
    ("block_id", pa.string()),
    ("parent_doc_id", pa.string()),
    ("source_key", pa.string()),
    ("dedup_family_id", pa.string()),
    ("split_bucket", pa.int64()),
    ("doc_ids", pa.list_(pa.string())),  # các đoạn đã nối, theo thứ tự
    ("text", pa.string()),
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


def build_units(rows: list[dict], counter: TokenCounter) -> list[dict]:
    """
    Sinh các knowledge unit từ các bản ghi có status = "kept".

    Hội thoại không có cặp hợp lệ nào thì được dùng như văn bản CPT. token_count của SFT đếm bằng cùng bộ đếm token
    với corpus (D-01).
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
            counts = counter.count_batch([q + "\n" + a for q, a in pairs])
            for i, ((q, a), n) in enumerate(zip(pairs, counts)):
                units.append({**base, "unit_id": f"{row['doc_id']}:sft{i}", "unit_type": "sft", "text": q,
                              "response": a, "token_count": n})
        else:
            units.append({**base, "unit_id": f"{row['doc_id']}:cpt", "unit_type": "cpt", "text": row["text"],
                          "response": None, "token_count": row["token_count"]})
    return units


def pack_cpt_blocks(rows: list[dict], max_tokens: int) -> list[dict]:
    """
    Nối các đoạn CPT liền nhau còn giữ của cùng văn bản thành khối <= max_tokens (xem docstring module).

    Bỏ qua hội thoại SEA-Instruct (là SFT). Văn bản không bị cắt (không có parent_doc_id) là một khối riêng.
    Đoạn đơn lẻ dài hơn max_tokens vẫn thành một khối riêng (không cắt lại). Khối mang dedup_family_id của đoạn đầu.
    """
    by_parent: dict[str, list[dict]] = {}
    for row in rows:
        if row["source_key"] == "sea_instruct_2602":
            continue
        by_parent.setdefault(row.get("parent_doc_id") or row["doc_id"], []).append(row)
    blocks: list[dict] = []

    def flush(parent: str, cur: list[dict]) -> None:
        """Ghi khối đang gom (nếu có) vào danh sách."""
        if cur:
            fam = cur[0]["dedup_family_id"]
            blocks.append({"block_id": f"{parent}:b{len(blocks)}", "parent_doc_id": parent,
                           "source_key": cur[0]["source_key"], "dedup_family_id": fam, "split_bucket": split_bucket(fam),
                           "doc_ids": [r["doc_id"] for r in cur], "text": "\n\n".join(r["text"] for r in cur),
                           "token_count": sum(r["token_count"] or 0 for r in cur)})

    for parent, group in by_parent.items():
        group.sort(key=lambda r: r.get("chunk_index") or 0)
        cur: list[dict] = []
        prev_idx = None
        for row in group:
            idx = row.get("chunk_index") or 0
            contiguous = prev_idx is not None and idx == prev_idx + 1
            prev_idx = idx
            if row["status"] != "kept":
                flush(parent, cur)
                cur = []
                continue
            if cur and (not contiguous or sum(r["token_count"] or 0 for r in cur) + (row["token_count"] or 0) > max_tokens):
                flush(parent, cur)
                cur = []
            cur.append(row)
        flush(parent, cur)
    return blocks


def _write(path: Path, items: list[dict], schema: pa.Schema) -> int:
    """Ghi các dòng ra Parquet theo schema, nguyên tử. Trả về số dòng."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(items, schema=schema), tmp)
    tmp.replace(path)
    return len(items)


def write_units(path: Path, units: list[dict]) -> int:
    """Ghi các knowledge unit ra Parquet (KU_SCHEMA), nguyên tử. Trả về số đơn vị."""
    return _write(path, units, KU_SCHEMA)


def write_blocks(path: Path, blocks: list[dict]) -> int:
    """Ghi các khối CPT ra Parquet (BLOCK_SCHEMA), nguyên tử. Trả về số khối."""
    return _write(path, blocks, BLOCK_SCHEMA)
