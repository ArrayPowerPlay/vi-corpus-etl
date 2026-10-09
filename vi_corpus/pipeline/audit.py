"""
Stage finalize (audit): đếm KPI từ kết quả cuối, ghi audit.json.

- Lineage (KPI >= 95%): tỷ lệ bản ghi có đủ source_key / source_path / source_sha256 (schema.lineage_rate).
- Funnel theo nguồn: số mẫu vào -> sau từng cổng (quality, rights, duplicate) -> giữ lại.
- Token accounting: tổng từ / token (tokenizer chính thức, D-01) của bản ghi vào và giữ lại, tỷ lệ token / từ theo nguồn;
  tuỳ chọn tổng token của phần giữ lại theo các tokenizer khác (để thấy KPI phụ thuộc tokenizer, M-01).
- Quét thông tin cá nhân (PII): email, số điện thoại VN trong các bản ghi giữ lại (chỉ đếm, không sửa text).
- Quét rò rỉ benchmark 13-gram (vi_corpus.pipeline.contamination); danh sách benchmark hiện rỗng (R-16).
"""

import re
from collections import Counter

from vi_corpus.common.schema import lineage_rate

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_VN = re.compile(r"(?<!\d)(?:\+84|0)(?:3|5|7|8|9)\d{8}(?!\d)")


def audit_rows(rows: list[dict], tokenizer: str = "words", contamination: dict | None = None,
               compare: dict[str, int] | None = None) -> dict:
    """
    Tính các số liệu kiểm toán của một lần chạy từ mọi bản ghi (đã có status).

    Args:
        rows:          Mọi bản ghi.
        tokenizer:     Tên tokenizer đã đếm token_count.
        contamination: Kết quả contamination.scan (None = chưa chạy).
        compare:       {tokenizer khác: tổng token của phần giữ lại}.

    Returns:
        dict JSON được: lineage, funnel, tokenizer, tokens, tokens_kept_other_tokenizers, pii, contamination.
    """
    funnel: dict[str, Counter] = {}
    tokens: dict[str, dict] = {}
    pii = Counter()
    for r in rows:
        f = funnel.setdefault(r["source_key"], Counter())
        f["input"] += 1
        f[r["status"]] += 1
        t = tokens.setdefault(r["source_key"], {"words_in": 0, "tokens_in": 0, "words_kept": 0, "tokens_kept": 0})
        t["words_in"] += r["word_count"] or 0
        t["tokens_in"] += r["token_count"] or 0
        if r["status"] == "kept":
            t["words_kept"] += r["word_count"] or 0
            t["tokens_kept"] += r["token_count"] or 0
            pii["docs_with_email"] += bool(_EMAIL.search(r["text"]))
            pii["docs_with_phone"] += bool(_PHONE_VN.search(r["text"]))
            pii["kept"] += 1
    for t in tokens.values():
        t["tokens_per_word_kept"] = round(t["tokens_kept"] / t["words_kept"], 3) if t["words_kept"] else None
    kept = [r for r in rows if r["status"] == "kept"]
    return {
        "lineage_rate_all": lineage_rate(rows),
        "lineage_rate_kept": lineage_rate(kept),
        "lineage_target": 0.95,
        "funnel": {k: dict(v) for k, v in funnel.items()},
        "tokenizer": tokenizer,
        "tokens": tokens,
        "tokens_kept_other_tokenizers": compare or {},
        "pii": dict(pii),
        "contamination": contamination or {"note": "chưa chạy"},
    }
