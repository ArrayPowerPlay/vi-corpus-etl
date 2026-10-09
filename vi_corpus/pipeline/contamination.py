"""
Quét rò rỉ benchmark (contamination) bằng 13-gram từ, chạy ở stage finalize trên các bản ghi giữ lại.

Danh sách benchmark hiện để TRỐNG (R-16: chưa lọc contamination ở giai đoạn này) nhưng bước quét vẫn chạy được: thả file
vào thư mục RunConfig.benchmarks_dir (mặc định configs/benchmarks/) là lần chạy sau tự quét. Mỗi file là một benchmark:
- *.txt: mỗi dòng một câu hỏi / đoạn văn;
- *.jsonl: mỗi dòng một object, lấy mọi trường kiểu chuỗi (question, context, choices dạng chuỗi, ...).
Nếu sau này thêm VMLU (đánh giá W6) thì phải quét trước khi huấn luyện W6.

Chỉ đếm và đánh dấu trong audit (số bản ghi chạm ít nhất một 13-gram của từng benchmark), không sửa / loại bản ghi.
"""

import hashlib
import json
import re
from pathlib import Path

NGRAM = 13
_WORD = re.compile(r"\w+", re.UNICODE)


def ngram_hashes(text: str, n: int = NGRAM) -> set[int]:
    """Tập băm 8 byte của các n-gram từ (chữ thường, bỏ dấu câu) trong text."""
    words = _WORD.findall(text.lower())
    return {int.from_bytes(hashlib.blake2b(" ".join(words[i:i + n]).encode("utf-8"), digest_size=8).digest(), "little")
            for i in range(len(words) - n + 1)}


def _texts_of(path: Path) -> list[str]:
    """Các đoạn văn bản của một file benchmark (.txt hoặc .jsonl)."""
    if path.suffix == ".txt":
        return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            obj = json.loads(ln)
            out.extend(v for v in (obj.values() if isinstance(obj, dict) else [obj]) if isinstance(v, str))
    return out


def load_benchmarks(bench_dir: Path | None) -> dict[str, set[int]]:
    """{tên benchmark (tên file không đuôi): tập 13-gram}; thư mục không có / rỗng thì trả {}."""
    if bench_dir is None or not bench_dir.is_dir():
        return {}
    out: dict[str, set[int]] = {}
    for path in sorted(list(bench_dir.glob("*.txt")) + list(bench_dir.glob("*.jsonl"))):
        grams: set[int] = set()
        for text in _texts_of(path):
            grams |= ngram_hashes(text)
        out[path.stem] = grams
    return out


def scan(rows: list[dict], benchmarks: dict[str, set[int]]) -> dict:
    """
    Đếm bản ghi giữ lại có chung ít nhất một 13-gram với từng benchmark.

    Returns:
        {"benchmarks": [tên], "ngram": 13, "flagged": {tên: số bản ghi}, "flagged_doc_ids": {tên: tối đa 20 doc_id},
         "note": ghi chú khi danh sách rỗng}.
    """
    result = {"benchmarks": sorted(benchmarks), "ngram": NGRAM, "flagged": {k: 0 for k in benchmarks},
              "flagged_doc_ids": {k: [] for k in benchmarks}}
    if not benchmarks:
        result["note"] = "danh sách benchmark rỗng (R-16): đã chạy bước quét nhưng không có gì để so"
        return result
    for row in rows:
        if row["status"] != "kept":
            continue
        grams = ngram_hashes(row["text"] or "")
        for name, bench in benchmarks.items():
            if grams & bench:
                result["flagged"][name] += 1
                if len(result["flagged_doc_ids"][name]) < 20:
                    result["flagged_doc_ids"][name].append(row["doc_id"])
    return result
