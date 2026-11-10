"""
So ghép đôi hai lần chạy pipeline VJOL (J-40 b, J-42 (1b)): chỉ xét các đoạn có CÙNG `doc_id` ở cả hai lần chạy.

Hai lần chạy khác mẫu 10.000 đoạn thì mức tăng / giảm vài chục đoạn không quy được cho một lần sửa (J-39). So ghép đôi bỏ độ lệch mẫu: cùng một đoạn, trước và sau khi sửa, giữ / loại thế nào,
mã loại đổi ra sao, chữ đổi chỗ nào. Chỉ CPU, đọc `04_quality.parquet` của mỗi lần chạy (nguồn vjol).

- paired_compare: so hai danh sách bản ghi (dict có doc_id, text, quality_band, reason_codes).
- format_paired: bản văn bản để đọc / lưu cạnh kết quả.
- diff_excerpt: trích đoạn quanh chỗ khác nhau đầu tiên của hai văn bản (theo từ), ưu tiên chỗ đổi ở THÂN đoạn, rút gọn đoạn dài.
- edit_kind: phân loại thay đổi giữa hai bản của cùng một đoạn: "same" / "inner" (đổi chữ ở thân) / "edge" (chỉ thừa hoặc thiếu chữ ở đầu / cuối). Đoạn cắt theo số token từ đầu bài nên khi một
  tiêu đề bị cắt hoặc một chữ đổi độ dài, mọi đoạn sau trong bài xê dịch biên; J-44: ở lần so v7 -> v8 thật, 4 / 20 mẫu là xê dịch biên bị đọc nhầm là "xoá cả đoạn văn".
"""

import difflib
import random
from collections import Counter

KEPT_BANDS = ("A", "B", "C")
DIFF_CONTEXT = 6  # số từ ngữ cảnh mỗi bên của chỗ khác nhau trong mẫu
DIFF_MAX_WORDS = 12  # đoạn cũ / mới dài hơn ngần này từ thì rút gọn giữa ("a b c … x y z") để mẫu dễ đọc


def _opcodes(a: list[str], b: list[str]) -> list[tuple[str, int, int, int, int, bool]]:
    """Các thao tác khác nhau (không "equal") của hai danh sách từ, mỗi thao tác kèm cờ "edge": chỉ chèn / xoá và chạm đầu hoặc cuối của CẢ HAI bên (xê dịch biên đoạn)."""
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        edge = tag in ("delete", "insert") and ((i1 == 0 and j1 == 0) or (i2 == len(a) and j2 == len(b)))
        ops.append((tag, i1, i2, j1, j2, edge))
    return ops


def edit_kind(old: str, new: str) -> str:
    """"same" nếu hai bản giống nhau (theo từ); "edge" nếu mọi chỗ khác chỉ là thừa / thiếu chữ ở đầu hoặc cuối đoạn; "inner" nếu có chỗ đổi ở thân."""
    ops = _opcodes(old.split(), new.split())
    if not ops:
        return "same"
    return "inner" if any(not op[5] for op in ops) else "edge"


def _short(words: list[str]) -> str:
    """Nối các từ; quá DIFF_MAX_WORDS từ thì giữ nửa đầu và nửa cuối, thay giữa bằng "…" (chuỗi rỗng thành "∅")."""
    if len(words) > DIFF_MAX_WORDS:
        half = DIFF_MAX_WORDS // 2
        words = words[:half] + ["…"] + words[-half:]
    return " ".join(words) or "∅"


def diff_excerpt(old: str, new: str, context: int = DIFF_CONTEXT) -> str:
    """
    Trích chỗ khác nhau đầu tiên của hai văn bản, so theo từ: "…ngữ cảnh [cũ -> mới] ngữ cảnh…", ưu tiên chỗ đổi ở thân (không phải xê dịch biên); chỉ có xê dịch biên thì trích chỗ đó
    và thêm "(biên)". Hai văn bản giống nhau trả về chuỗi rỗng. Đoạn cũ / mới dài được rút gọn.
    """
    a, b = old.split(), new.split()
    ops = _opcodes(a, b)
    if not ops:
        return ""
    _tag, i1, i2, j1, j2, edge = next((op for op in ops if not op[5]), ops[0])
    left = " ".join(a[max(0, i1 - context):i1])
    right = " ".join(a[i2:i2 + context])
    return f"…{left} [{_short(a[i1:i2])} -> {_short(b[j1:j2])}] {right}…" + (" (biên)" if edge else "")


def _status(row: dict) -> str:
    """"kept" nếu đoạn thuộc band giữ lại (A / B / C), ngược lại "dropped"."""
    return "kept" if row["quality_band"] in KEPT_BANDS else "dropped"


def paired_compare(old_rows: list[dict], new_rows: list[dict], samples: int = 20, seed: int = 0) -> dict:
    """
    So ghép đôi hai lần chạy theo `doc_id` (xem docstring module).

    Args:
        old_rows, new_rows: Bản ghi vjol của lần chạy cũ / mới (cần doc_id, text, quality_band, reason_codes).
        samples:            Số đoạn đổi chữ lấy ngẫu nhiên để đọc.
        seed:               Hạt giống chọn mẫu.

    Returns:
        {"old_chunks", "new_chunks", "common", "only_old", "only_new",
         "text_changed": số đoạn chung đổi chữ, gồm "text_changed_inner" (đổi ở thân) + "text_changed_edge" (chỉ xê dịch biên đoạn), "kept_both", "dropped_both", "kept_to_dropped", "dropped_to_kept",
         "codes_gained": {mã: số đoạn chung được thêm mã}, "codes_lost": {mã: số đoạn chung mất mã},
         "transitions": {"kept->dropped": {mã mới: số đoạn}, "dropped->kept": {mã cũ: số đoạn}},
         "samples": [{doc_id, status_old, status_new, codes_old, codes_new, diff}]} (chỉ đoạn đổi chữ ở thân).
    """
    old = {r["doc_id"]: r for r in old_rows}
    new = {r["doc_id"]: r for r in new_rows}
    common = sorted(set(old) & set(new))
    kind = {d: edit_kind(old[d]["text"] or "", new[d]["text"] or "") for d in common}
    changed = [d for d in common if kind[d] != "same"]
    inner = [d for d in changed if kind[d] == "inner"]
    gained: Counter = Counter()
    lost: Counter = Counter()
    went_down: Counter = Counter()
    went_up: Counter = Counter()
    both = Counter()
    for d in common:
        o, n = old[d], new[d]
        oc, nc = set(o.get("reason_codes") or []), set(n.get("reason_codes") or [])
        gained.update(nc - oc)
        lost.update(oc - nc)
        so, sn = _status(o), _status(n)
        both[(so, sn)] += 1
        if (so, sn) == ("kept", "dropped"):
            went_down.update(nc - oc or nc)
        elif (so, sn) == ("dropped", "kept"):
            went_up.update(oc - nc or oc)
    picked = random.Random(seed).sample(inner, min(samples, len(inner)))
    return {
        "old_chunks": len(old), "new_chunks": len(new), "common": len(common), "only_old": len(set(old) - set(new)), "only_new": len(set(new) - set(old)),
        "text_changed": len(changed), "text_changed_inner": len(inner), "text_changed_edge": len(changed) - len(inner), "kept_both": both[("kept", "kept")], "dropped_both": both[("dropped", "dropped")],
        "kept_to_dropped": both[("kept", "dropped")], "dropped_to_kept": both[("dropped", "kept")],
        "codes_gained": dict(gained.most_common()), "codes_lost": dict(lost.most_common()),
        "transitions": {"kept->dropped": dict(went_down.most_common()), "dropped->kept": dict(went_up.most_common())},
        "samples": [{"doc_id": d, "status_old": _status(old[d]), "status_new": _status(new[d]), "codes_old": list(old[d].get("reason_codes") or []),
                     "codes_new": list(new[d].get("reason_codes") or []), "diff": diff_excerpt(old[d]["text"] or "", new[d]["text"] or "")} for d in sorted(picked)],
    }


def format_paired(res: dict, old_name: str = "cũ", new_name: str = "mới") -> str:
    """Bản văn bản của kết quả paired_compare: số đoạn chung, đổi trạng thái (kèm mã), đổi chữ và các mẫu để đọc."""
    n = res["common"] or 1
    lines = [f"So ghép đôi {old_name} -> {new_name} (cùng doc_id)", f"- đoạn: cũ {res['old_chunks']}, mới {res['new_chunks']}, chung {res['common']}, chỉ ở cũ {res['only_old']}, chỉ ở mới {res['only_new']}",
             f"- đoạn chung đổi chữ: {res['text_changed']} ({res['text_changed'] / n:.2%}), trong đó đổi ở thân {res['text_changed_inner']} ({res['text_changed_inner'] / n:.2%}), chỉ xê dịch biên {res['text_changed_edge']}",
             f"- giữ cả hai {res['kept_both']}, loại cả hai {res['dropped_both']}, giữ -> loại {res['kept_to_dropped']}, loại -> giữ {res['dropped_to_kept']}",
             f"- mã được thêm: {res['codes_gained'] or '-'}", f"- mã mất: {res['codes_lost'] or '-'}",
             f"- giữ -> loại, theo mã mới: {res['transitions']['kept->dropped'] or '-'}", f"- loại -> giữ, theo mã cũ: {res['transitions']['dropped->kept'] or '-'}", "", f"## {len(res['samples'])} mẫu đổi chữ ở thân (ngẫu nhiên, để đọc)"]
    for s in res["samples"]:
        lines.append(f"- `{s['doc_id']}` {s['status_old']}->{s['status_new']} {s['codes_old']}->{s['codes_new']}: {s['diff']}")
    return "\n".join(lines)
