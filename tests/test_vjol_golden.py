"""
Ca thử vàng cho các bộ dò và phép sửa chữ của VJOL (J-42 (1e), P7): tests/data/vjol_golden.jsonl.

Mỗi dòng của tệp là một ca: {"_vocab": "..."} (một dòng, các âm tiết thêm vào tập âm tiết giả), hoặc
- kind "bug":    {"id", "text", "detectors": [...]}: mọi bộ dò liệt kê PHẢI gắn cờ lên text (ca lỗi cần bắt);
- kind "keep":   {"id", "text", "forbid": [...]}: không bộ dò nào trong `forbid` được gắn cờ (ca cấm đổi);
- kind "repair": {"id", "src", "want"}: restore_u_in_text(src) phải ra đúng `want` (want == src nghĩa là phải giữ nguyên, kể cả ca mơ hồ).
Tệp sinh ra từ nhãn chuẩn bằng scripts/vjol/make_golden.py (đoạn chỉ rút các câu cần thiết); tệp gốc trong repo là bộ hạt giống viết tay từ các ví dụ ở docs/DECISION_LOG.md. Test bỏ qua khi thiếu tệp.
Ngoài kiểm từng ca còn có sàn precision / recall của từng bộ dò, để sửa đổi sau này không làm tụt các bộ dò. Không cần mạng, GPU hay mô hình.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from vi_corpus.vjol import audit as A
from vi_corpus.vjol.hyphen_fix import restore_u_in_text

GOLDEN = Path(__file__).parent / "data" / "vjol_golden.jsonl"
MIN_RECALL = 0.8  # sàn recall của mỗi bộ dò trên ca "bug"
MIN_PRECISION = 0.9  # sàn precision của mỗi bộ dò (ca "keep" không được bị gắn cờ nhầm)

pytestmark = pytest.mark.skipif(not GOLDEN.is_file(), reason="thiếu tests/data/vjol_golden.jsonl")


def load_cases() -> tuple[frozenset[str], list[dict]]:
    """Đọc tệp ca thử vàng: (tập âm tiết giả, danh sách ca). Tập âm tiết gồm mọi từ của câu `want` / ca "keep" và các từ ở dòng `_vocab`."""
    cases = [json.loads(ln) for ln in GOLDEN.read_text(encoding="utf-8").splitlines() if ln.strip()]
    words: set[str] = set()
    for c in cases:
        if "_vocab" in c:
            words |= set(c["_vocab"].lower().split())
    vocab = frozenset(words | {w.lower().strip(".,;:“”()") for c in cases if c.get("kind") == "repair" for w in c["want"].split()})
    return vocab, [c for c in cases if "_vocab" not in c]


VOCAB, CASES = load_cases()


def test_tep_ca_thu_vang_hop_le():
    """Mỗi ca có id duy nhất và đúng khoá theo kind; có đủ ba loại ca."""
    assert len({c["id"] for c in CASES}) == len(CASES)
    need = {"bug": {"text", "detectors"}, "keep": {"text", "forbid"}, "repair": {"src", "want"}}
    for c in CASES:
        assert c["kind"] in need and need[c["kind"]] <= set(c), c["id"]
    assert {c["kind"] for c in CASES} == set(need)


@pytest.mark.parametrize("case", [c for c in CASES if c["kind"] == "bug"], ids=lambda c: c["id"])
def test_ca_loi_phai_bi_bat(case):
    """Ca lỗi cần bắt: mọi bộ dò liệt kê gắn cờ lên text."""
    assert set(case["detectors"]) <= A.detect_all(case["text"], VOCAB), case["source"]


@pytest.mark.parametrize("case", [c for c in CASES if c["kind"] == "keep"], ids=lambda c: c["id"])
def test_ca_cam_doi_khong_bi_gan_co(case):
    """Ca cấm đổi: không bộ dò nào trong `forbid` được gắn cờ."""
    assert not set(case["forbid"]) & A.detect_all(case["text"], VOCAB), case["source"]


@pytest.mark.parametrize("case", [c for c in CASES if c["kind"] == "repair"], ids=lambda c: c["id"])
def test_ca_sua_chu(case):
    """Ca sửa chữ: restore_u_in_text ra đúng `want` (ca cấm đổi và ca mơ hồ giữ nguyên)."""
    assert restore_u_in_text(case["src"], VOCAB)[0] == case["want"], case["source"]


def test_san_precision_recall_tung_bo_do():
    """Recall của mỗi bộ dò trên ca "bug" >= MIN_RECALL và precision (ca "keep" bị gắn cờ nhầm) >= MIN_PRECISION."""
    tp, fn, fp = Counter(), Counter(), Counter()
    for c in CASES:
        if c["kind"] == "bug":
            got = A.detect_all(c["text"], VOCAB)
            for d in c["detectors"]:
                (tp if d in got else fn)[d] += 1
        elif c["kind"] == "keep":
            for d in set(c["forbid"]) & A.detect_all(c["text"], VOCAB):
                fp[d] += 1
    for d in set(tp) | set(fn):
        assert tp[d] / (tp[d] + fn[d]) >= MIN_RECALL, d
    for d in set(tp) | set(fp):
        assert tp[d] / (tp[d] + fp[d]) >= MIN_PRECISION, d
