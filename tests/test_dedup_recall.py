"""
Đo độ nhạy (recall) của fuzzy dedup hiện tại bằng cách cấy bản trùng (G-05 bước 1, theo thí nghiệm M-04).

Văn bản gốc là chuỗi âm tiết ngẫu nhiên (~400 từ) nên các bản gốc gần như không chung shingle với nhau. Mỗi bản gốc sinh
các biến thể: chép nguyên, sửa 1% / 2% / 5% số từ, cắt 10% đầu + 10% cuối, nằm trong một văn bản lớn hơn gấp 2 / 4 lần.
Test khẳng định mức sàn đã đo với cấu hình A (shingle 5, 16 x 8, ngưỡng 0,8; R-34 giữ cấu hình này tới khi có S-11) và
0 dương tính giả giữa các bản gốc. Các trường hợp MinHash cấu hình A bắt kém (sửa 5%, văn bản nằm trong bản lớn) được
ghi là giới hạn đã biết; đổi cấu hình thì cập nhật bảng EXPECTED. Văn bản nằm trong bản lớn do L1 (hash đoạn văn,
dedup.find_contained) bắt, test ở tests/test_dedup.py.
"""

import random

import numpy as np

from vi_corpus.pipeline.dedup import lsh_pairs, minhash

SYLLABLES = ("ba bé bê bi bo bô bơ bu bư ca cá cả cạ cò cô cơ cu cư da dê di do đô đơ đu đư ga gà gỗ ha hè hi ho hô "
             "hơ hu hư la lá lê li lo lô lơ lu lư ma mè mi mo mô mơ mu mư na nê ni no nô nơ nu nư pha phê phi pho "
             "qua quê qui ra rê ri ro rô rơ ru rư sa sê si so sô sơ su sư ta tê ti to tô tơ tu tư va vê vi vo vô vơ "
             "vu vư xa xê xi xo xô xơ xu xư").split()
N_BASE, LEN = 40, 400
THRESHOLD = 0.8
# Mức sàn recall (tỷ lệ biến thể được nối trực tiếp với bản gốc) của cấu hình A. Đo 2026-10-08 trên bộ sinh này (seed
# 2026): exact 1,0; edit1 1,0; edit2 0,875; edit5 0,0; trim10 0,5; contained2 0,0; contained4 0,0 (khớp M-04).
EXPECTED = {"exact": 1.0, "edit1": 1.0, "edit2": 0.75, "edit5": 0.0, "trim10": 0.4, "contained2": 0.0,
            "contained4": 0.0}


def _variants(words: list[str], rng: random.Random) -> dict[str, list[str]]:
    """Các biến thể cấy trùng của một văn bản gốc."""
    def edit(frac: float) -> list[str]:
        """Thay ngẫu nhiên frac số từ."""
        out = list(words)
        for i in rng.sample(range(len(out)), max(1, int(frac * len(out)))):
            out[i] = rng.choice(SYLLABLES) + "x"
        return out

    cut = len(words) // 10
    filler = lambda k: [rng.choice(SYLLABLES) for _ in range(k * len(words))]  # noqa: E731
    return {"exact": list(words), "edit1": edit(0.01), "edit2": edit(0.02), "edit5": edit(0.05),
            "trim10": words[cut:-cut], "contained2": filler(1)[: len(words) // 2] + words + filler(1)[: len(words) // 2],
            "contained4": filler(2)[: 3 * len(words) // 2] + words + filler(2)[: 3 * len(words) // 2]}


def test_recall_cay_ban_trung_cau_hinh_a():
    """Mỗi loại biến thể đạt mức sàn recall; các bản gốc không bị nối với nhau (0 dương tính giả)."""
    rng = random.Random(2026)
    bases = [[rng.choice(SYLLABLES) for _ in range(LEN)] for _ in range(N_BASE)]
    kinds = list(EXPECTED)
    texts = [" ".join(b) for b in bases]
    owner = [(-1, "base")] * N_BASE
    for i, b in enumerate(bases):
        for kind, words in _variants(b, rng).items():
            texts.append(" ".join(words))
            owner.append((i, kind))
    sigs = np.stack([minhash(t) for t in texts])
    pairs = {tuple(p) for p in lsh_pairs(sigs, THRESHOLD).tolist()}
    false_pos = [p for p in pairs if p[0] < N_BASE and p[1] < N_BASE]
    assert not false_pos
    recall = {}
    for kind in kinds:
        idx = [j for j, (i, k) in enumerate(owner) if k == kind]
        recall[kind] = sum((owner[j][0], j) in pairs for j in idx) / len(idx)
    print("recall cấu hình A:", recall)
    for kind, floor in EXPECTED.items():
        assert recall[kind] >= floor, (kind, recall[kind])
