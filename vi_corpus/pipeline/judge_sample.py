"""
Lấy mẫu ~50.000 đoạn cho LLM lớn chấm điểm (R-11, R-12, G-06): điểm LLM là "đáp án" để chỉnh ngưỡng rule và dạy mô
hình chấm nhỏ (Q6). Module này chỉ CHỌN mẫu; việc chấm chạy trên GPU server (chưa code: thang 0-5 chưa duyệt, Q6-thang).

Cách lấy (R-11):
- Đầu vào là kết quả stage quality (04_quality.parquet) của một hoặc nhiều lần chạy: đã nhận diện ngôn ngữ và đã có kết
  quả rule, nhưng CHƯA lọc. Bỏ trùng chính xác (cùng sha256 text, giữ bản gặp đầu tiên) và đoạn rỗng.
- Hai nhóm: "sft" (hỏi - đáp SEA-Instruct, chấm bằng rubric riêng, khoảng 10% = ~5.000 mẫu, R-12) và "cpt" (còn lại).
- Trong mỗi nhóm: ~70% mẫu qua rule (band thuộc keep_bands), ~30% mẫu bị rule loại (để đo rule loại nhầm / bỏ sót ở đâu).
  Thiếu mẫu bị loại thì lấy hết và bù bằng mẫu qua rule (và ngược lại).
- Phân tầng theo nguồn x độ dài (số token: < 256, 256-1.024, 1.024-2.048, > 2.048): chia đều cho các nguồn, rồi chia đều
  cho các khoảng độ dài trong nguồn; tầng thiếu thì phần dư chuyển cho tầng khác (water-filling). Chia đều thay vì theo tỷ
  lệ để nguồn nhỏ / độ dài hiếm vẫn đủ mẫu đo; phân tích theo tỷ lệ thật thì dùng cột stratum để gán trọng số.
Ngẫu nhiên theo seed (cùng seed + cùng đầu vào thì cùng mẫu).
"""

import hashlib
import math
import random
from collections.abc import Iterable

LENGTH_BUCKETS = ((256, "<256"), (1024, "256-1k"), (2048, "1k-2k"), (math.inf, ">2k"))
SFT_SOURCES = frozenset({"sea_instruct_2602"})


def length_bucket(tokens: int | None) -> str:
    """Khoảng độ dài (theo số token) của một đoạn."""
    for upper, name in LENGTH_BUCKETS:
        if (tokens or 0) < upper:
            return name
    return LENGTH_BUCKETS[-1][1]


def allocate(n: int, caps: dict[str, int]) -> dict[str, int]:
    """
    Chia n cho các khoá càng đều càng tốt, mỗi khoá không vượt caps[khoá] (water-filling).

    Returns:
        {khoá: số lượng}; tổng = min(n, tổng caps).
    """
    out = {k: 0 for k in caps}
    left = min(n, sum(caps.values()))
    open_keys = sorted(k for k in caps if caps[k] > 0)
    while left > 0 and open_keys:
        share = max(left // len(open_keys), 1)
        for k in list(open_keys):
            take = min(share, caps[k] - out[k], left)
            out[k] += take
            left -= take
            if out[k] >= caps[k]:
                open_keys.remove(k)
            if left == 0:
                break
    return out


def candidate_pool(rows: Iterable[dict], keep_bands: tuple[str, ...]) -> list[dict]:
    """Bỏ đoạn rỗng và trùng chính xác; gắn judge_group ("sft" / "cpt"), rule_pass, length_bucket."""
    seen: set[str] = set()
    out = []
    for row in rows:
        text = row["text"] or ""
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not text.strip() or h in seen:
            continue
        seen.add(h)
        out.append({**row, "judge_group": "sft" if row["source_key"] in SFT_SOURCES else "cpt",
                    "rule_pass": row["quality_band"] in keep_bands, "length_bucket": length_bucket(row["token_count"])})
    return out


def _draw(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Lấy n mẫu phân tầng nguồn x độ dài (chia đều, water-filling) từ một pool."""
    strata: dict[str, dict[str, list[dict]]] = {}
    for row in pool:
        strata.setdefault(row["source_key"], {}).setdefault(row["length_bucket"], []).append(row)
    per_source = allocate(n, {s: sum(map(len, b.values())) for s, b in strata.items()})
    picked = []
    for src, k in per_source.items():
        per_bucket = allocate(k, {b: len(v) for b, v in strata[src].items()})
        for b, m in per_bucket.items():
            group = sorted(strata[src][b], key=lambda r: r["doc_id"])
            picked += [{**r, "stratum": f"{src}|{b}"} for r in rng.sample(group, m)]
    return picked


def draw_sample(rows: Iterable[dict], n: int = 50_000, pass_share: float = 0.7, sft_share: float = 0.1,
                keep_bands: tuple[str, ...] = ("A", "B", "C"), seed: int = 42) -> tuple[list[dict], dict]:
    """
    Lấy mẫu chấm điểm (xem docstring module).

    Returns:
        (các mẫu đã chọn, thống kê: số mẫu theo nhóm x qua / bị loại, theo tầng, cỡ pool).
    """
    pool = candidate_pool(rows, keep_bands)
    rng = random.Random(f"{seed}:judge")
    sample: list[dict] = []
    stats: dict = {"pool": len(pool), "groups": {}}
    groups = {g: [r for r in pool if r["judge_group"] == g] for g in ("cpt", "sft")}
    want_sft = min(round(n * sft_share), len(groups["sft"]))
    for g, want in (("sft", want_sft), ("cpt", n - want_sft)):
        passed = [r for r in groups[g] if r["rule_pass"]]
        failed = [r for r in groups[g] if not r["rule_pass"]]
        n_fail = min(round(want * (1 - pass_share)), len(failed))
        n_pass = min(want - n_fail, len(passed))
        n_fail = min(want - n_pass, len(failed))  # thiếu mẫu qua rule thì bù bằng mẫu bị loại
        chosen = _draw(passed, n_pass, rng) + _draw(failed, n_fail, rng)
        stats["groups"][g] = {"pool": len(groups[g]), "pass": n_pass, "fail": n_fail}
        sample += chosen
    strata: dict[str, int] = {}
    for r in sample:
        strata[r["stratum"]] = strata.get(r["stratum"], 0) + 1
    stats["strata"] = dict(sorted(strata.items()))
    stats["total"] = len(sample)
    return sample, stats
