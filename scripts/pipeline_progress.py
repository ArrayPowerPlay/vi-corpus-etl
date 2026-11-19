"""
Xem tiến độ và ước giờ xong của `make pipeline-cpu` (scripts/run_pipeline_cpu.sh), chỉ đọc file, không chạy gì thêm.

Cách đoán: mỗi phần (shard) đi qua 5 stage (ingest, prepare, language, quality, dedup), mỗi stage xong để lại một file
`0N_*.parquet` và ghi số giây vào manifest.json. Từ các phần đã xong stage đó, lấy số giây trung bình của từng stage; phần nào chưa
xong thì cộng số giây các stage còn lại (trừ đi thời gian nó đã chạy ở stage hiện tại). Các phần chạy song song nên nguồn xong
khi phần chậm nhất xong. Nguồn chưa có phần nào xong stage nào thì mượn số giây của nguồn khác (ghi chú "thô") và tính lại khi có số thật.
Bước dedup toàn cục và finalize chỉ được đếm số phần, không ước giờ (chưa có số đo).

Ví dụ:
    uv run --no-sync python scripts/pipeline_progress.py --data-root data
    watch -n 60 "uv run --no-sync python scripts/pipeline_progress.py --data-root data"
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path

# Stage của bước `shards`, theo thứ tự, kèm file kết quả (xem vi_corpus/pipeline/runner.py).
STAGES = [("ingest", "01_ingest.parquet"), ("prepare", "02_prepare.parquet"), ("language", "03_language.parquet"),
          ("quality", "04_quality.parquet"), ("dedup", "05_dedup.parquet")]


def fmt_duration(seconds: float) -> str:
    """Đổi số giây thành chuỗi dễ đọc, ví dụ `2 giờ 15 phút`."""
    seconds = max(0, int(seconds))
    h, m = divmod(seconds // 60, 60)
    if h:
        return f"{h} giờ {m:02d} phút"
    return f"{m} phút" if m else f"{seconds} giây"


def read_seconds(run_dir: Path, stage: str) -> float | None:
    """Đọc số giây stage đã chạy từ manifest.json của phần; None nếu chưa có hoặc file đang ghi dở."""
    try:
        return float(json.loads((run_dir / "manifest.json").read_text()).get(stage, {}).get("seconds"))
    except (OSError, ValueError, TypeError):
        return None


def mtime(path: Path) -> float | None:
    """Thời điểm sửa cuối của file hoặc thư mục; None nếu chưa tồn tại."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def main() -> int:
    """Đọc tham số, đếm file stage của từng phần, in bảng tiến độ và giờ dự kiến xong bước shards. Luôn trả về 0."""
    p = argparse.ArgumentParser(description="Xem tiến độ make pipeline-cpu.")
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("SEA_DATA_ROOT", "data")), help="Thư mục gốc dữ liệu.")
    p.add_argument("--shards", type=int, default=128, help="Số phần (khớp SHARDS của lệnh đang chạy).")
    p.add_argument("--sources", default="vjol,vista", help="Các nguồn, cách nhau bằng dấu phẩy.")
    args = p.parse_args()
    runs = args.data_root / "processed" / "pipeline_runs"
    n, now = args.shards, time.time()
    sources = args.sources.split(",")

    def run_dir(src: str, i: int) -> Path:
        """Thư mục kết quả của phần i của nguồn src."""
        return runs / f"{src}_full_s{i}of{n}"

    # 1) Trạng thái từng phần: số stage đã xong (0..5) và mốc giờ stage cuối; số giây từng stage từ manifest.
    done: dict[str, list[int]] = {}
    last_t: dict[str, list[float | None]] = {}
    secs: dict[str, dict[str, list[float]]] = {s: {st: [] for st, _ in STAGES} for s in sources}
    for s in sources:
        done[s], last_t[s] = [], []
        for i in range(n):
            d = run_dir(s, i)
            k, t = 0, mtime(d)
            for st, fn in STAGES:
                m = mtime(d / fn)
                if m is None:
                    break
                k, t = k + 1, m
                v = read_seconds(d, st)
                if v is not None:
                    secs[s][st].append(v)
            done[s].append(k)
            last_t[s].append(t)

    # 2) Số giây trung bình từng stage: của nguồn đó, thiếu thì mượn trung bình chung (đánh dấu thô).
    pooled = {st: [v for s in sources for v in secs[s][st]] for st, _ in STAGES}
    rough = False

    def mean_sec(s: str, st: str) -> float | None:
        """Trung bình số giây của stage st ở nguồn s; thiếu thì mượn nguồn khác; None nếu chưa nguồn nào có số đo."""
        nonlocal rough
        if secs[s][st]:
            return statistics.mean(secs[s][st])
        if pooled[st]:
            rough = True
            return statistics.mean(pooled[st])
        return None

    print(f"== Tiến độ pipeline-cpu ({n} phần mỗi nguồn) lúc {datetime.fromtimestamp(now):%F %T} ==")
    print(f"{'nguồn':8}" + "".join(f"{st:>10}" for st, _ in STAGES) + "   (số phần đã xong stage đó)")
    for s in sources:
        counts = [sum(1 for k in done[s] if k > j) for j in range(len(STAGES))]
        print(f"{s:8}" + "".join(f"{c:>10}" for c in counts))

    # 3) Ước giờ xong bước shards: mỗi nguồn xong khi phần chậm nhất xong.
    finish_all, unknown = 0.0, False
    overrun: tuple[str, str, float, float] | None = None  # (nguồn, stage, đã chạy, trung bình) của phần quá lâu
    total_w = done_w = 0.0
    for s in sources:
        worst = 0.0
        for i in range(n):
            k, t = done[s][i], last_t[s][i]
            rest = 0.0
            for j in range(k, len(STAGES)):
                m = mean_sec(s, STAGES[j][0])
                if m is None:
                    unknown = True
                    break
                spent = max(0.0, now - t) if (j == k and t) else 0.0
                if j == k and spent > m:  # stage này đã chạy lâu hơn trung bình: không đoán được nữa
                    if overrun is None or spent > overrun[2]:
                        overrun = (s, STAGES[j][0], spent, m)
                    continue
                rest += m - spent if j == k else m
            worst = max(worst, rest)
        finish_all = max(finish_all, worst)
        for j, (st, _) in enumerate(STAGES):
            m = mean_sec(s, st)
            if m is not None:
                total_w += m * n
                done_w += m * sum(1 for k in done[s] if k > j)
    all_done = all(k == len(STAGES) for s in sources for k in done[s])
    print()
    if all_done:
        print("Bước shards: XONG (mọi phần đã qua dedup vòng 1).")
    elif overrun:
        o = overrun
        print(f"Chưa ước được giờ xong: nguồn {o[0]} đang ở stage {o[1]} đã {fmt_duration(o[2])}, lâu hơn trung bình "
              f"{fmt_duration(o[3])} dùng để so (mượn nguồn khác nếu nguồn này chưa có số đo). Nguồn này nặng hơn, hoặc đang kẹt.")
    elif unknown:
        print("Chưa ước được giờ: chưa có phần nào xong stage nào (xem lại sau vài phút).")
    else:
        pct = 100 * done_w / total_w if total_w else 0
        print(f"Bước shards: khoảng {pct:.0f}% (tính theo thời gian của từng stage), còn khoảng {fmt_duration(finish_all)}, "
              f"dự kiến xong lúc {datetime.fromtimestamp(now) + timedelta(seconds=finish_all):%H:%M}"
              + (" (thô: mượn số đo nguồn khác, nguồn kia có thể to nhỏ khác)" if rough else ""))

    # 4) Các bước sau: chỉ đếm.
    verdict = args.data_root / "processed" / "dedup_global" / "verdict.parquet"
    fin = sum(1 for s in sources for i in range(n) if (run_dir(s, i) / "clean.parquet").exists())
    print(f"Dedup toàn cục: {'đã có verdict.parquet' if verdict.exists() else 'chưa chạy xong'}")
    print(f"Finalize: {fin}/{n * len(sources)} phần có clean.parquet")
    failed = args.data_root / "logs" / "pipe_failed_shards.txt"
    if failed.exists() and failed.stat().st_size:
        print(f"!!! Có phần lỗi, xem {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
