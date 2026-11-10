"""
Thử tốc độ và tỷ lệ trang bị báo của chuỗi OCR VLM trên trang sách stbook thật (O-20, mục "chưa duyệt").

Quy trình (scripts/stbook/ocr_vlm_bench.py, scripts/stbook/run_ocr_vlm_bench.sh), thư mục làm việc <out>:
    sample : chọn ~1.000 trang ngẫu nhiên từ nhiều sách (5-95% mỗi cuốn, bỏ bìa và mục lục cuối), ghi ảnh gốc vào pages/ và
             pages.jsonl (cùng định dạng bộ trang của đợt so sánh nên dùng lại engines.run_engine).
    run    : chạy cả bộ trang qua engine chính (`qwen3vl_8b_mt4k`) ở MỘT mức `concurrency`, mỗi mức một GPU và một server riêng
             (runs/c<N>/outputs.jsonl); các mức chạy song song trên các GPU khác nhau nên không dùng chung bộ nhớ đệm của vLLM
             và cùng một bộ trang cho số liệu so sánh được.
    retry  : lấy trang bị bộ kiểm báo ở mức đầu tiên, chạy lại bằng `qwen3vl_8b_rp110` (retry/runs/rp110).
    report : tổng hợp report/summary.json và in bảng: trang/giây ổn định, độ trễ, token, tỷ lệ trang bị báo theo lý do, giờ chạy cả
             kho nếu mỗi GPU chạy một server ở mức đó, và số trang còn bị báo sau khi chạy lại (sẽ rơi xuống OCR cũ).
Số trang/giây là số đo ổn định (bỏ các trang khởi động, scoring.throughput). Giờ cả kho giả định tốc độ tăng tuyến tính theo số GPU.
"""

import io
import json
import logging
import random
import statistics
from collections import Counter
from pathlib import Path

import pymupdf
from PIL import Image

from vi_corpus.ocr_bakeoff.guard import guard_page
from vi_corpus.ocr_bakeoff.pages import page_image_bytes
from vi_corpus.ocr_bakeoff.scoring import throughput

logger = logging.getLogger("vi_corpus")

DEFAULT_CORPUS_PAGES = 366089  # tổng số trang cần OCR ở O-20 (stbook + giáo trình cần OCR)
MIN_BOOK_PAGES = 40  # sách ngắn hơn không vào mẫu (trang 5-95% quá ít)


def sample_stbook_pages(books: list[tuple[str, dict, Path]], out: Path, n_pages: int, n_books: int, seed: int) -> list[dict]:
    """
    Chọn n_pages trang ngẫu nhiên từ n_books cuốn ngẫu nhiên (chia đều, cuốn đầu nhận phần dư), ghi ảnh gốc vào out/pages/.

    Cuốn có ít trang hơn phần được chia thì lấy hết trang hợp lệ của nó; phần thiếu được bù từ các cuốn còn lại (cuốn rút thêm).

    Returns:
        Bản ghi trang (page_id "C0001"..., set "C", image, gt None, size, source) và cũng ghi out/pages.jsonl.

    Raises:
        ValueError: không có sách đủ dài để lấy mẫu.
    """
    rng = random.Random(seed)
    eligible = []
    for slug, book, pdf in books:
        try:
            with pymupdf.open(pdf) as doc:
                if doc.page_count >= MIN_BOOK_PAGES and not doc.is_repaired:
                    eligible.append((slug, book, pdf, doc.page_count))
        except Exception as e:  # noqa: BLE001 - PDF hỏng không vào mẫu
            logger.warning("Bỏ qua %s: %s", pdf, e)
    if not eligible:
        raise ValueError("Không có sách stbook đủ dài để lấy mẫu")
    rng.shuffle(eligible)
    chosen = eligible[:n_books]
    (out / "pages").mkdir(parents=True, exist_ok=True)
    picks: list[tuple[tuple, int]] = []
    pools = {id(c): list(range(int(0.05 * c[3]), max(int(0.95 * c[3]), 1))) for c in eligible}
    for c in chosen:
        rng.shuffle(pools[id(c)])
    quota = {id(c): n_pages // len(chosen) + (j < n_pages % len(chosen)) for j, c in enumerate(chosen)}
    for c in chosen:
        take = min(quota[id(c)], len(pools[id(c)]))
        picks += [(c, i) for i in pools[id(c)][:take]]
        del pools[id(c)][:take]
    for c in chosen * 4:  # bù phần thiếu từ các cuốn đã chọn còn trang
        if len(picks) >= n_pages:
            break
        if pools[id(c)]:
            picks.append((c, pools[id(c)].pop()))
    picks = picks[:n_pages]
    rng.shuffle(picks)  # trộn thứ tự để trang của một cuốn không dồn vào một đoạn
    records = []
    docs: dict[int, pymupdf.Document] = {}
    try:
        for c, i in picks:
            doc = docs.setdefault(id(c), pymupdf.open(c[2]))
            data, ext = page_image_bytes(doc[i])
            page_id = f"C{len(records) + 1:04d}"
            path = out / "pages" / f"{page_id}.{ext}"
            path.write_bytes(data)
            w, h = Image.open(io.BytesIO(data)).size
            records.append({"page_id": page_id, "set": "C", "image": f"pages/{path.name}", "gt": None, "flags": {}, "size": [w, h],
                            "source": {"kind": "stbook", "slug": c[0], "product_id": str(c[1]["product_id"]),
                                       "title": c[1].get("title") or c[1].get("name"), "page": i + 1}})
    finally:
        for d in docs.values():
            d.close()
    with open(out / "pages.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    logger.info("Mẫu: %d trang từ %d cuốn", len(records), len({r["source"]["product_id"] for r in records}))
    return records


def read_outputs(run_dir: Path) -> dict[str, dict]:
    """Bản ghi cuối cùng của từng trang trong runs/<tên>/outputs.jsonl (kể cả bản ghi lỗi); {} nếu chưa có."""
    path = run_dir / "outputs.jsonl"
    out: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[rec["page_id"]] = rec
    return out


def _run_info(run_dir: Path) -> dict:
    """run_info.json của một lượt chạy ({} nếu chưa có)."""
    path = run_dir / "run_info.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def flagged_pages(run_dir: Path) -> dict[str, list[str]]:
    """{page_id: mã lý do bộ kiểm báo} cho các trang bị báo (kể cả trang lỗi) của một lượt chạy."""
    max_tokens = _run_info(run_dir).get("config", {}).get("max_tokens")
    flagged = {}
    for pid, rec in read_outputs(run_dir).items():
        reasons = guard_page(rec, max_tokens)
        if reasons:
            flagged[pid] = reasons
    return flagged


def make_retry_dir(out: Path, page_ids: list[str]) -> Path:
    """Tạo out/retry/ với pages.jsonl chỉ gồm các trang `page_ids` (ảnh trỏ về out/pages/ bằng đường dẫn tương đối); trả về thư mục."""
    retry = out / "retry"
    retry.mkdir(exist_ok=True)
    keep = set(page_ids)
    pages = [json.loads(line) for line in (out / "pages.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
    with open(retry / "pages.jsonl", "w", encoding="utf-8") as f:
        for p in pages:
            if p["page_id"] in keep:
                f.write(json.dumps({**p, "image": "../" + p["image"]}, ensure_ascii=False) + "\n")
    return retry


def _pct(values: list[float], q: float) -> float | None:
    """Phân vị q (0-1) của danh sách, None nếu rỗng."""
    if not values:
        return None
    s = sorted(values)
    return s[min(int(q * len(s)), len(s) - 1)]


def summarize_run(run_dir: Path, corpus_pages: int, num_gpus: int, n_pages: int | None = None) -> dict:
    """
    Số đo của một lượt chạy (một mức concurrency).

    Returns:
        {"concurrency", "pages", "errors", "pages_per_s" (ổn định), "wall_pages_per_s", "latency_p50_s", "latency_p95_s",
         "completion_tokens_mean", "finish_reasons", "flagged", "flagged_rate", "flag_reasons", "corpus_hours" (cả kho nếu
         mỗi GPU chạy một server ở mức này), "complete" (đã chạy đủ n_pages)}.
    """
    info = _run_info(run_dir)
    recs = read_outputs(run_dir)
    sessions = info.get("sessions") or [{}]
    conc = sessions[-1].get("concurrency")
    ok = [r for r in recs.values() if not r.get("error")]
    errors = len(recs) - len(ok)
    speed = throughput(run_dir, warmup=max(20, conc or 0))
    ends = [r["t_end"] for r in ok if r.get("t_end")]
    starts = [r["t_start"] for r in ok if r.get("t_start")]
    lat = [r["t_end"] - r["t_start"] for r in ok if r.get("t_end") and r.get("t_start")]
    toks = [r["completion_tokens"] for r in ok if r.get("completion_tokens")]
    flagged = flagged_pages(run_dir)
    reasons: Counter = Counter(x for rs in flagged.values() for x in rs)
    pps = speed["pages_per_s"]
    return {"concurrency": conc, "pages": len(ok), "errors": errors, "pages_per_s": pps,
            "wall_pages_per_s": len(ok) / (max(ends) - min(starts)) if ends and starts and max(ends) > min(starts) else None,
            "latency_p50_s": _pct(lat, 0.5), "latency_p95_s": _pct(lat, 0.95),
            "completion_tokens_mean": statistics.fmean(toks) if toks else None,
            "finish_reasons": dict(Counter(r.get("finish_reason") for r in ok)),
            "flagged": len(flagged), "flagged_rate": len(flagged) / len(recs) if recs else None, "flag_reasons": dict(reasons),
            "corpus_hours": corpus_pages / (pps * num_gpus) / 3600 if pps else None,
            "complete": n_pages is None or len(ok) >= n_pages}


def summarize(out: Path, corpus_pages: int = DEFAULT_CORPUS_PAGES, num_gpus: int = 4) -> dict:
    """
    Tổng hợp mọi mức concurrency (runs/c*) và lượt chạy lại (retry/runs/rp110) rồi ghi out/report/summary.json.

    Returns:
        {"sample_pages", "levels": {tên: summarize_run}, "retry": {"pages", "still_flagged", "reasons"} hoặc None, "best_level"}.
    """
    n = sum(1 for line in (out / "pages.jsonl").read_text(encoding="utf-8").split("\n") if line.strip())
    runs = sorted((d for d in (out / "runs").glob("c*") if d.is_dir()), key=lambda d: int(d.name[1:]))
    levels = {d.name: summarize_run(d, corpus_pages, num_gpus, n) for d in runs}
    retry = None
    retry_run = out / "retry" / "runs" / "rp110"
    if retry_run.exists():
        recs = read_outputs(retry_run)
        still = flagged_pages(retry_run)
        retry = {"pages": len(recs), "still_flagged": len(still),
                 "reasons": dict(Counter(x for rs in still.values() for x in rs)),
                 "recovered_rate": 1 - len(still) / len(recs) if recs else None}
    steady = {k: v for k, v in levels.items() if v["pages_per_s"] and v["complete"] and not v["errors"]}
    best = max(steady, key=lambda k: steady[k]["pages_per_s"]) if steady else None
    summary = {"sample_pages": n, "corpus_pages": corpus_pages, "num_gpus": num_gpus, "levels": levels, "retry": retry,
               "best_level": best}
    (out / "report").mkdir(exist_ok=True)
    (out / "report" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def format_report(summary: dict) -> str:
    """Bảng tóm tắt (văn bản) của summarize() để in ra terminal."""

    def f(v, spec=".2f", unit="") -> str:
        """Định dạng số, '—' nếu None."""
        return "—" if v is None else f"{v:{spec}}{unit}"

    lines = [f"Mẫu {summary['sample_pages']} trang stbook thật; quy ra cả kho {summary['corpus_pages']} trang trên "
             f"{summary['num_gpus']} GPU (giả định tuyến tính theo số GPU)",
             f"{'mức':>6s} {'trang':>6s} {'lỗi':>4s} {'trang/s':>8s} {'trễ p50':>8s} {'trễ p95':>8s} {'token TB':>9s} {'bị báo':>7s} {'giờ cả kho':>11s}"]
    for name, v in summary["levels"].items():
        lines.append(f"{name:>6s} {v['pages']:6d} {v['errors']:4d} {f(v['pages_per_s']):>8s} {f(v['latency_p50_s'], '.1f', 's'):>8s} "
                     f"{f(v['latency_p95_s'], '.1f', 's'):>8s} {f(v['completion_tokens_mean'], '.0f'):>9s} "
                     f"{f(None if v['flagged_rate'] is None else 100 * v['flagged_rate'], '.1f', '%'):>7s} {f(v['corpus_hours'], '.1f', ' giờ'):>11s}"
                     + ("" if v["complete"] else "  (CHƯA chạy đủ trang)"))
    for name, v in summary["levels"].items():
        lines.append(f"  {name}: lý do bị báo {v['flag_reasons']}; finish_reason {v['finish_reasons']}")
    r = summary["retry"]
    if r:
        lines.append(f"Chạy lại {r['pages']} trang bị báo bằng qwen3vl_8b_rp110: còn bị báo {r['still_flagged']} "
                     f"({f(None if r['recovered_rate'] is None else 100 * r['recovered_rate'], '.0f', '%')} sửa được), lý do {r['reasons']}; "
                     "số này rơi xuống OCR cũ (baseline)")
    lines.append(f"Mức nhanh nhất (đủ trang, không lỗi): {summary['best_level'] or 'chưa có'}")
    return "\n".join(lines)
