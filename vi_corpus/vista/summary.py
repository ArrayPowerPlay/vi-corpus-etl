"""
Tổng hợp các lần chạy pipeline của VISTA (VS-06): gộp manifest.json + audit.json của các phần (`--shard I/N`) hoặc của một lần chạy duy nhất.

Cho biết, theo từng lần chạy và tổng: số đoạn vào, số đoạn giữ lại, token giữ lại (tokenizer chính thức), phễu loại theo lý do (quality / rights / duplicate), RAM đỉnh và thời gian
từng stage, và tỷ lệ lineage. Tổng token sau dedup vòng 2 chính xác khi các phần đã chạy lại với --global-verdict (finalize ghi lại audit.json theo verdict).
Chỉ đọc file nhỏ (manifest, audit), không đọc Parquet.
"""

import json
from pathlib import Path

RUNS_REL = "processed/pipeline_runs"
SOURCE = "vista"
STAGES = ("ingest", "prepare", "language", "quality", "dedup", "finalize")


def _read(path: Path) -> dict:
    """Đọc JSON; file thiếu hoặc hỏng thì {}."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run_dirs(data_root: Path, prefix: str) -> list[Path]:
    """Các thư mục lần chạy có tên bắt đầu bằng `prefix` (sắp theo tên) trong processed/pipeline_runs/."""
    base = data_root / RUNS_REL
    return sorted(p for p in base.glob(f"{prefix}*") if p.is_dir()) if base.is_dir() else []


def summarize_run(run: Path) -> dict:
    """
    Tóm tắt một lần chạy: {"run", "finished", "chunks", "kept", "tokens_kept", "funnel", "lineage_kept", "peak_rss_mb", "seconds", "global_verdict"}.

    `finished` = đã qua stage finalize (có audit.json). Lần chạy dang dở vẫn được liệt kê (số liệu lấy từ manifest các stage đã xong).
    """
    manifest = _read(run / "manifest.json")
    audit = _read(run / "audit.json")
    funnel = (audit.get("funnel") or {}).get(SOURCE) or {}
    tokens = (audit.get("tokens") or {}).get(SOURCE) or {}
    done = [s for s in STAGES if s in manifest]
    return {
        "run": run.name, "finished": bool(audit), "stages_done": done,
        "chunks": funnel.get("input", (manifest.get("prepare") or {}).get("rows")), "kept": funnel.get("kept"), "tokens_kept": tokens.get("tokens_kept"),
        "funnel": {k: v for k, v in funnel.items() if k != "input"}, "lineage_kept": audit.get("lineage_rate_kept"),
        "peak_rss_mb": max((manifest.get(s, {}).get("peak_rss_mb") or 0 for s in STAGES), default=0),
        "seconds": round(sum(manifest.get(s, {}).get("seconds") or 0 for s in STAGES), 1),
        "global_verdict": bool((manifest.get("finalize") or {}).get("global_verdict")),
    }


def summarize(data_root: Path, prefix: str) -> dict:
    """
    Tổng hợp mọi lần chạy có tên bắt đầu bằng `prefix`.

    Returns:
        {"runs": [summarize_run...], "total": {"chunks", "kept", "tokens_kept", "funnel", "unfinished", "without_global_verdict"}}. Số tổng chỉ cộng các lần đã xong.
    """
    runs = [summarize_run(r) for r in run_dirs(data_root, prefix)]
    total = {"chunks": 0, "kept": 0, "tokens_kept": 0, "funnel": {}, "unfinished": [r["run"] for r in runs if not r["finished"]],
             "without_global_verdict": [r["run"] for r in runs if r["finished"] and not r["global_verdict"]]}
    for r in runs:
        if not r["finished"]:
            continue
        total["chunks"] += r["chunks"] or 0
        total["kept"] += r["kept"] or 0
        total["tokens_kept"] += r["tokens_kept"] or 0
        for k, v in r["funnel"].items():
            total["funnel"][k] = total["funnel"].get(k, 0) + v
    return {"runs": runs, "total": total}
