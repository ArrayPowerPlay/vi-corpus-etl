"""
Perplexity của đầu ra OCR theo một mô hình ngôn ngữ nhỏ (mặc định Qwen/Qwen3-0.6B), F-11 bộ B.

Đo trên PPL_TOKENS token đầu của văn bản thuần mỗi trang: văn bản đúng thứ tự đọc có perplexity thấp hơn văn bản bị trộn
dòng giữa hai cột hay đọc sai dấu. Chỉ dùng để so các engine với nhau trên cùng trang (giá trị tuyệt đối không có nghĩa).
Ghi <bakeoff>/runs/<engine>/ppl.jsonl ({"page_id", "ppl", "tokens"}); trang đã có thì bỏ qua. Cần transformers + torch
(group curator, cài mặc định) và nên chạy trên GPU.
"""

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger("vi_corpus")

PPL_MODEL = "Qwen/Qwen3-0.6B"
PPL_TOKENS = 512


def latest_outputs(run_dir: Path) -> dict[str, dict]:
    """Kết quả mới nhất không lỗi của từng trang trong runs/<engine>/outputs.jsonl."""
    out: dict[str, dict] = {}
    path = run_dir / "outputs.jsonl"
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("error"):
                out[rec["page_id"]] = rec
    return out


def score_runs(bakeoff_dir: Path, engines: list[str], model_id: str = PPL_MODEL, device: str = "cuda",
               max_tokens: int = PPL_TOKENS) -> dict[str, int]:
    """
    Tính perplexity cho mọi trang của các engine (trang đã có trong ppl.jsonl thì bỏ qua).

    Returns:
        {engine: số trang mới tính}.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if device == "cuda" else None)
    model.to(device).eval()
    done_counts = {}
    for name in engines:
        run_dir = bakeoff_dir / "runs" / name
        ppl_path = run_dir / "ppl.jsonl"
        have = set(latest_outputs_ppl(ppl_path))
        n = 0
        with open(ppl_path, "a", encoding="utf-8") as f:
            for page_id, rec in sorted(latest_outputs(run_dir).items()):
                if page_id in have:
                    continue
                ids = tok(rec.get("text") or "", return_tensors="pt", truncation=True,
                          max_length=max_tokens)["input_ids"].to(device)
                if ids.shape[1] < 2:
                    ppl, ntok = None, int(ids.shape[1])
                else:
                    with torch.no_grad():
                        loss = model(ids, labels=ids).loss.item()
                    ppl, ntok = math.exp(loss), int(ids.shape[1])
                f.write(json.dumps({"page_id": page_id, "ppl": ppl, "tokens": ntok}) + "\n")
                n += 1
        done_counts[name] = n
        logger.info("[ppl] %s: %d trang mới", name, n)
    return done_counts


def latest_outputs_ppl(path: Path) -> dict[str, float | None]:
    """Đọc ppl.jsonl: {page_id: ppl}."""
    out: dict[str, float | None] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["page_id"]] = rec.get("ppl")
    return out
