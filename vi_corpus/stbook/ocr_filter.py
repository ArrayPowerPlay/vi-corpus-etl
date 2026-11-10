"""
Lọc trang OCR stbook sau khi chạy xong (O-30 mục 2 và 3), chỉ dùng CPU, không chạy lại GPU.

Đọc các file `interim/stbook_ocr/<slug>/<product_id>.json` do chuỗi VLM ghi và chuyển hai loại trang sang `flagged`, đúng quy ước của O-20:
`page_info[i].status = "flagged"`, `pages[i] == ""`, văn bản cũ giữ ở `page_info[i].rejected_text`:
    * `low_real_syllable`      : trang `baseline` có tỷ lệ âm tiết thật (1 - tỷ lệ âm tiết lạ trong tập SEA-PILE) dưới BASELINE_MIN_REAL_RATE.
                                Trang dưới SYL_MIN_WORDS từ không tính được tỷ lệ nên giữ nguyên.
    * `foreign_page_in_vi_book`: trang `rp110` mà fastText gắn ngôn ngữ khác tiếng Việt, nằm trong sách có từ VI_BOOK_MIN_SHARE số trang (đủ chữ)
                                là tiếng Việt. Trang `rp110` ở sách ngoại ngữ không đụng vào (việc lọc ngôn ngữ của corpus lo, O-30 mục 1).
Mỗi trang bị lọc ghi thêm `page_info[i].post_filter = <mã lý do>`, và file có khoá `post_filter` ghi tham số cùng số trang theo lý do.

Chạy lại được và đổi ngưỡng được: mỗi lần chạy trước hết trả lại mọi trang mà bộ lọc này đã loại (văn bản về `pages`, status về "ok"), rồi
mới lọc lại từ đầu, nên kết quả chỉ phụ thuộc dữ liệu và tham số hiện tại. Trang do chuỗi OCR loại (không có `post_filter`) không bị đụng.
Cuốn nào kết quả lọc không đổi thì không ghi lại file. Cuốn đang được tiến trình OCR khác giữ khoá (claim_book) thì bỏ qua.
"""

import csv
import json
import logging
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vi_corpus.common.state import read_json, write_json_atomic
from vi_corpus.ocr_bakeoff.syllables import load_syllables, unknown_rate
from vi_corpus.pipeline.language import get_lid
from vi_corpus.stbook.ocr_books import OCR_DIR
from vi_corpus.stbook.ocr_check import DEFAULT_LANG_MODEL, SYL_MIN_WORDS, VI_BOOK_MIN_SHARE, detect_page_languages
from vi_corpus.stbook.ocr_vlm import ENGINE_TAG, claim_book

logger = logging.getLogger("vi_corpus")

FILTER_DIR = "processed/ocr_page_filter"
BASELINE_TIER = "baseline"
RP110_TIER = "qwen3vl_8b_rp110"
# O-30 mục 2: bộ kiểm O-29 trên 1.247 trang baseline cho phân bố hai đỉnh (858 trang dưới 0,6; chỉ 33 trang trong 0,7-0,85; 298 trang từ 0,85),
# nên 0,85 nằm ở thung lũng: dưới mức này là chữ rác hoặc ảnh ngoại ngữ lẫn dòng rác.
BASELINE_MIN_REAL_RATE = 0.85
REASON_SYLLABLE = "low_real_syllable"
REASON_FOREIGN = "foreign_page_in_vi_book"


def restore_filtered(done: dict) -> int:
    """
    Trả lại mọi trang mà bộ lọc này từng loại (có `post_filter`): văn bản về `pages`, status về "ok", sửa lại `stats`, bỏ khoá `post_filter` của file.

    Returns:
        Số trang đã trả lại.
    """
    n = 0
    for i, pi in enumerate(done["page_info"]):
        if "post_filter" not in pi:
            continue
        done["pages"][i] = pi.pop("rejected_text", "")
        pi["status"] = "ok"
        del pi["post_filter"]
        done["stats"]["ok_by_tier"][pi["tier"]] = done["stats"]["ok_by_tier"].get(pi["tier"], 0) + 1
        done["stats"]["flagged"] -= 1
        n += 1
    done.pop("post_filter", None)
    return n


def plan_book(done: dict, lid, vocab: frozenset[str], min_real_rate: float = BASELINE_MIN_REAL_RATE) -> dict[int, dict]:
    """
    Chọn trang cần loại trong một cuốn (chưa sửa gì). Gọi sau restore_filtered để tính trên văn bản gốc.

    Returns:
        {chỉ số trang: {"reason": mã lý do, ...số đo}}.
    """
    texts, info = done["pages"], done["page_info"]
    langs = detect_page_languages(texts, lid)
    judged = [lg for lg in langs if lg]
    vi_share = sum(lg[0] == "vi" for lg in judged) / len(judged) if judged else None
    plan: dict[int, dict] = {}
    for i, (text, pi) in enumerate(zip(texts, info)):
        if pi.get("status") != "ok":
            continue
        if pi.get("tier") == BASELINE_TIER:
            unk, n_words = unknown_rate(text, vocab)
            if n_words >= SYL_MIN_WORDS and 1 - unk / n_words < min_real_rate:
                plan[i] = {"reason": REASON_SYLLABLE, "real_syllable_rate": round(1 - unk / n_words, 4)}
        elif pi.get("tier") == RP110_TIER and langs[i] and langs[i][0] != "vi" and vi_share is not None and vi_share >= VI_BOOK_MIN_SHARE:
            plan[i] = {"reason": REASON_FOREIGN, "lang": langs[i][0], "lang_conf": round(langs[i][1], 3), "book_vi_share": round(vi_share, 3)}
    return plan


def apply_plan(done: dict, plan: dict[int, dict], params: dict) -> None:
    """Loại các trang trong `plan`: giữ văn bản ở `rejected_text`, `pages[i] = ""`, status "flagged", cập nhật `stats` và ghi khoá `post_filter` của file."""
    by_reason: Counter = Counter()
    for i, item in plan.items():
        pi = done["page_info"][i]
        pi["rejected_text"] = done["pages"][i]
        pi["status"] = "flagged"
        pi["post_filter"] = item["reason"]
        done["pages"][i] = ""
        tiers = done["stats"]["ok_by_tier"]
        tiers[pi["tier"]] -= 1
        if not tiers[pi["tier"]]:
            del tiers[pi["tier"]]
        done["stats"]["flagged"] += 1
        by_reason[item["reason"]] += 1
    if plan:
        done["post_filter"] = {"params": params, "by_reason": dict(by_reason), "applied_at": datetime.now(timezone.utc).isoformat()}


def _filtered_now(done: dict) -> dict[int, str]:
    """{chỉ số trang: mã lý do} của các trang bộ lọc này đang loại trong `done`."""
    return {i: pi["post_filter"] for i, pi in enumerate(done["page_info"]) if "post_filter" in pi}


def filter_pages(data_root: Path, lang_model: str = DEFAULT_LANG_MODEL, syllables: Path | None = None,
                 min_real_rate: float = BASELINE_MIN_REAL_RATE, dry_run: bool = False) -> dict:
    """
    Lọc trang của mọi cuốn đã OCR bằng chuỗi VLM (cách làm ở docstring module) và ghi báo cáo vào <data_root>/processed/ocr_page_filter/.

    Args:
        lang_model:    Bộ nhận diện ngôn ngữ ("fasttext:lid.176", "fasttext:<file.bin>" hoặc "heuristic"). Không nạp được thì báo lỗi, KHÔNG tự rơi
                       về heuristic (bộ lọc xoá dữ liệu nên không được đoán kém chính xác âm thầm).
        syllables:     File tập âm tiết (mặc định <data_root>/processed/ocr_bakeoff/syllables.json); bắt buộc có.
        min_real_rate: Ngưỡng tỷ lệ âm tiết thật của trang baseline.
        dry_run:       Chỉ đếm và ghi báo cáo, không sửa file OCR.

    Returns:
        Tóm tắt (cũng ghi vào summary.json): số cuốn, số cuốn đổi, số trang bị loại theo lý do và theo engine, cuốn bị bỏ qua.

    Raises:
        FileNotFoundError: thiếu tập âm tiết.
    """
    syl_path = syllables or data_root / "processed/ocr_bakeoff/syllables.json"
    if not syl_path.exists():
        raise FileNotFoundError(f"Thiếu {syl_path}: chạy scripts/ocr_bakeoff/build_syllables.py trước")
    vocab = load_syllables(syl_path)
    lid = get_lid(lang_model)
    params = {"baseline_min_real_rate": min_real_rate, "syllable_min_words": SYL_MIN_WORDS, "vi_book_min_share": VI_BOOK_MIN_SHARE,
              "lang_model": lang_model}
    rows: list[dict] = []
    skipped, changed, books = [], 0, 0
    for path in sorted((data_root / OCR_DIR).glob("*/*.json")):
        name = f"{path.parent.name}/{path.stem}"
        with claim_book(data_root, path.parent.name, path.stem) as got:
            if not got:
                skipped.append(name)
                logger.warning("Bỏ qua %s: tiến trình OCR khác đang làm cuốn này", name)
                continue
            done = read_json(path)
            if not done or (done.get("ocr") or {}).get("engine") != ENGINE_TAG:
                continue
            books += 1
            before = _filtered_now(done)
            restore_filtered(done)
            plan = plan_book(done, lid, vocab, min_real_rate)
            for i, item in plan.items():
                rows.append({"book": name, "page": i + 1, "tier": done["page_info"][i]["tier"], **item})
            if {i: it["reason"] for i, it in plan.items()} == before:
                continue  # kết quả lọc không đổi: không ghi lại file
            apply_plan(done, plan, params)
            changed += 1
            if not dry_run:
                write_json_atomic(path, done)
    out_dir = data_root / FILTER_DIR
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out_dir / "flagged_pages.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    summary = {"dry_run": dry_run, "params": params, "books": books, "books_changed": changed, "pages_flagged": len(rows),
               "by_reason": dict(Counter(r["reason"] for r in rows)), "by_tier": dict(Counter(r["tier"] for r in rows)),
               "top_books": Counter(r["book"] for r in rows).most_common(30), "skipped_books": skipped}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
