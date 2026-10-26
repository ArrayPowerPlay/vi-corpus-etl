"""
Phép thử "mỏ vàng" theo nhóm font của VJOL (J-42 mục (4), P2): tìm nhóm PDF mà lớp chữ KHÔNG khớp với cái mắt thấy, mà không phải viết bộ dò mới cho từng kiểu lỗi.

Với mỗi cặp (producer, creator) của PDF lấy vài bài (cố định theo seed), vẽ vài trang giữa bài, OCR bằng chuỗi engine VLM rồi so với lớp chữ nhúng bằng độ giống ký tự (char_similarity: bỏ khoảng trắng,
chữ thường, SequenceMatcher). Độ giống thấp nghĩa là lớp chữ sai so với ảnh trang (lệch dấu, rơi chữ, phông dịch ký tự): cả nhóm nên OCR cả bài. Lưu ý: lỗi mà chính ảnh cũng in sai (dấu "-" thay "ư",
J-32) cho độ giống CAO, đúng như mong muốn vì OCR không sửa được.

- group_of / pick_docs: gom bài theo nhóm, chọn tối đa `per_group` bài mỗi nhóm bằng băm (seed, sha256) nên không phụ thuộc thứ tự liệt kê.
- measure_doc: OCR `n_pages` trang chữ giữa bài, trả về độ giống từng trang (chỉ trang OCR đọc được, status ok và có chữ).
- run_canary: chạy trên danh sách bài, trả về thống kê từng nhóm.
- label_reference: đối chiếu nhóm với nhãn chuẩn (bài nào được gán nhãn lỗi font / sạch) để chủ dự án chọn ngưỡng; ngưỡng KHÔNG đoán trước.
- bad_groups / write_groups: nhóm dưới ngưỡng ghi vào font_groups_bad.json (extract_pdf đọc qua reocr.load_font_groups).
"""

import csv
import difflib
import hashlib
import logging
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from vi_corpus.common.pdf_text import is_tcvn3, page_text
from vi_corpus.common.state import write_json_atomic

logger = logging.getLogger("vi_corpus")

PER_GROUP = 3  # số bài mỗi nhóm
PAGES_PER_DOC = 2  # số trang giữa bài mỗi bài
MIN_LAYER_CHARS = 200  # trang phải có ít nhất ngần này ký tự (không tính khoảng trắng) lớp chữ mới đem so
SIM_CHARS = 1500  # chỉ so ngần này ký tự đầu của mỗi bên (SequenceMatcher chậm với trang dài)
# Nhãn chuẩn (audit.LABELS) coi là lỗi do font / lớp chữ, để đối chiếu nhóm; `reference` và `running_header` là lỗi làm sạch, không phải lỗi font.
FONT_BAD_LABELS = frozenset({"lost_diacritics", "hyphen_font", "font_misread", "old_ocr", "font_shift", "dropped_letters", "joined_words", "garbage"})


def char_similarity(a: str, b: str, limit: int = SIM_CHARS) -> float:
    """Độ giống ký tự trong [0, 1] của hai văn bản: bỏ mọi khoảng trắng, chữ thường, so `limit` ký tự đầu bằng difflib.SequenceMatcher.ratio; một bên rỗng thì 0."""
    x = "".join(a.split()).lower()[:limit]
    y = "".join(b.split()).lower()[:limit]
    if not x or not y:
        return 0.0
    return difflib.SequenceMatcher(None, x, y, autojunk=False).ratio()


def group_of(doc: dict) -> tuple[str, str]:
    """Nhóm font của một bài: (producer, creator) đã cắt khoảng trắng; ("", "") là bài không có metadata (không phải một nhóm)."""
    return (doc.get("producer") or "").strip(), (doc.get("creator") or "").strip()


def pick_docs(docs: Iterable[dict], per_group: int = PER_GROUP, seed: int = 0) -> dict[tuple[str, str], list[dict]]:
    """
    Gom bài theo nhóm font (bỏ nhóm ("", "")) và chọn tối đa `per_group` bài mỗi nhóm theo băm (seed, sha256).

    Returns:
        {(producer, creator): [bài đã chọn]}; "docs_total" của nhóm lấy từ `group_sizes`.
    """
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in docs:
        if group_of(d) != ("", ""):
            by_group[group_of(d)].append(d)
    return {g: sorted(ds, key=lambda d: hashlib.sha256(f"{seed}:{d['sha256']}".encode()).hexdigest())[:per_group] for g, ds in by_group.items()}


def middle_text_pages(texts: Sequence[str], n: int = PAGES_PER_DOC, min_chars: int = MIN_LAYER_CHARS) -> list[int]:
    """Chỉ số của tối đa `n` trang có lớp chữ đủ dài, lấy từ giữa bài (trang chữ gần trang giữa nhất trước), trả về theo thứ tự trang."""
    ok = [i for i, t in enumerate(texts) if len("".join(t.split())) >= min_chars]
    mid = (len(texts) - 1) / 2
    return sorted(sorted(ok, key=lambda i: (abs(i - mid), i))[:n])


def measure_doc(path: Path, get_ocr: Callable[[], object | None], n_pages: int = PAGES_PER_DOC) -> list[float]:
    """
    Độ giống giữa lớp chữ và OCR của `n_pages` trang chữ giữa bài (mỗi trang một số). Trang OCR không đọc được (status khác ok hoặc rỗng) bị bỏ.

    Raises:
        ServerDown, EngineUnavailable: chuỗi OCR hỏng (lỗi hệ thống) — để dừng cả lượt như vi_corpus.vjol.extract.extract_file; PDF hỏng thì trả [].
    """
    import pymupdf

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001 — một PDF không mở được chỉ làm nhóm thiếu một mẫu
        logger.warning("Canary: không mở được %s: %s", path, exc)
        return []
    with doc:
        if doc.is_repaired or doc.needs_pass:
            return []
        texts = [page_text(p) for p in doc]
        if is_tcvn3("".join(texts)):
            texts = [page_text(p, tcvn3_all=True) for p in doc]
        idx = middle_text_pages(texts, n_pages)
        ocr = get_ocr() if idx else None
        if not idx or ocr is None:
            return []
        results = ocr.ocr_pages([doc[i] for i in idx], render=[True] * len(idx))
    return [char_similarity(texts[i], r["text"]) for i, r in zip(idx, results) if r.get("status") == "ok" and (r.get("text") or "").strip()]


def run_canary(docs: Iterable[dict], get_ocr: Callable[[], object | None], per_group: int = PER_GROUP, n_pages: int = PAGES_PER_DOC, seed: int = 0) -> list[dict]:
    """
    Chạy phép thử trên danh sách bài {sha256, path, article_id, producer, creator}.

    Returns:
        [{"producer", "creator", "docs_total", "docs_tested", "pages", "similarity_mean", "similarity_min", "article_ids"}] theo độ giống tăng dần (nhóm không đo được trang nào ở cuối,
        similarity_mean None).
    """
    docs = list(docs)
    sizes = Counter(group_of(d) for d in docs)
    out = []
    for group, picked in pick_docs(docs, per_group, seed).items():
        sims: list[float] = []
        tested = 0
        for d in picked:
            got = measure_doc(Path(d["path"]), get_ocr, n_pages)
            tested += bool(got)
            sims += got
        out.append({"producer": group[0], "creator": group[1], "docs_total": sizes[group], "docs_tested": tested, "pages": len(sims),
                    "similarity_mean": round(statistics.mean(sims), 4) if sims else None, "similarity_min": round(min(sims), 4) if sims else None,
                    "article_ids": [d.get("article_id") for d in picked]})
    return sorted(out, key=lambda g: (g["similarity_mean"] is None, g["similarity_mean"] or 0.0))


def label_reference(groups: list[dict], docs: Iterable[dict], labels: Iterable[dict]) -> list[dict]:
    """
    Đối chiếu nhóm với nhãn chuẩn: mỗi nhóm đếm số dòng nhãn `clean` và số dòng nhãn lỗi font (FONT_BAD_LABELS) của các bài thuộc nhóm đó (nối qua article_id), để chọn ngưỡng độ giống.

    Args:
        docs:   Mọi bài (có article_id, producer, creator), không chỉ bài đã thử.
        labels: Dòng nhãn chuẩn (cột article_id, label; nhãn `other` đã chia nhỏ nên truyền qua audit.refine_label trước nếu muốn).

    Returns:
        `groups` thêm hai khoá "labeled_clean", "labeled_bad".
    """
    group_by_article = {str(d.get("article_id")): group_of(d) for d in docs}
    clean: Counter = Counter()
    bad: Counter = Counter()
    for row in labels:
        g = group_by_article.get(str(row.get("article_id")))
        if g is None:
            continue
        clean[g] += row.get("label") == "clean"
        bad[g] += row.get("label") in FONT_BAD_LABELS
    return [{**g, "labeled_clean": clean[(g["producer"], g["creator"])], "labeled_bad": bad[(g["producer"], g["creator"])]} for g in groups]


def bad_groups(groups: list[dict], threshold: float) -> list[dict]:
    """Nhóm có độ giống trung bình dưới `threshold` (nhóm không đo được thì không tính là xấu)."""
    return [g for g in groups if g["similarity_mean"] is not None and g["similarity_mean"] < threshold]


def write_canary(groups: list[dict], out_dir: Path, threshold: float | None = None) -> None:
    """
    Ghi canary.json, canary.csv và (khi có `threshold`) font_groups_bad.json vào `out_dir`. Không có threshold thì KHÔNG ghi font_groups_bad.json: ngưỡng do chủ dự án chọn sau khi xem canary.csv.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "canary.json", groups)
    cols = ["producer", "creator", "docs_total", "docs_tested", "pages", "similarity_mean", "similarity_min", "labeled_clean", "labeled_bad"]
    with open(out_dir / "canary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(groups)
    if threshold is not None:
        bad = bad_groups(groups, threshold)
        write_json_atomic(out_dir / "font_groups_bad.json", {"threshold": threshold, "groups": [{k: g[k] for k in ("producer", "creator", "docs_total", "docs_tested", "similarity_mean")} for g in bad]})
        logger.info("Canary: %d / %d nhóm dưới ngưỡng %.3f -> font_groups_bad.json (%d bài của mẫu)", len(bad), len(groups), threshold, sum(g["docs_total"] for g in bad))


def read_labels(path: Path) -> list[dict]:
    """Đọc labeling_sample.csv (có cột article_id, label, note) thành danh sách dòng; nhãn `other` được đổi thành nhãn con theo audit.refine_label."""
    from vi_corpus.vjol.audit import refine_label

    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [{**r, "label": refine_label(r)} for r in rows]

