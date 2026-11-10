"""
Test cho bộ lọc trang OCR stbook (vi_corpus/stbook/ocr_filter.py, O-30 mục 2 và 3): trang baseline tỷ lệ âm tiết thật thấp và trang rp110
khác tiếng Việt trong sách tiếng Việt bị chuyển sang flagged; giữ văn bản, cập nhật stats, chạy lại được, đổi ngưỡng được.
Dùng file OCR giả và bộ nhận diện heuristic, không cần mạng hay GPU.
"""

import csv
import json

import pytest

from vi_corpus.common.state import read_json, write_json_atomic
from vi_corpus.stbook.ocr_filter import filter_pages
from vi_corpus.stbook.ocr_vlm import ENGINE_TAG, claim_book

GOOD = "Người dân miền núi trồng chè trên những sườn đồi thoai thoải quanh năm. " * 8
ENGLISH = "The people of the mountain region have grown tea on the hills for many years and this is what they have done. " * 6
NONSENSE = " ".join(f"qzx{n}k" for n in range(80))  # không âm tiết nào nằm trong tập âm tiết của test
SHORT_NONSENSE = "qzx qzy qzz"  # dưới 30 từ: không tính được tỷ lệ
MT4K, RP110 = "qwen3vl_8b_mt4k", "qwen3vl_8b_rp110"


def _book(root, slug, pid, pages, tiers):
    """Ghi một file OCR giả theo định dạng của chuỗi VLM và trả đường dẫn."""
    stats = {"pages": len(pages), "ok_by_tier": {t: tiers.count(t) for t in set(tiers)}, "blank": 0, "flagged": 0, "rerun": 0, "seconds": 1.0}
    path = root / "interim/stbook_ocr" / slug / f"{pid}.json"
    write_json_atomic(path, {"ocr": {"engine": ENGINE_TAG}, "pages": pages, "page_info": [{"status": "ok", "tier": t} for t in tiers], "stats": stats})
    return path


@pytest.fixture
def root(tmp_path):
    """Thư mục dữ liệu giả có tập âm tiết từ GOOD."""
    syl = tmp_path / "processed/ocr_bakeoff/syllables.json"
    syl.parent.mkdir(parents=True)
    syl.write_text(json.dumps({"syllables": sorted(set(GOOD.lower().replace(".", "").split()))}), encoding="utf-8")
    return tmp_path


def test_filters_low_syllable_baseline_and_foreign_rp110_page(root):
    """Baseline chữ rác và rp110 tiếng Anh trong sách tiếng Việt bị loại, còn lại giữ nguyên; stats và văn bản cũ đúng."""
    vi = _book(root, "vi", "1", [GOOD] * 8 + [ENGLISH, ENGLISH], [MT4K] * 8 + [RP110, MT4K])  # trang tiếng Anh của mt4k không thuộc phạm vi lọc
    bl = _book(root, "bl", "2", [GOOD, NONSENSE, SHORT_NONSENSE], ["baseline"] * 3)
    en = _book(root, "en", "3", [ENGLISH] * 4, [RP110] * 4)  # sách ngoại ngữ: không đụng vào rp110

    s = filter_pages(root, lang_model="heuristic")
    assert s["pages_flagged"] == 2 and s["by_reason"] == {"low_real_syllable": 1, "foreign_page_in_vi_book": 1}
    d = read_json(vi)
    assert d["pages"][8] == "" and d["page_info"][8]["status"] == "flagged" and d["page_info"][8]["rejected_text"] == ENGLISH
    assert d["page_info"][8]["post_filter"] == "foreign_page_in_vi_book" and d["page_info"][9]["status"] == "ok"
    assert d["stats"]["flagged"] == 1 and d["stats"]["ok_by_tier"] == {MT4K: 9}
    d = read_json(bl)
    assert [p["status"] for p in d["page_info"]] == ["ok", "flagged", "ok"]  # trang ngắn giữ nguyên
    assert d["pages"][1] == "" and d["page_info"][1]["rejected_text"] == NONSENSE and d["stats"]["ok_by_tier"] == {"baseline": 2}
    assert d["post_filter"]["by_reason"] == {"low_real_syllable": 1}
    assert read_json(en)["stats"]["flagged"] == 0
    rows = list(csv.DictReader(open(root / "processed/ocr_page_filter/flagged_pages.csv", encoding="utf-8")))
    assert {(r["book"], r["page"]) for r in rows} == {("vi/1", "9"), ("bl/2", "2")}


def test_rerun_is_idempotent_and_threshold_change_restores_pages(root):
    """Chạy lại không đổi file; hạ ngưỡng thì trang đã loại được trả lại nguyên văn bản."""
    bl = _book(root, "bl", "2", [GOOD, NONSENSE], ["baseline"] * 2)
    filter_pages(root, lang_model="heuristic")
    first = read_json(bl)
    assert filter_pages(root, lang_model="heuristic")["books_changed"] == 0
    assert read_json(bl) == first  # không ghi lại, kể cả applied_at
    s = filter_pages(root, lang_model="heuristic", min_real_rate=0.0)
    assert s["pages_flagged"] == 0 and s["books_changed"] == 1
    d = read_json(bl)
    assert d["pages"] == [GOOD, NONSENSE] and d["stats"]["flagged"] == 0 and d["stats"]["ok_by_tier"] == {"baseline": 2}
    assert "post_filter" not in d and all(set(p) == {"status", "tier"} for p in d["page_info"])


def test_dry_run_does_not_touch_files_and_missing_syllables_raises(root):
    """--dry-run chỉ báo cáo; thiếu tập âm tiết thì báo lỗi rõ."""
    bl = _book(root, "bl", "2", [GOOD, NONSENSE], ["baseline"] * 2)
    before = read_json(bl)
    assert filter_pages(root, lang_model="heuristic", dry_run=True)["pages_flagged"] == 1
    assert read_json(bl) == before
    with pytest.raises(FileNotFoundError):
        filter_pages(root, lang_model="heuristic", syllables=root / "nope.json")


def test_book_being_ocrd_is_skipped(root):
    """Cuốn đang được tiến trình OCR giữ khoá thì bị bỏ qua và ghi trong báo cáo."""
    bl = _book(root, "bl", "2", [GOOD, NONSENSE], ["baseline"] * 2)
    before = read_json(bl)
    with claim_book(root, "bl", "2") as got:
        assert got
        s = filter_pages(root, lang_model="heuristic")
    assert s["skipped_books"] == ["bl/2"] and read_json(bl) == before
