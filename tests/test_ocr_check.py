"""
Test cho bộ kiểm kết quả OCR chuỗi VLM (vi_corpus/stbook/ocr_check.py): số đo trang, phát hiện lệch số trang / trang rỗng,
tổng hợp theo engine, đọc log và ghi trang mẫu, loại chữ viết, ngôn ngữ theo trang, sách ngoại ngữ, trang lạ trong sách tiếng Việt
và phân bố tỷ lệ âm tiết thật của baseline (O-27 a, b). Dùng file OCR giả và bộ nhận diện heuristic, không cần mạng hay GPU.
"""

import csv
import json

import pymupdf

from vi_corpus.common.state import write_json_atomic
from vi_corpus.stbook.ocr_check import check, page_metrics, script_of
from vi_corpus.stbook.ocr_vlm import ENGINE_TAG

GOOD = "Người dân miền núi trồng chè trên những sườn đồi thoai thoải quanh năm. " * 8
ENGLISH = "The people of the mountain region have grown tea on the hills for many years and this is what they have done. " * 6
NONSENSE = " ".join(f"qzx{n}k" for n in range(80))  # không âm tiết nào nằm trong tập âm tiết của test
LAO = "ສະບາຍດີ ທຸກຄົນ ຍິນດີຕ້ອນຮັບ " * 20
NO_DIAC = "Nguoi dan mien nui trong che tren nhung suon doi thoai thoai quanh nam. " * 8


def _write_book(root, slug, pid, pages, tiers, pdf=None):
    """Ghi một file OCR giả theo định dạng của chuỗi VLM."""
    info = [{"status": "ok", "tier": t} for t in tiers]
    stats = {"pages": len(pages), "ok_by_tier": {t: tiers.count(t) for t in set(tiers)}, "blank": 0, "flagged": 0, "rerun": 0, "seconds": 10.0}
    write_json_atomic(root / "interim/stbook_ocr" / slug / f"{pid}.json", {
        "ocr": {"engine": ENGINE_TAG}, "pages": pages, "page_info": info, "stats": stats,
        "pdf_path": str(pdf) if pdf else "missing.pdf"})


def test_page_metrics_flags_missing_diacritics_and_loops():
    """Văn bản đủ dấu thì sạch; mất dấu, chữ Hán và dãy lặp bị gắn mã lý do."""
    assert page_metrics(GOOD)["reasons"] == []
    assert "low_diacritic" in page_metrics(NO_DIAC)["reasons"]
    assert "non_latin" in page_metrics(GOOD + "漢字漢字漢字漢字漢字" * 20)["reasons"]
    assert "loop" in page_metrics("Đầu trang\n" + " . " * 400)["reasons"]


def test_check_writes_summary_samples_and_catches_inconsistency(tmp_path):
    """Chạy bộ kiểm trên dữ liệu giả: tổng hợp theo engine, trang đáng nghi vào mẫu, cuốn lệch số trang bị báo."""
    pdf = tmp_path / "book.pdf"
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    doc.save(pdf)
    _write_book(tmp_path, "a", "1", [GOOD, GOOD, NO_DIAC], ["qwen3vl_8b_mt4k", "qwen3vl_8b_mt4k", "baseline"], pdf)
    _write_book(tmp_path, "a", "2", [GOOD, ""], ["qwen3vl_8b_mt4k"] * 2)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/stbook_ocr_vlm_gpu0.out").write_text("x [a/1] 3 trang, 1.50 trang/s | chọn\nWARNING Lỗi OCR x\n", encoding="utf-8")

    out = check(tmp_path, per_tier=2, n_suspect_samples=2, lang_model="heuristic")
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["books"] == 2 and s["pages"] == 5
    assert s["by_tier"]["baseline"]["suspect"] == {"low_diacritic": 1}
    assert any("văn bản rỗng" in p for p in s["problems_consistency"])
    assert s["logs"]["per_gpu"]["stbook_ocr_vlm_gpu0"]["pages_per_sec_p50"] == 1.5
    assert any("WARNING" in m for m in s["logs"]["messages"])
    assert (out / "suspect_pages.csv").exists() and (out / "books.csv").exists()
    assert list((out / "samples/tier_baseline").glob("*.txt")) and list((out / "samples/tier_baseline").glob("*.jpg"))


def test_script_of_picks_dominant_script():
    """Loại chữ viết theo nhóm có nhiều chữ cái nhất; không có chữ cái thì "none"."""
    assert script_of(GOOD) == "latin"
    assert script_of(LAO) == "lao"
    assert script_of("漢字漢字 ab") == "han"
    assert script_of("123 ... !!!") == "none"


def test_check_finds_foreign_books_odd_pages_and_baseline_syllables(tmp_path):
    """Sách phần lớn tiếng Anh vào bảng sách ngoại ngữ; trang tiếng Anh trong sách tiếng Việt vào danh sách trang lạ;
    trang baseline có tỷ lệ âm tiết thật và vào dải mẫu tương ứng."""
    vi_book = [GOOD] * 9 + [ENGLISH]
    _write_book(tmp_path, "vi", "1", vi_book, ["qwen3vl_8b_mt4k"] * 9 + ["qwen3vl_8b_rp110"])
    _write_book(tmp_path, "en", "2", [ENGLISH] * 4, ["qwen3vl_8b_mt4k"] * 4)
    _write_book(tmp_path, "bl", "3", [GOOD, NONSENSE], ["baseline", "baseline"])
    syl = tmp_path / "processed/ocr_bakeoff/syllables.json"
    syl.parent.mkdir(parents=True)
    syl.write_text(json.dumps({"syllables": sorted(set(GOOD.lower().replace(".", "").split()))}), encoding="utf-8")

    out = check(tmp_path, per_tier=3, n_suspect_samples=3, lang_model="heuristic")
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["lang_model"] == "heuristic"
    assert s["scripts_by_tier"]["baseline"] == {"latin": 2}
    assert s["foreign_books"]["n"] == 1 and s["foreign_books"]["top"][0][0] == "en/2"
    odd = s["odd_pages_in_vi_books"]
    assert odd["total"] == 1 and odd["by_tier"]["qwen3vl_8b_rp110"]["by_lang"] == {"en": 1}
    assert odd["top_books"] == [["qwen3vl_8b_rp110", "vi/1", 1]]
    rows = list(csv.DictReader(open(out / "baseline_pages.csv", encoding="utf-8")))
    assert [r["real_syllable_rate"] for r in rows] == ["0.0", "1.0"]  # sắp từ thấp đến cao
    assert s["syllables"]["real_rate_by_tier"]["baseline"]["n"] == 2
    assert s["syllables"]["baseline_histogram"]["0.95-1.00"] == 1
    assert (out / "samples/baseline_syllable_0-0.5").exists() and (out / "samples/baseline_syllable_0.85-1").exists()
    assert (out / "samples/odd_qwen3vl_8b_rp110").exists()


def test_check_without_syllables_file_skips_that_metric(tmp_path):
    """Thiếu syllables.json thì bộ kiểm vẫn chạy, chỉ bỏ phần tỷ lệ âm tiết thật."""
    _write_book(tmp_path, "a", "1", [GOOD, GOOD], ["baseline", "baseline"])
    s = json.loads((check(tmp_path, lang_model="heuristic") / "summary.json").read_text(encoding="utf-8"))
    assert s["syllables"] is None
