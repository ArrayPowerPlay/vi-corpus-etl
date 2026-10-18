"""
Test cho bộ kiểm kết quả OCR chuỗi VLM (vi_corpus/stbook/ocr_check.py): số đo trang, phát hiện lệch số trang / trang rỗng,
tổng hợp theo engine, đọc log và ghi trang mẫu. Dùng file OCR giả, không cần mạng hay GPU.
"""

import json

import pymupdf

from vi_corpus.common.state import write_json_atomic
from vi_corpus.stbook.ocr_check import check, page_metrics
from vi_corpus.stbook.ocr_vlm import ENGINE_TAG

GOOD = "Người dân miền núi trồng chè trên những sườn đồi thoai thoải quanh năm. " * 8
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

    out = check(tmp_path, per_tier=2, n_suspect_samples=2)
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["books"] == 2 and s["pages"] == 5
    assert s["by_tier"]["baseline"]["suspect"] == {"low_diacritic": 1}
    assert any("văn bản rỗng" in p for p in s["problems_consistency"])
    assert s["logs"]["per_gpu"]["stbook_ocr_vlm_gpu0"]["pages_per_sec_p50"] == 1.5
    assert any("WARNING" in m for m in s["logs"]["messages"])
    assert (out / "suspect_pages.csv").exists() and (out / "books.csv").exists()
    assert list((out / "samples/tier_baseline").glob("*.txt")) and list((out / "samples/tier_baseline").glob("*.jpg"))
