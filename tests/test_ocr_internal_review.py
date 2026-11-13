"""
Test scripts/ocr_internal/review_report.py: dựng một lần chạy thử giả (PDF thật + kết quả OCR giả ghi bằng build_report của trial_vjol_vista) rồi kiểm
index.html, trang chi tiết, ảnh, zip, giải thích thông số và cách xử lý file lỗi.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ocr_internal_trial import TEXT, make_pdf, trial  # noqa: E402  (dùng lại bộ dựng PDF thử)

spec = importlib.util.spec_from_file_location("review_report", Path(trial.__file__).parent / "review_report.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


@pytest.fixture
def run_dir(tmp_path):
    """Lần chạy giả: 1 file 2 lớp OCR đúng, 1 file 2 lớp OCR lệch hẳn, 1 file 1 lớp, 1 file lỗi API; trả về thư mục data-root."""
    out = tmp_path / "processed" / "ocr_internal_trial" / "trial"
    sample = []
    for name, kind in (("good", "image_text"), ("bad", "image_text"), ("scan", "scan"), ("err", "scan")):
        p = tmp_path / "raw" / "vjol" / "2020" / f"{name}.pdf"
        make_pdf(p, kind, pages=3)
        cls = trial.classify_pdf(p)
        sample.append({"source": "vjol", "path": str(p), "rel": f"2020/{name}.pdf", "sha256": trial.sha256_of(p), "size_mb": 0.01, **cls})
    for e in sample:
        pages = {"1": "x", "2": "y"}
        if e["rel"].endswith("good.pdf"):
            pages = {str(i + 1): f"{TEXT} good trang {i}" for i in range(3)}
        elif e["rel"].endswith("bad.pdf"):
            pages = {str(i + 1): "hoàn toàn khác nội dung lớp chữ hoàn toàn" for i in range(3)}
        rec = {**e, "exporter": "json"}
        if e["rel"].endswith("err.pdf"):
            rec.update(status="error", error="HTTP 500 thử")
        else:
            text = "\n\n".join(pages.values())
            rec.update(status="ok", session_id="sess-" + e["rel"][5:9], elapsed_s=3.0, billed_pages=3, total_pages=3, warnings=[], ocr_chars=len(text),
                       ocr_pages_with_text=len(pages), text=text, ocr_pages=pages)
        trial.write_atomic(out / "results" / f"{e['sha256']}.json", json.dumps(rec, ensure_ascii=False))
    trial.write_atomic(out / "sample.json", json.dumps({"sample": sample}))
    trial.build_report(out, sample, None)
    return tmp_path


def test_pick_pages():
    """Rải đều gồm trang đầu + cuối; k lớn hơn số trang hoặc 0 thì lấy hết."""
    assert review.pick_pages(10, 3) == [0, 4, 5, 9] or review.pick_pages(10, 3)[0] == 0 and review.pick_pages(10, 3)[-1] == 9
    assert review.pick_pages(2, 3) == [0, 1] and review.pick_pages(5, 0) == list(range(5)) and review.pick_pages(5, 1) == [0]


def test_review_build(run_dir, capsys):
    """Dựng đủ index + 3 trang chi tiết (không dựng cho file lỗi) + ảnh + zip; file OCR lệch có cờ và hiện số trùng thấp; file lỗi hiện lý do."""
    assert review.main(["--data-root", str(run_dir), "--pages", "2"]) == 0
    out = run_dir / "processed" / "ocr_internal_trial" / "trial"
    rv = out / "review"
    index = (rv / "index.html").read_text(encoding="utf-8")
    assert len(list((rv / "files").glob("*.html"))) == 3 and len(list((rv / "images").glob("*.jpg"))) == 6  # 3 file x 2 trang
    assert (out / "review.zip").exists()
    assert "Cách tính" in index and "Dice" in index and "lệch_lớp_chữ" in index and "HTTP 500 thử" in index
    bad_id = next(p for p in (rv / "files").glob("*.html") if "hoàn toàn khác" in p.read_text(encoding="utf-8"))
    page = bad_id.read_text(encoding="utf-8")
    assert "PDF GỐC" in page and "LỚP CHỮ CÓ SẴN" in page and "trùng lớp chữ của trang" in page
    assert any("PDF 1 lớp" in p.read_text(encoding="utf-8") for p in (rv / "files").glob("*.html"))


def test_explain_and_missing(tmp_path, capsys):
    """--explain in đủ mọi thông số; thiếu report thì thoát mã 2."""
    assert review.main(["--explain"]) == 0
    out = capsys.readouterr().out
    assert all(g["name"] in out for g in review.GLOSSARY)
    assert review.main(["--data-root", str(tmp_path)]) == 2
