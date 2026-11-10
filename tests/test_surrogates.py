"""
Test lỗi surrogate lẻ làm dừng cả lượt trích text (lần chạy full VJOL 2026-10-29).

Lỗi gốc: PDF có bảng ToUnicode hỏng cho ra "\\ud835" lẻ; json.dump(ensure_ascii=False) ghi UTF-8 ném UnicodeEncodeError ở
run_batch, tiến trình chết, flock nhả, tiến trình GPU kế tiếp nhận lại đúng file đó và chết theo.
"""

from pathlib import Path

import pymupdf

from vi_corpus.common import pdf_text
from vi_corpus.common.batch_extract import file_sha256, run_batch
from vi_corpus.common.state import read_json
from vi_corpus.common.unicode_clean import scrub_obj, scrub_surrogates

BAD = "Công thức \ud835 và \ud853 hết"


def _make_pdf(path: Path) -> None:
    """Tạo PDF một trang có đủ chữ để được coi là trang có lớp chữ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), f"Day la mot trang co du chu de duoc coi la lop chu that su cua PDF {path.name}.")  # tên file làm nội dung khác nhau
    doc.save(path)


def test_scrub_surrogates_va_scrub_obj():
    """Surrogate lẻ thành U+FFFD ở mọi tầng dict / list; chuỗi sạch và kiểu khác giữ nguyên; ghi UTF-8 được."""
    assert scrub_surrogates("abc") == ("abc", 0)
    assert scrub_surrogates(BAD) == ("Công thức � và � hết", 2)
    obj = {"pages": [{"text": BAD, "n": 3}], "ok": None, "t": ("x", "\ud835")}
    clean, n = scrub_obj(obj)
    assert n == 3 and clean["pages"][0]["n"] == 3 and clean["ok"] is None and clean["t"] == ("x", "�")
    assert "\ud835" in obj["pages"][0]["text"]  # không sửa tại chỗ
    str(clean).encode("utf-8")


def test_extract_pdf_thay_surrogate_va_ghi_log_duong_dan(tmp_path, monkeypatch, caplog):
    """extract_pdf nhận text có surrogate lẻ: trang ra sạch, log cảnh báo nêu đường dẫn file và số trang."""
    pdf = tmp_path / "bad.pdf"
    _make_pdf(pdf)
    monkeypatch.setattr(pdf_text, "page_text", lambda page, tcvn3_all=False: "Trang " + BAD + " " + "x" * 50)
    with caplog.at_level("WARNING", logger="vi_corpus"):
        pages = pdf_text.extract_pdf(pdf, lambda: None)
    assert "\ud835" not in pages[0]["text"] and "�" in pages[0]["text"]
    assert str(pdf) in caplog.text and "trang 1" in caplog.text


def test_run_batch_khong_chet_vi_surrogate(tmp_path):
    """extract_file trả text có surrogate lẻ: checkpoint vẫn được ghi (đã sạch), lượt chạy không dừng, không đếm là lỗi."""
    root = tmp_path / "raw"
    root.mkdir()
    (root / "a.pdf").write_bytes(b"one")
    (root / "b.pdf").write_bytes(b"two")

    def extract_file(path, get_ocr):
        return {"format": "pdf", "pages": [{"text": BAD, "method": "text"}], "error": None, "skipped": None}

    failed = run_batch(root, tmp_path, "interim/t", extract_file, lambda p: {}, lambda: None, ocr_enabled=False)
    assert failed == 0
    done = read_json(tmp_path / "interim/t" / f"{file_sha256(root / 'a.pdf')}.json")
    assert done["pages"][0]["text"] == "Công thức � và � hết"


def test_run_batch_loi_ghi_checkpoint_thanh_loi_cua_file_do(tmp_path):
    """Dữ liệu không ghi được JSON (không phải surrogate): file đó thành checkpoint lỗi, file sau vẫn được làm, lượt chạy không dừng."""
    root = tmp_path / "raw"
    root.mkdir()
    (root / "a.pdf").write_bytes(b"one")
    (root / "b.pdf").write_bytes(b"two")

    def extract_file(path, get_ocr):
        pages = [{"text": object(), "method": "text"}] if path.name == "a.pdf" else [{"text": "ok", "method": "text"}]
        return {"format": "pdf", "pages": pages, "error": None, "skipped": None}

    failed = run_batch(root, tmp_path, "interim/t", extract_file, lambda p: {}, lambda: None, ocr_enabled=False)
    assert failed == 1
    bad = read_json(tmp_path / "interim/t" / f"{file_sha256(root / 'a.pdf')}.json")
    assert bad["error"].startswith("CheckpointWriteError") and bad["pages"] == []
    assert read_json(tmp_path / "interim/t" / f"{file_sha256(root / 'b.pdf')}.json")["pages"][0]["text"] == "ok"


def test_find_surrogate_pdfs_tim_file_chua_co_checkpoint(tmp_path, monkeypatch, capsys):
    """scripts/vjol/find_surrogate_pdfs.py: chỉ quét PDF chưa có checkpoint, tìm đúng file có surrogate lẻ, chép ra thư mục kết quả và ghi report.json."""
    import importlib.util
    import json
    import sys

    from vi_corpus.vjol.extract import TEXT_DIR

    for name in ("a", "b", "c"):
        _make_pdf(tmp_path / "raw" / "vjol" / "2020" / f"{name}_1.pdf")
    # b_1.pdf đã có checkpoint nên phải bị bỏ qua dù có surrogate
    sha_b = file_sha256(tmp_path / "raw" / "vjol" / "2020" / "b_1.pdf")
    (tmp_path / TEXT_DIR).mkdir(parents=True)
    (tmp_path / TEXT_DIR / f"{sha_b}.json").write_text("{}", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("find_surrogate_pdfs", Path(__file__).resolve().parents[1] / "scripts/vjol/find_surrogate_pdfs.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "find_surrogate_pdfs", mod)  # để multiprocessing đọc được hàm scan_pdf theo tên module
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "page_text", lambda page: "x" + BAD if page.parent.name.endswith(("b_1.pdf", "c_1.pdf")) else "sạch")
    monkeypatch.setattr(sys, "argv", ["find", "--data-root", str(tmp_path), "--workers", "1"])
    assert mod.main() == 0
    report = json.loads((tmp_path / "processed" / "vjol_surrogate_pdfs" / "report.json").read_text(encoding="utf-8"))
    assert [Path(h["path"]).name for h in report["hits"]] == ["c_1.pdf"]
    assert report["hits"][0]["codes"] == ["U+D835", "U+D853"] and report["hits"][0]["pages"] == [1]
    assert (tmp_path / "processed" / "vjol_surrogate_pdfs" / "c_1.pdf").exists()
