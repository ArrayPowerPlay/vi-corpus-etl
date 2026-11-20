"""
Test xử lý thô giáo trình: clean_pages, kiểm kê/dedup theo sha256, checkpoint + resume và iter_records.
Không cần mô hình OCR (dùng bộ OCR giả), không cần mạng.
"""

import json

import pymupdf

from vi_corpus.common.pdf_text import clean_pages
from vi_corpus.common.registry import SOURCES
from vi_corpus.common.schema import CORPUS_SCHEMA, lineage_rate
from vi_corpus.giao_trinh.extract import checkpoint_path, inventory, run_extract, status
from vi_corpus.giao_trinh.records import export_parquet, iter_records


def test_clean_pages_bo_header_footer_so_trang_va_ghep_dong():
    """Tiêu đề chạy (chữ số -> #) lặp >= 3 trang, số trang bị bỏ; dòng ghép thành đoạn, nối từ gạch nối."""
    pages = [
        f"Giáo trình Mật mã\nChương {i}: Mở đầu\nMật mã học là ngành nghiên\ncứu về an toàn thông tin. Nó có exam-\nple rõ ràng.\nBản mã là gì?\nHết trang {i}\n{i}"
        for i in range(1, 4)
    ]
    out = clean_pages(pages)
    assert "Giáo trình Mật mã" not in out and "Chương" not in out
    assert not any(p.strip().isdigit() for p in out.split("\n\n"))
    assert "nghiên cứu về an toàn thông tin. Nó có example rõ ràng." in out
    # câu hỏi kết thúc bằng "?" và dòng sau (trang sau) viết hoa -> xuống đoạn mới
    assert "Bản mã là gì?\n\nMật mã học" in out


def test_clean_pages_giu_dong_khi_khong_lap_va_khong_ghep_slide():
    """Dòng đầu trang chỉ ở 1-2 trang thì giữ; slides=True giữ mỗi dòng một đoạn."""
    out = clean_pages(["Tiêu đề\nnội dung", "Tiêu đề\nkhác"])
    assert out == "Tiêu đề nội dung Tiêu đề khác"
    assert clean_pages(["Ý một\nÝ hai"], slides=True) == "Ý một\n\nÝ hai"
    # slide: dòng lặp cấu trúc ở >= 3 trang vẫn giữ (không coi là tiêu đề chạy)
    assert clean_pages([f"Bài {i}\nNội dung {i}" for i in range(3)], slides=True).count("Bài") == 3


def _make_pdf(path, text):
    """Tạo PDF một trang có lớp chữ (đủ dài để không bị coi là trang scan)."""
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _make_root(tmp_path):
    """Cây raw/giao_trinh: 1 PDF, bản sao ở môn khác, 1 .ppt giả, 1 PDF hỏng, 1 file ẩn."""
    root = tmp_path / "raw/giao_trinh"
    (root / "Toán/Giải tích").mkdir(parents=True)
    (root / "Vật lý/Cơ học").mkdir(parents=True)
    _make_pdf(root / "Toán/Giải tích/a.pdf", "The derivative of a function is the limit of the ratio of increments.")
    (root / "Vật lý/Cơ học/b_copy.pdf").write_bytes((root / "Toán/Giải tích/a.pdf").read_bytes())
    (root / "Vật lý/Cơ học/x.ppt").write_bytes(b"fake ppt")
    (root / "Vật lý/Cơ học/bad.pdf").write_bytes(b"khong phai pdf")
    (root / ".cache").mkdir()
    (root / ".cache/an.pdf").write_bytes(b"x")
    return root


def test_inventory_gom_trung_sha256_va_bo_file_an(tmp_path):
    """Hai đường dẫn cùng nội dung gom thành một nhóm; file trong thư mục ẩn bị bỏ."""
    root = _make_root(tmp_path)
    groups = inventory(root)
    assert sorted(len(v) for v in groups.values()) == [1, 1, 2]
    dup = next(v for v in groups.values() if len(v) == 2)
    assert [p.name for p in dup] == ["a.pdf", "b_copy.pdf"]


def test_run_extract_checkpoint_resume_va_records(tmp_path, monkeypatch):
    """Chạy đủ, lỗi/skip được ghi, chạy lại không làm lại; đường dẫn trùng nằm trong checkpoint; records sạch."""
    from vi_corpus.common.parsers import office

    monkeypatch.setattr(office, "office_binary", lambda: None)  # .ppt giả: coi như máy chưa cài LibreOffice -> skipped
    root = _make_root(tmp_path)
    calls = []
    failed = run_extract(root, tmp_path, lambda: calls.append(1))
    assert failed == 1 and not calls  # PDF hỏng lỗi; trang có chữ nên không nạp OCR
    s = status(root, tmp_path)
    assert (s["checkpoints"], s["errors"], s["skipped"], s["pages"]) == (3, 1, 1, {"text": 1})

    sha = next(v for v in inventory(root).values() if len(v) == 2)
    from vi_corpus.giao_trinh.extract import file_sha256
    done = checkpoint_path(tmp_path, file_sha256(sha[0]))
    before = done.stat().st_mtime_ns
    assert run_extract(root, tmp_path, lambda: None) == 1  # chỉ file lỗi được thử lại
    assert done.stat().st_mtime_ns == before

    rows = list(iter_records(SOURCES["giao_trinh"], tmp_path))
    assert len(rows) == 1 and set(rows[0]) == set(CORPUS_SCHEMA.names)
    assert rows[0]["source_path"] == "raw/giao_trinh/Toán/Giải tích/a.pdf"
    assert lineage_rate(rows) == 1.0
    assert export_parquet(rows, tmp_path / "out.parquet") == 1


def test_trang_scan_dung_ocr_gia_va_needs_ocr_duoc_lam_lai(tmp_path):
    """Trang chỉ có ảnh: --no-ocr cho needs_ocr; chạy lại có OCR thì OCR lại (bộ OCR giả)."""
    import io

    from PIL import Image

    root = tmp_path / "raw/giao_trinh/Toán/Đại số"
    root.mkdir(parents=True)
    buf = io.BytesIO()
    Image.new("RGB", (50, 50), "white").save(buf, "PNG")
    doc = pymupdf.open()
    doc.new_page().insert_image(pymupdf.Rect(0, 0, 200, 200), stream=buf.getvalue())
    doc.save(root / "scan.pdf")
    doc.close()

    class FakeOcr:
        """Bộ OCR giả: trả text cố định."""

        def page_text(self, image):
            """Trả text giả cho mọi ảnh."""
            return "Nội dung OCR giả"

    assert run_extract(root.parent.parent, tmp_path, lambda: None, ocr_enabled=False) == 0
    assert status(root.parent.parent, tmp_path)["pages"] == {"needs_ocr": 1}
    assert list(iter_records(SOURCES["giao_trinh"], tmp_path)) == []  # trang needs_ocr không thành text
    assert run_extract(root.parent.parent, tmp_path, lambda: FakeOcr()) == 0
    assert status(root.parent.parent, tmp_path)["pages"] == {"ocr": 1}
    assert list(iter_records(SOURCES["giao_trinh"], tmp_path))[0]["text"] == "Nội dung OCR giả"


def _make_scan_pdf(path, n_pages=2):
    """Tạo PDF `n_pages` trang chỉ có ảnh (trang scan)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (50, 50), "white").save(buf, "PNG")
    doc = pymupdf.open()
    for _ in range(n_pages):
        doc.new_page().insert_image(pymupdf.Rect(0, 0, 200, 200), stream=buf.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


class FakeCascadeGetter:
    """get_ocr giả có đúng giao diện CascadeOcrGetter: có signature(), trả bộ OCR có ocr_pages; đếm số trang đã OCR."""

    def __init__(self, texts=None, error=None):
        """texts: text từng trang (None = trang bị loại flagged); error: ngoại lệ ném khi gọi ocr_pages."""
        self.texts, self.error, self.ocr_calls = texts or ["Trang một.", None], error, 0

    def signature(self):
        """Chữ ký bộ OCR giả."""
        return "fake"

    def __call__(self):
        """Trả chính nó làm bộ OCR."""
        return self

    def ocr_pages(self, pages, render=None):
        """OCR giả: mỗi trang một kết quả theo self.texts; ném self.error nếu có."""
        if self.error:
            raise self.error
        pages = list(pages)
        self.ocr_calls += len(pages)
        out = []
        for i in range(len(pages)):
            text = self.texts[i % len(self.texts)]
            out.append({"text": text or "", "method": "ocr_vlm", "tier": "qwen3vl_8b_mt4k",
                        "status": "ok" if text else "flagged", **({} if text else {"rejected_text": "rác"})})
        return out


def test_ocr_vlm_trang_flagged_bi_bo_va_cache_ocr_dung_lai(tmp_path):
    """Chuỗi OCR VLM: trang ok vào bản ghi, trang flagged thì không; meta đếm trang OCR; xoá checkpoint rồi chạy lại thì lấy từ cache, không OCR lại."""
    root = tmp_path / "raw/giao_trinh"
    _make_scan_pdf(root / "Toán/Đại số/scan.pdf")
    getter = FakeCascadeGetter()
    assert run_extract(root, tmp_path, getter) == 0 and getter.ocr_calls == 2
    s = status(root, tmp_path)
    assert s["pages"] == {"ocr_vlm": 2} and s["ocr_status"] == {"ok": 1, "flagged": 1} and s["ocr_tier"] == {"qwen3vl_8b_mt4k": 2}
    rec = next(iter_records(SOURCES["giao_trinh"], tmp_path, with_meta=True))
    assert rec["text"] == "Trang một."
    meta = json.loads(rec["meta"])
    assert (meta["pages"], meta["ocr_pages"], meta["ocr_flagged_pages"], meta["needs_ocr_pages"]) == (2, 2, 1, 0)
    assert any((tmp_path / "interim/giao_trinh_ocr_cache").rglob("*.json"))

    for ckpt in (tmp_path / "interim/giao_trinh_text").glob("*.json"):
        ckpt.unlink()
    again = FakeCascadeGetter()
    assert run_extract(root, tmp_path, again) == 0
    assert again.ocr_calls == 0  # cả hai trang lấy từ cache
    assert status(root, tmp_path)["pages"] == {"ocr_vlm": 2}


def test_chuoi_ocr_hong_dung_ca_luot_khong_ghi_checkpoint(tmp_path):
    """ServerDown / EngineUnavailable không được ghi thành lỗi của file: ném lên để dừng cả lượt, checkpoint không có."""
    import pytest

    from vi_corpus.ocr_bakeoff.cascade import EngineUnavailable, ServerDown

    root = tmp_path / "raw/giao_trinh"
    _make_scan_pdf(root / "Toán/Đại số/scan.pdf")
    for exc in (ServerDown("server chết"), EngineUnavailable("thiếu libGL")):
        with pytest.raises(type(exc)):
            run_extract(root, tmp_path, FakeCascadeGetter(error=exc))
        assert not list((tmp_path / "interim/giao_trinh_text").glob("*.json"))


def test_nhieu_tien_trinh_dung_khoa_nhan_file(tmp_path):
    """multi_process=True: file được nhận bằng khoá flock ở state/giao_trinh_claims/ (mỗi sha256 một file khoá)."""
    root = _make_root(tmp_path)
    run_extract(root, tmp_path, lambda: None, multi_process=True)
    assert len(list((tmp_path / "state/giao_trinh_claims").glob("*.lock"))) == 3
