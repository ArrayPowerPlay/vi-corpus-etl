"""
Test xử lý thô giáo trình: clean_pages, kiểm kê/dedup theo sha256, checkpoint + resume và iter_records.
Không cần mô hình OCR (dùng bộ OCR giả), không cần mạng.
"""

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


def test_run_extract_checkpoint_resume_va_records(tmp_path):
    """Chạy đủ, lỗi/skip được ghi, chạy lại không làm lại; đường dẫn trùng nằm trong checkpoint; records sạch."""
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
