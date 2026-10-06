"""
Test cho nguồn stbook (vi_corpus.stbook.ocr_books) và phần gom dòng của OCR, không cần mô hình.
"""

import json

from vi_corpus.common.schema import CORPUS_SCHEMA, lineage_rate
from vi_corpus.common.registry import SOURCES
from vi_corpus.stbook.ocr_books import find_books, iter_records, ocr_status


def _make_stbook(root):
    """Tạo cây thư mục giống stbook-crawler: 1 sách có PDF, 1 sách có giá, 1 sách đang tải dở."""
    cat = root / "raw/stbook/kinh-dien"
    (cat / "content/3_pages").mkdir(parents=True)
    (cat / "content/1.pdf").write_bytes(b"%PDF")
    books = [{"product_id": "1", "title": "A"}, {"product_id": "2"}, {"product_id": "3"}]
    (cat / "books.json").write_text(json.dumps({"books": books}), encoding="utf-8")
    (root / "raw/stbook/hong").mkdir()
    (root / "raw/stbook/hong/books.json").write_text("{hỏng", encoding="utf-8")


def test_find_books_chi_lay_sach_co_pdf(tmp_path):
    """Chỉ sách đã có <id>.pdf được tính; books.json hỏng bị bỏ qua, không crash."""
    _make_stbook(tmp_path)
    found = list(find_books(tmp_path / "raw/stbook"))
    assert [(slug, b["product_id"]) for slug, b, _ in found] == [("kinh-dien", "1")]
    assert ocr_status(tmp_path / "raw/stbook", tmp_path) == (0, 1)


def test_iter_records_tu_ket_qua_ocr(tmp_path):
    """Kết quả OCR thành bản ghi schema chung, source_path trỏ về PDF gốc, lineage 100%."""
    _make_stbook(tmp_path)
    out = tmp_path / "interim/stbook_ocr/kinh-dien"
    out.mkdir(parents=True)
    (out / "1.json").write_text(json.dumps({
        "pdf_path": "raw/stbook/kinh-dien/content/1.pdf", "pages": ["trang một", "trang hai"],
    }), encoding="utf-8")
    (out / "2.json.tmp").write_text("dở", encoding="utf-8")  # file tạm khi đang ghi, phải bỏ qua

    rows = list(iter_records(SOURCES["stbook"], tmp_path))
    assert len(rows) == 1 and set(rows[0]) == set(CORPUS_SCHEMA.names)
    assert rows[0]["text"] == "trang một\n\ntrang hai"
    assert rows[0]["source_path"] == "raw/stbook/kinh-dien/content/1.pdf"
    assert ocr_status(tmp_path / "raw/stbook", tmp_path) == (1, 1)
    assert lineage_rate(rows) == 1.0


def test_group_lines_theo_thu_tu_doc():
    """Khung cùng hàng gom thành một dòng (trái sang phải), các dòng xếp từ trên xuống."""
    from vi_corpus.common.ocr import group_lines

    boxes = [(500, 12, 900, 48), (10, 100, 400, 140), (10, 10, 400, 50)]
    assert group_lines(boxes) == [[(10, 10, 400, 50), (500, 12, 900, 48)], [(10, 100, 400, 140)]]


def _fake_requests(monkeypatch, calls):
    """Thay requests.get bằng bản trả về một file zip hợp lệ (đếm số lần tải, tải chậm để dễ lộ tranh chấp)."""
    import io
    import time
    import zipfile

    import requests

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("w.txt", "x" * 1000)
    data = buf.getvalue()

    class Resp:
        """Phản hồi giả."""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            """Không lỗi."""

        def iter_content(self, chunk_size):
            """Trả file theo hai nửa, nghỉ giữa chừng."""
            calls.append(1)
            yield data[: len(data) // 2]
            time.sleep(0.2)
            yield data[len(data) // 2:]

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())


def test_ensure_rec_weights_tai_lai_file_hong_va_khong_tranh_chap(tmp_path, monkeypatch):
    """File trọng số hỏng bị tải lại; nhiều luồng cùng gọi thì chỉ tải một lần và đều nhận file nguyên vẹn."""
    import tempfile
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    from vi_corpus.common.ocr import ensure_rec_weights

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    (tmp_path / "w.pth").write_bytes(b"tai do dang")  # file hỏng của lần chạy trước
    calls: list[int] = []
    _fake_requests(monkeypatch, calls)

    with ThreadPoolExecutor(4) as pool:
        paths = list(pool.map(lambda _: ensure_rec_weights("https://x.test/w.pth"), range(4)))

    assert set(paths) == {str(tmp_path / "w.pth")}
    assert zipfile.is_zipfile(tmp_path / "w.pth")
    assert len(calls) == 1
    assert not (tmp_path / "w.pth.part").exists()
