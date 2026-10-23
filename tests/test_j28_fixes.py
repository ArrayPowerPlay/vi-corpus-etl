"""
Kiểm các sửa J-28, rút ra từ kết quả chạy vjol_10k_v4a trên server: đo mất dấu không tính đường dẫn / tiếng Anh / thuật ngữ lặp một từ,
gạch nối thay ư / ơ không khớp "C- section", và che số điện thoại không đụng vào số liệu bảng.
"""

from vi_corpus.common.garbled import hyphen_lost, unaccented_ratio
from vi_corpus.vjol.clean import find_phones, mask_phones

LEXICON = frozenset("va cua cac duoc khong nhung mot voi nguoi viec dong dang tien hanh phat trien hoc".split())
BARE = ("Viec phat trien cua doanh nghiep khong chi dua vao von ma con dua vao nguoi lao dong va cong nghe, "
        "nhung mot so hoc gia cho rang cac chinh sach duoc ap dung voi tien do cham. ")


def test_chu_tran_that_van_bi_do():
    """Tiếng Việt mất dấu hoàn toàn cho tỷ lệ rất cao, xa ngưỡng 0,02."""
    ratio, n = unaccented_ratio(BARE * 6, LEXICON)
    assert n >= 100 and ratio > 0.15


def test_duong_dan_khong_tinh_la_chu_mat_dau():
    """Tên bài trong URL ("Trung-Quoc-dang-tranh-chap-voi-nhung-quoc-gia") và slug bị ngắt dòng không làm đoạn có dấu đúng thành mất dấu."""
    accented = "Các nhà nghiên cứu đã phân tích tình hình tranh chấp lãnh thổ ở Biển Đông trong giai đoạn từ năm 2012 đến nay. " * 6
    urls = "http://infonet.vn/The-gioi/Trung-Quoc-dang-tranh-chap-lanh-tho-voi-nhung-quoc-gia-nao/25212.info Giac-mong-Trung-Hoa-va-hanh-dong-ngang-nguoc-cua-Trung-Quoc " * 4
    assert unaccented_ratio(accented + urls, LEXICON)[0] < 0.02


def test_tieng_anh_khong_bi_do_mat_dau():
    """Đoạn tiếng Anh có "day", "mat"... trùng chữ trần tiếng Việt không bị tính (tỷ lệ 0); đoạn tiếng Việt không dấu vẫn bị tính."""
    english = "The students of the school have been reading the day and the mat with the teachers, which was not easy for them. " * 5
    lexicon = LEXICON | {"day", "mat", "the"}
    assert unaccented_ratio(english, lexicon)[0] == 0.0
    assert unaccented_ratio(BARE * 6, lexicon)[0] > 0.1


def test_thuat_ngu_lap_mot_tu_khong_lam_hong_doan():
    """Kế toán viết "sản phẩm dở dang" hàng chục lần: mỗi từ chỉ tính tối đa 3 lần nên tỷ lệ vẫn dưới 0,02."""
    text = "Chi phí sản phẩm dở dang cuối kỳ được tính theo phương pháp bình quân đối với từng loại nguyên vật liệu trực tiếp trong kỳ. " * 3
    text += "dang " * 20 + "Kế toán tập hợp chi phí sản xuất cho từng đối tượng tính giá thành sản phẩm trong doanh nghiệp. " * 6
    assert unaccented_ratio(text, LEXICON)[0] < 0.02


def test_gach_noi_chu_hoa_don_khong_phai_mat_u_o():
    """"C- section", "N- TiO2" (tiếng Anh / hoá học) không bị tính; "ng-ời", "đ-ợc", "c- dân", "th- viện" vẫn bị tính."""
    assert hyphen_lost("Women with C- section delivery and N- TiO2 samples")[2] == 0
    assert hyphen_lost("ng-ời Mạ đ-ợc c- dân th- viện h-ớng dẫn")[2] == 5


def test_che_so_dien_thoai_khong_dung_vao_so_lieu_bang():
    """Số điện thoại thật bị che; trục biểu đồ, ma trận tương quan, số thập phân dài thì giữ nguyên."""
    line, n = mask_phones("Tel: 0983 226 880, ĐT 024.3825.1234, di động 0912345678, Tel. 084 38251234")
    assert n == 4 and "<PHONE>" in line and "983" not in line and "3825" not in line
    for table in ("0.0 0.2 0.4 0.6 0.8 1.0 0.5 0.34", "var10 0.2629 0.3785 0.5425 0.1234", "-0.0000000000059739", "0.15 0.23 0.42 5.89 5.903 0.38 0.41"):
        assert mask_phones(table) == (table, 0)
    assert find_phones("Mobile: 0903 123 456 và bảng 0.2 0.4 0.6 0.8 1.0 1.2 1.4") == ["0903 123 456"]


def test_mau_audit_trich_quanh_cho_loi():
    """excerpt_of lấy ngữ cảnh quanh chỗ lỗi đã định vị (tiêu đề tạp chí, số điện thoại), còn lỗi không định vị được thì lấy đầu đoạn."""
    from vi_corpus.vjol.audit import excerpt_of
    text = "Mở đầu " * 60 + "Tạp chí Kinh tế số 12 chen giữa " + "Nội dung " * 60
    assert "Tạp chí Kinh tế số 12" in excerpt_of(text, "journal_header") and excerpt_of(text, "journal_header").startswith("…")
    assert excerpt_of(text, "khong_biet") == text[:240]
    assert "0903 123 456" in excerpt_of("x " * 200 + "Mobile: 0903 123 456 " + "y " * 200, "phone")


def test_producer_ocr_cu_thi_ca_bai_di_ocr(tmp_path):
    """J-29 (D2): PDF có producer "ABBYY FineReader 11" + creator "ScanGate" thì mọi trang có chữ đi OCR (lý do old_ocr_producer) dù chữ trông bình thường; producer khác thì không."""
    import pymupdf
    from tests.test_garbled import mixed_pdf, ocr_engine
    from tests.test_pipeline import vi_text
    from vi_corpus.common import garbled as g
    from vi_corpus.common.pdf_text import extract_pdf
    from vi_corpus.vjol.extract import OCR_ALL_PRODUCERS

    def build(name: str, producer: str, creator: str):
        """Tạo PDF 3 trang chữ tốt với metadata cho trước."""
        pdf = tmp_path / name
        mixed_pdf(pdf, [vi_text(4, k) for k in range(3)])
        with pymupdf.open(pdf) as d:
            d.set_metadata({"producer": producer, "creator": creator})
            d.saveIncr()
        return pdf

    seen: list = []
    old = extract_pdf(build("old.pdf", "ABBYY FineReader 11", "ScanGate V2.53"), ocr_engine(seen), detect_garbled=True, ocr_all_producers=OCR_ALL_PRODUCERS)
    assert [p["reason"] for p in old] == [g.REASON_PRODUCER] * 3 and len(seen) == 3
    seen2: list = []
    other = extract_pdf(build("new.pdf", "ABBYY FineReader 14", "ABBYY FineReader 14"), ocr_engine(seen2), detect_garbled=True, ocr_all_producers=OCR_ALL_PRODUCERS)
    assert all("reason" not in p for p in other) and not seen2
