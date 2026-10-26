"""
Kiểm các sửa J-37 cho VJOL, rút ra từ kết quả chạy lại S-19 (vjol_10k_v7) và 400 nhãn chuẩn: mở rộng khôi phục "ư" (A1-A4), ghép lại "ư" bị tách rời (B), cắt tiêu đề tạp chí ở đầu bài (C),
số đo / bộ dò mới của audit (D), chữ "ƣ" của phông Việt cũ (E1). Câu ví dụ lấy từ các dòng nhãn thật nêu trong docs/DECISION_LOG.md J-37. Không cần mạng, GPU hay mô hình.
"""

import json

from tests.test_pipeline import vi_text
from tests.test_j25_fixes import _quality_row
from vi_corpus.pipeline import quality as Q
from vi_corpus.pipeline.normalize import normalize_text
from vi_corpus.vjol import audit as A
from vi_corpus.vjol.clean import _strip_front_line, clean_paper_pages, strip_front_headers
from vi_corpus.vjol.hyphen_fix import restore_u_in_pages, restore_u_in_text

# Tập âm tiết giả đủ cho các câu ví dụ.
VOCAB = frozenset(("vương đường mường hương ước ương trung dưới cương tương lưu hưng ưu tiên điểm đãi tối như nhưng người trước xưa chưa thưa đưa nước lưới cưới hướng hưởng được ảnh "
                   "tưởng thư viện cư dân liệu sau ở ngưỡng phương và các là của một những quy ước mưu sinh kế").split())


# ---------------------------------------------------------------------------------------------------------------- A: khôi phục "ư" thêm dạng


def test_chu_dau_viet_hoa_tach_khoang_trang():
    """A1: "V- ơng", "Đ- ờng", "M- ỜNG", "L- u", "H- ng" thành Vương, Đường, MƯỜNG, Lưu, Hưng; "C- section", "N- TiO2", "Xa N- ở" giữ nguyên."""
    cases = {"Vương": "V- ơng", "Đường": "Đ- ờng", "MƯỜNG": "M- ỜNG", "Lưu": "L- u", "Hưng": "H- ng", "Dưới": "D- ới"}
    for want, src in cases.items():
        assert restore_u_in_text(f"ông {src} đã", VOCAB) == (f"ông {want} đã", 1), src
    for keep in ("Phẫu thuật C- section", "mẫu N- TiO2 được", "đến Xa N- ở", "IFN- γ và TNF- α", "D- ong"):
        assert restore_u_in_text(keep, VOCAB | {"section", "ong", "nư"}) == (keep, 0), keep


def test_dau_gach_noi_dung_mot_minh_la_am_tiet_u():
    """A2: "trung - ơng" -> "trung ương", "hương - ớc" -> "hương ước", "quy - ước" -> "quy ước"; dấu ngắt thật và công thức giữ nguyên."""
    for src, want in (("Ban chấp hành trung - ơng khóa", "Ban chấp hành trung ương khóa"), ("theo hương - ớc làng", "theo hương ước làng"), ("xây dựng quy - ước chung", "xây dựng quy ước chung"),
                      ("thủ tục -ơng", "thủ tục ương")):
        assert restore_u_in_text(src, VOCAB) == (want, 1), src
    # J-42 P6: "H- ơng" có cả "Hương" lẫn "Hưởng" trong tập âm tiết nên không sửa; " - ớc" vẫn sửa vì "ước" có dấu thanh rõ
    assert restore_u_in_text("H- ơng - ớc làng", VOCAB) == ("H- ơng ước làng", 1)
    for keep in ("td σ - ứng suất tương đương", "Hà Nội - ở đây", "Trung - ơng-quận-huyện-h-ơng"):
        assert restore_u_in_text(keep, VOCAB)[1] == 0, keep


def test_chu_u_roi_sau_dau_gach_noi():
    """A3: "tối - u" -> "tối ưu", "các - u điểm" -> "các ưu điểm"; công thức "u^{(k)} - u^{(k-1)}" và "x - u_i" giữ nguyên."""
    assert restore_u_in_text("giải pháp tối - u cho mâu thuẫn", VOCAB) == ("giải pháp tối ưu cho mâu thuẫn", 1)
    assert restore_u_in_text("Bên cạnh các - u điểm, cuốn sách", VOCAB)[0] == "Bên cạnh các ưu điểm, cuốn sách"
    for keep in ("(u^{(k+1)} - u^{(k)}) d\\Omega", "ta có x - u_i = 0", "\\varphi - u(x)"):
        assert restore_u_in_text(keep, VOCAB) == (keep, 0), keep


def test_nh_dung_truoc_dau_cau_va_nh_luon_la_nhu():
    """A4: "nh- “bức" -> "như “bức"; "nh- người" sửa luôn dù cả bài chỉ có một chỗ (không cần ngưỡng); "nh- ng" vẫn ra "nhưng"."""
    assert restore_u_in_text("Bắc nh- “bức tranh”", VOCAB) == ("Bắc như “bức tranh”", 1)
    one = ["Giống nh- người Việt khác."]
    assert restore_u_in_pages(one, VOCAB) == (["Giống như người Việt khác."], 1)
    assert restore_u_in_text("cũng nh- ng đã", VOCAB)[0] == "cũng nhưng đã"


# ---------------------------------------------------------------------------------------------------------------- B: "ư" bị tách rời


def test_u_tach_roi_ghep_lai():
    """B: "xư a", "lư u ý", "chư a", "đư a", "hư ớng", "nư ớc", "đư ợc", "ngư ời", "nư óc", "lư ối", "cư ói" (ói), "hư ưởng" được ghép lại; không cần ngưỡng cả bài."""
    cases = {"xư a": "xưa", "đáng lư u ý": "đáng lưu ý", "vẫn chư a đạt": "vẫn chưa đạt", "để đư a sợi": "để đưa sợi", "xu hư ớng": "xu hướng", "Nhà nư ớc": "Nhà nước",
             "hư ưởng": "hưởng", "con ngư ời": "con người", "nư óc ta": "nước ta", "tóc, tai, lư ối": "tóc, tai, lưới", "thách cư ới": "thách cưới"}
    for src, want in cases.items():
        assert restore_u_in_text(src, VOCAB | {"ưởng"}, spaced=False)[0] == want, src
    assert restore_u_in_pages(["Cha ông ta xư a."], VOCAB) == (["Cha ông ta xưa."], 1)


def test_u_tach_roi_khong_ghep_cau_that():
    """B: câu thật giữ nguyên ("như sau", "tư liệu", "cư dân", "ung thư ở", "đầu tư ứng", "như ong", "sư ông", "ngư ông", "ung thư ống", "Như Lai") và chữ ơ bị mất ("phư ng", "tư ng") cũng không ghép sai."""
    for keep in ("như sau", "tư liệu", "cư dân", "ung thư ở phụ nữ", "đầu tư ứng trước", "chăm chỉ như ong", "sư ông", "ngư ông", "ung thư ống tiêu hóa", "Phật Như Lai", "phư ng pháp", "đối tư ng", "tư ước"):
        assert restore_u_in_text(keep, VOCAB | {"ong", "ông", "ống", "ng", "ứng"}) == (keep, 0), keep


# ---------------------------------------------------------------------------------------------------------------- C: tiêu đề tạp chí ở đầu bài


def test_cat_tieu_de_tap_chi_o_dau_dong():
    """C: các kiểu tiêu đề ở đầu bài lấy từ nhãn chuẩn bị cắt, phần thân còn nguyên."""
    cases = {
        "Tạp chí Dân tộc học số 5 – 2004 ĐẶC ĐIỂM HỆ THỐNG": "ĐẶC ĐIỂM HỆ THỐNG",
        "Tạp chí KHOA HỌC ĐÀO TẠO VÀ HUẤN LUYỆN THỂ THAO - SỐ 4/2019 ĐẶC ĐIỂM NGƯỜI DÂN": "ĐẶC ĐIỂM NGƯỜI DÂN",
        "TẠP CHÍ THÔNG TIN VÀ TƯ LIỆU ISSN 1859-2929 Số 1/2005 No. 1/2005 Tin hoạt động": "Tin hoạt động",
        "Tạp chí Dân tộc học số 6 – 2005 47 Cùng với quy mô": "Cùng với quy mô",
        "Tạp chí Khoa học Ngoại ngữ Số 5 (tháng 12 năm 2005) Hoạt động": "Hoạt động",
        "TẠP CHÍ KHOA HỌC ĐHQGHN, NGOẠI NGỮ, T. XIX, SỐ 1, 2003 TÍNH QUA LỐI": "TÍNH QUA LỐI",
        "Tạp chí Công nghệ ngân hàng | Tháng 01&02.2017 | Số 130&131 MỐI QUAN HỆ": "MỐI QUAN HỆ",
        "Tạp chí Khoa học, Trường Đại học Sư phạm, Đại học Huế ISSN 1859-1612, Số 01(49)/2019: tr. 22-31 Khoa Ngữ": "Khoa Ngữ",
        "Tạp chí Kinh tế - Luật và Ngân hàng Năm thứ 27(12)- Tháng 11. 2025- Số 284 © Học viện Ngân hàng ISSN 3030 - 4199 Tác động của": "Tác động của",
        "BÀI BÁO KHOA HỌC Tạp chí KHOA HỌC ĐÀO TẠO VÀ HUẤN LUYỆN THỂ THAO - SỐ 1/2018 NGHIÊN CỨU ỨNG DỤNG": "NGHIÊN CỨU ỨNG DỤNG",
        ". Số 3 - 2005 45 về sự biến đổi": "về sự biến đổi",
        "22 . Số 9 - 2010 GÓP THÊM CÁC BẢN DỊCH": "GÓP THÊM CÁC BẢN DỊCH",
        "Nghiên cứu Tôn giáo. Số 6 - 2001 35 ĐỜI SỐNG ĐẠO": "ĐỜI SỐNG ĐẠO",
        "SỐ 12 (206)-2012 NGÔN NGỮ & ĐỜI SỐNG 27 NGÔN NGỮ VỚI VĂN CHƯƠNG": "NGÔN NGỮ VỚI VĂN CHƯƠNG",
        "© Học viện Ngân hàng ISSN 1859 - 011X Chính sách giám sát": "Chính sách giám sát",
        "Vietnam J. Agri. Sci. 2016, Vol. 14, No. 6: 843-851 Tạp chí KH Nông nghiệp Việt Nam 2016, tập 14, số 6: 843-851 www.vnua.edu.vn KẾT QUẢ ĐÁNH GIÁ": "KẾT QUẢ ĐÁNH GIÁ",
        "http://tapchikttv.vn/ Bài báo khoa học Tác động của hạ thấp": "Tác động của hạ thấp",
        "Tạp chí Khoa học DHQGHN, Câu trung Pháp": "Câu trung Pháp",
    }
    for src, want in cases.items():
        assert _strip_front_line(src, True) == (want, 1) or _strip_front_line(src, True)[0] == want, (src, _strip_front_line(src, True))


def test_khong_cat_cau_trich_dan_va_cau_van():
    """C: câu trích dẫn tên tạp chí có dấu phẩy trước "số" và câu văn thường giữ nguyên, kể cả ở đầu bài."""
    for keep in ("Tạp chí Nghiên cứu Tôn giáo, số 4/2000, tr. 5", "Tạp chí Nghiên cứu Trung Quốc, số 10 (110), tháng 10/2010, tr.77 – 85.",
                 "Tạp chí Cẩm Thành, Sở Văn hoá - Thông tin tỉnh Quảng Ngãi, số 32, tháng 8/2002, tr. 37.", "Tạp chí là một loại ấn phẩm định kỳ được xuất bản theo số.",
                 "Số 3 người tham gia khảo sát", "Số liệu năm 2005 cho thấy"):
        assert _strip_front_line(keep, True) == (keep, 0), keep


def test_cat_tieu_de_chi_o_dau_trang_va_ghi_so_dem():
    """C: trang đầu xét 8 dòng đầu, trang sau chỉ 2 dòng đầu và không nhận dạng có dấu phẩy; dòng chỉ gồm tiêu đề thì bị bỏ; clean_paper_pages ghi front_headers."""
    pages = ["Tạp chí Dân tộc học số 3 - 2004\n" + vi_text(30, 1), vi_text(30, 2) + "\nTạp chí Nghiên cứu Tôn giáo, số 4/2000, tr. 5 là một bài báo.", "Tạp chí Dân tộc học, số 5 – 2004 tr. 12\n" + vi_text(30, 3)]
    out, n = strip_front_headers(pages)
    assert n == 1 and out[0].startswith(vi_text(30, 1)[:20]) and "Tạp chí Dân tộc học số 3" not in out[0]
    assert out[1] == pages[1] and out[2] == pages[2]
    deep = ["\n".join(f"dòng {i}" for i in range(9)) + "\nTạp chí Dân tộc học số 3 - 2004"]
    assert strip_front_headers(deep) == (deep, 0)
    cleaned, st = clean_paper_pages(["Nghiên cứu Tôn giáo. Số 6 - 2001 35 ĐỜI SỐNG ĐẠO CỦA GIÁO DÂN\n" + vi_text(40, 1), vi_text(40, 2), vi_text(40, 3)])
    assert st["front_headers"] == 1 and cleaned[0].startswith("ĐỜI SỐNG ĐẠO")


# ---------------------------------------------------------------------------------------------------------------- D: audit, bộ dò, ngưỡng


def test_audit_do_header_start_va_u_split():
    """D2: header_start đếm tiêu đề trong 200 ký tự đầu (không đếm ở giữa đoạn), u_split đếm "ư" tách rời còn sót; HIT_LOCATORS định vị được."""
    rows = [_quality_row(i, vi_text(3, i)) for i in range(20)]
    rows[0]["text"] = "Tạp chí Dân tộc học số 4 – 2005 " + rows[0]["text"]
    rows[1]["text"] = rows[1]["text"] + " " * 300 + " Tạp chí Dân tộc học số 4 – 2005"
    rows[2]["text"] += " người xư a trát đất"
    rows[3]["text"] += " ung thư ở phụ nữ"
    res = A.audit_rows(rows, min_group=5)
    assert res["error_counts"]["header_start"] == 1 and res["error_counts"]["u_split"] == 1
    assert A.excerpt_of(rows[2]["text"], "u_split").find("xư a") >= 0


def test_bo_do_running_header_va_nhan_moi():
    """D4: evaluate_labels đo bộ dò running_header (= header_start) và nhận nhãn running_header / hyphen_font / other."""
    items = []
    for i in range(40):
        header = i % 4 == 0
        text = ("Nghiên cứu Tôn giáo. Số 6 - 2001 35 " if header else "") + vi_text(3, i)
        items.append({"doc_id": f"d{i}", "split": "tune" if i % 3 else "holdout", "label": "running_header" if header else "clean", "quality_metrics": json.dumps({"hyphen_lost": 0}), "text": text})
    res = A.evaluate_labels(items)
    assert res["by_label"] == {"running_header": 10, "clean": 30}
    assert res["detectors"]["running_header"]["tune"]["precision"] == 1.0 and res["detectors"]["running_header"]["holdout"]["recall"] == 1.0
    assert res["clean_false_flags"]["tune"]["running_header"] == 0 and "hyphen_font" in res["detectors"]


def test_nguong_hyphen_cung_la_3():
    """D1: đoạn có 3 chỗ "-" thay ư là lỗi cứng lost_letters (band D); 2 chỗ thì không."""
    assert Q.HYPHEN_HARD_COUNT == 3
    assert A.DETECTORS["hyphen_font"]({"hyphen_lost": 3}, "") and not A.DETECTORS["hyphen_font"]({"hyphen_lost": 2}, "")


# ---------------------------------------------------------------------------------------------------------------- E1: chữ ƣ


def test_normalize_doi_chu_oi_thanh_u():
    """E1: "ƣ" / "Ƣ" (U+01A3 / U+01A2) của phông Việt cũ thành "ư" / "Ư": "Hƣng Yên", "đƣợc", "NGƢỜI"; chữ khác giữ nguyên."""
    assert normalize_text("Hƣng Yên đƣợc ngƣời") == "Hưng Yên được người"
    assert normalize_text("NGƢỜI Ƣ") == "NGƯỜI Ư"
    assert normalize_text("Việt Nam") == "Việt Nam"
