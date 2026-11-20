"""Kiểm tra nhận diện font TCVN3 theo tên và bảng đổi mã trong vi_corpus.common.pdf_text."""

from vi_corpus.common.pdf_text import (_TCVN3_FONT, _TCVN3_UPPER_FONT, is_tcvn3, is_tcvn3_document, page_has_tcvn3_evidence, page_text,
                                       should_convert_tcvn3_span, tcvn3_to_unicode)


def test_tcvn3_font_names():
    """Tên font TCVN3 lấy từ bìa giáo trình thật; VNI và font Unicode "UVN..." không được khớp."""
    for name in (".VnTime", ".VnTime,Bold", "VnArial", "VnArialH", "VnSouthernH", "VnBahamasBHBold"):
        assert _TCVN3_FONT.match(name), name
    for name in ("VNI-Times", "UVNChinhLuan", "TimesNewRoman", "ArialMT"):
        assert not _TCVN3_FONT.match(name), name
    for name in (".VnTimeH", "VnArialH", "VnSouthernH", "VnBahamasBHBold", "VnArialH,Bold"):
        assert _TCVN3_UPPER_FONT.search(name), name
    for name in (".VnTime", "VnArial", "VnArialBold", "VnAristoteMedium"):
        assert not _TCVN3_UPPER_FONT.search(name), name


def test_only_convert_vn_font_spans_when_page_looks_like_tcvn3():
    """Font Unicode VnBitstreamCharter (trang không có bằng chứng TCVN3) không được áp bảng chuyển mã TCVN3."""
    unicode_page = [("VnBitstreamCharter", "Mã hoá khoá đối xứng")]
    tcvn3_page = [("VnArial", "M«men tõ: §iÖn tö")]
    assert not page_has_tcvn3_evidence(unicode_page)
    assert page_has_tcvn3_evidence(tcvn3_page)
    assert not should_convert_tcvn3_span("VnBitstreamCharter", False)
    assert should_convert_tcvn3_span("VnArial", True)
    assert not should_convert_tcvn3_span("TimesNewRoman", True)  # font không mang tên TCVN3 thì không đổi dù trang có bằng chứng
    assert should_convert_tcvn3_span("TimesNewRoman", False, tcvn3_all=True)


def test_page_evidence_ignores_marker_text_in_non_vn_fonts():
    """Ký tự giống TCVN3 trong font không mang tên TCVN3 (vd VNI) không tạo bằng chứng cho trang."""
    assert not page_has_tcvn3_evidence([("VNI-Times", "§iÖn tö M«men tõ")])


class _FakePage:
    """Trang giả trả về rawdict gồm các span (font, text) trên các dòng ngang cách nhau, đủ cho page_text."""

    def __init__(self, spans):
        """spans: danh sách (tên font, text); mỗi span một dòng."""
        self.spans = spans

    def get_text(self, mode):
        """Trả về rawdict giả: mỗi ký tự có bbox liền kề nhau (không sinh dấu cách thừa)."""
        lines = []
        for font, text in self.spans:
            chars = [{"c": c, "bbox": (i * 5.0, 0.0, i * 5.0 + 5.0, 10.0)} for i, c in enumerate(text)]
            lines.append({"dir": (1, 0), "spans": [{"font": font, "size": 10.0, "chars": chars}]})
        return {"blocks": [{"lines": lines}]}


def test_page_text_converts_sparse_and_ascii_spans_on_tcvn3_page():
    """Một span thân bài đạt ngưỡng kéo theo cả span ít dấu và tiêu đề chữ hoa toàn ASCII của trang (không bị bỏ sót / mất chữ hoa)."""
    sparse = "Cac dinh nghÜa cua mot khai niem co ban trong giao trinh dien tu hoc"
    assert not is_tcvn3(sparse) and not is_tcvn3("tin hoc")  # từng span một không đạt ngưỡng (hành vi cũ ở mức span sẽ bỏ sót)
    page = _FakePage([("ABCDEF+.VnTimeH", "tin hoc"),                        # tiêu đề: mã ASCII, hiện chữ hoa
                      ("ABCDEF+.VnTime", sparse),                              # ít dấu: tỉ lệ ký tự đặc trưng thấp hơn ngưỡng
                      ("ABCDEF+.VnTime", "M«men tõ: §iÖn tö")])               # thân bài: vượt ngưỡng
    assert page_text(page).splitlines() == ["TIN HOC", "Cac dinh nghĩa cua mot khai niem co ban trong giao trinh dien tu hoc", "Mômen từ: Điện tử"]


def test_page_text_keeps_unicode_vn_font_page_unchanged():
    """Trang chỉ có font Unicode tên Vn* (VnBitstreamCharter) giữ nguyên, không bị đổi thành mojibake."""
    page = _FakePage([("VnBitstreamCharter", "Mã hoá khoá đối xứng")])
    assert page_text(page) == "Mã hoá khoá đối xứng"


def test_vni_fonts_do_not_trigger_whole_document_tcvn3_conversion():
    """Các ký tự VNI không được làm bật chuyển mã TCVN3 cho toàn PDF."""
    vni_text = "Söï thay ñoài cuûa UL vaø UC theo ñieän dung C " * 3
    assert not is_tcvn3_document(vni_text, ["VNI-Times", "VNI-Centur"])
    assert is_tcvn3_document("§iÖn tö " * 10)


def test_tcvn3_to_unicode_real_cover():
    """Chuỗi thật từ bìa "Giao Trinh Mat Ma Hoc" (font VnArialH / VnBahamasBHBold / VnArial)."""
    assert tcvn3_to_unicode("NGUYÔN THÞ THU Hμ").upper() == "NGUYỄN THỊ THU HÀ"
    assert tcvn3_to_unicode("NHμ XUÊT B¶N").upper() == "NHÀ XUẤT BẢN"
    assert tcvn3_to_unicode("Phè Ngôy Nh− Kon Tum") == "Phố Ngụy Như Kon Tum"
