"""Kiểm tra nhận diện font TCVN3 theo tên và bảng đổi mã trong vi_corpus.common.pdf_text."""

from vi_corpus.common.pdf_text import _TCVN3_FONT, _TCVN3_UPPER_FONT, tcvn3_to_unicode


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


def test_tcvn3_to_unicode_real_cover():
    """Chuỗi thật từ bìa "Giao Trinh Mat Ma Hoc" (font VnArialH / VnBahamasBHBold / VnArial)."""
    assert tcvn3_to_unicode("NGUYÔN THÞ THU Hμ").upper() == "NGUYỄN THỊ THU HÀ"
    assert tcvn3_to_unicode("NHμ XUÊT B¶N").upper() == "NHÀ XUẤT BẢN"
    assert tcvn3_to_unicode("Phè Ngôy Nh− Kon Tum") == "Phố Ngụy Như Kon Tum"
