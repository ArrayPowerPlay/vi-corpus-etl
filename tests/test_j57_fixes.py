"""
Kiểm J-57: gỡ chỗ "ph-ơng / l-ơng" mơ hồ của hyphen_fix bằng bằng chứng dấu thanh cả bài và bộ chọn ngữ cảnh.
"""

from vi_corpus.vjol.hyphen_fix import Ambiguous, has_tone_evidence, restore_u_in_pages, restore_u_in_text

VOCAB = frozenset("phương phường phướng phượng chương chường hương hường hướng hưởng trường được người nước ương mương mường".split())


def test_mo_ho_giu_nguyen_khi_khong_co_bo_chon():
    """Không có bằng chứng và không có bộ chọn: giữ nguyên như J-42, đếm vào ambiguous."""
    stats: dict = {}
    assert restore_u_in_text("phương pháp ph-ơng pháp", VOCAB, stats=stats) == ("phương pháp ph-ơng pháp", 0)
    assert stats["ambiguous"] == 1


def test_bang_chung_dau_thanh_cho_phep_sua_thanh_ngang():
    """Bài còn 'tr-ờng', 'đ-ợc', 'ng-ời' thì 'ơ' trần là thanh ngang: 'ph-ơng' -> 'phương'."""
    page = "tr-ờng đ-ợc ng-ời h-ớng và ph-ơng pháp ch-ơng 1"
    assert has_tone_evidence([page])
    out, n = restore_u_in_pages([page], VOCAB)
    assert out == ["trường được người hướng và phương pháp chương 1"] and n == 6


def test_bai_mat_dau_thanh_khong_co_bang_chung():
    """Bài chỉ có 'ơ' trần (mất dấu thanh) thì không có bằng chứng, kể cả khi lẻ tẻ một chỗ có dấu."""
    page = "ph-ơng pháp l-ơng ch-ơng h-ơng th-ơng tr-ờng"
    assert not has_tone_evidence([page])


def test_bo_chon_gỡ_mo_ho_va_giu_ca_chua_chac():
    """Bộ chọn trả cách đọc thì sửa; trả None thì giữ chữ hỏng; ngữ cảnh trái / phải được truyền đúng."""
    seen = []

    def chooser(left, cands, right):
        """Chọn thanh ngang nếu bên phải là 'pháp', ngược lại không chắc."""
        seen.append((left, cands, right))
        return cands[0] if right.startswith(" pháp") else None

    stats: dict = {}
    text = "Cách ph-ơng pháp này; còn h-ơng đó"
    out, n = restore_u_in_text(text, VOCAB, stats=stats, chooser=chooser)
    assert out == "Cách phương pháp này; còn h-ơng đó" and n == 1
    assert stats == {"chosen": 1, "ambiguous": 1}
    assert seen[0][0].endswith("Cách ") and seen[0][1][0] == "phương" and set(seen[0][1]) == {"phương", "phường", "phướng", "phượng"}


def test_ambiguous_giu_hoa_va_tien_to():
    """Các cách đọc giữ kiểu chữ hoa; dạng ' - ơng' giữ khoảng trắng đứng trước."""
    assert Ambiguous("Phương").alternatives(VOCAB)[0] == "Phương" and "Phường" in Ambiguous("Phương").alternatives(VOCAB)
    out, _ = restore_u_in_text("trung - ơng", VOCAB | {"ương"}, chooser=lambda l, c, r: c[0])
    assert out == "trung ương"
