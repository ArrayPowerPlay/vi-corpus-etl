"""
Kiểm các việc đã chốt ở J-40 / J-42 cho VJOL (code ở J-43): mẫu chunk ổn định (1a), so ghép đôi (1b), bộ dò và số đo mới của audit (1c), nhãn chuẩn chia nhỏ (1d), sửa "ư" chỉ khi xác định + repairs (5),
cắt mức dòng (6), OCR lại cả bài (3), phép thử nhóm font (4), kế hoạch OCR lại (2), mẫu thi cuối (7). Câu ví dụ lấy từ docs/DECISION_LOG.md J-41 / J-42. Không cần mạng, GPU hay mô hình.
"""

import json

import pymupdf

from tests.test_garbled import mixed_pdf
from tests.test_j25_fixes import _quality_row
from tests.test_pipeline import WORDS, make_cfg, vi_text
from vi_corpus.common import defects as Df
from vi_corpus.common import garbled as g
from vi_corpus.common.pdf_text import extract_pdf
from vi_corpus.pipeline import quality as Q
from vi_corpus.pipeline.chunk import chunk_rank, chunk_rows, sample_chunks
from vi_corpus.vjol import audit as A
from vi_corpus.vjol import font_canary as FC
from vi_corpus.vjol import plan_reocr as PR
from vi_corpus.vjol import reocr as R
from vi_corpus.vjol.clean import clean_paper_pages, find_journal_headers, is_junk_line, strip_junk_lines, strip_mid_headers
from vi_corpus.vjol.hyphen_fix import restore_u_in_pages, restore_u_in_text
from vi_corpus.vjol.paired import diff_excerpt, edit_kind, format_paired, paired_compare
from vi_corpus.vjol.records import repairs_of

VOCAB = frozenset("tương hỗ tóm tắt nghiên nhân khánh cứu bệnh trường người được nhưng như ương trung tối ưu mương mường vương vưởng vượng hương hưởng hường".split())


# ---------------------------------------------------------------------------------------------------------------- (1a) mẫu chunk ổn định


def _vjol_docs(n_docs: int, edit: int | None = None) -> list[dict]:
    """n_docs bài vjol giả, mỗi bài nhiều đoạn; bài thứ `edit` (nếu có) bị sửa chữ làm đổi số đoạn."""
    return [{"doc_id": f"vjol:{i}:0", "source_key": "vjol", "text": vi_text(6 if i == edit else 40, i + (100 if i == edit else 0))} for i in range(n_docs)]


def test_mau_chunk_theo_bam_on_dinh_khi_sua_chu_mot_bai():
    """J-40 a: sửa chữ trong một bài (đổi số đoạn của bài đó) chỉ làm đổi các đoạn được chọn của bài khác ở mức biên (không xáo cả mẫu như random.sample); mẫu n nhỏ nằm trong mẫu n lớn."""
    cfg = make_cfg(mix={"vjol": 40}, chunk_target_tokens=100, chunk_max_tokens=150, chunk_min_tokens=30)
    before, _ = chunk_rows(_vjol_docs(12), cfg, WORDS)
    after, _ = chunk_rows(_vjol_docs(12, edit=3), cfg, WORDS)
    assert len(before) == len(after) == 40

    def others(rows: list[dict]) -> set[str]:
        """doc_id của các đoạn được chọn thuộc các bài khác bài 3."""
        return {r["doc_id"] for r in rows if r["parent_doc_id"] != "vjol:3:0"}

    moved = len(others(before) ^ others(after))
    delta = abs(sum(r["parent_doc_id"] == "vjol:3:0" for r in before) - sum(r["parent_doc_id"] == "vjol:3:0" for r in after))
    assert moved <= 2 * max(delta, 1) + 2 and moved < 20, (moved, delta)
    pieces = [{"doc_id": f"d{i}#c{j}", "parent_doc_id": f"d{i}", "chunk_index": j} for i in range(5) for j in range(6)]
    small, big = sample_chunks(pieces, 8, 0, "vjol"), sample_chunks(pieces, 16, 0, "vjol")
    assert {p["doc_id"] for p in small} <= {p["doc_id"] for p in big}
    assert chunk_rank(0, "vjol", "d1", 2) == chunk_rank(0, "vjol", "d1", 2) != chunk_rank(1, "vjol", "d1", 2)


# ---------------------------------------------------------------------------------------------------------------- (1b) so ghép đôi


def _row(i: int, text: str, band: str = "A", codes=()) -> dict:
    """Bản ghi giả của so ghép đôi."""
    return {"doc_id": f"vjol:{i}:0#c0", "text": text, "quality_band": band, "reason_codes": list(codes)}


def test_so_ghep_doi_chi_xet_doan_chung():
    """J-40 b: chỉ đoạn có ở cả hai lần chạy; đếm đoạn đổi chữ, giữ -> loại (kèm mã mới), loại -> giữ (kèm mã cũ), mã thêm / mất, và có mẫu đổi chữ để đọc."""
    old = [_row(0, "a b c d e f g"), _row(1, "x y z", "A"), _row(2, "u v w", "D", ["lost_letters"]), _row(3, "chỉ cũ"), _row(4, "k l m", "D", ["odd_words"])]
    new = [_row(0, "a b c X e f g"), _row(1, "x y z", "D", ["garbled_font"]), _row(2, "u v w", "A"), _row(5, "chỉ mới"), _row(4, "k l m", "D", ["odd_words", "lost_letters"])]
    res = paired_compare(old, new, samples=5)
    assert (res["common"], res["only_old"], res["only_new"], res["text_changed"]) == (4, 1, 1, 1)
    assert (res["kept_to_dropped"], res["dropped_to_kept"], res["dropped_both"], res["kept_both"]) == (1, 1, 1, 1)
    assert res["codes_gained"] == {"garbled_font": 1, "lost_letters": 1} and res["codes_lost"] == {"lost_letters": 1}
    assert res["transitions"] == {"kept->dropped": {"garbled_font": 1}, "dropped->kept": {"lost_letters": 1}}
    assert [s["doc_id"] for s in res["samples"]] == ["vjol:0:0#c0"] and "[d -> X]" in res["samples"][0]["diff"]
    assert "chung 4" in format_paired(res) and diff_excerpt("a b", "a b") == ""


def test_so_ghep_doi_tach_xe_dich_bien_khoi_doi_chu():
    """J-44: đoạn chỉ thừa / thiếu chữ ở đầu hoặc cuối (biên đoạn xê dịch khi cắt tiêu đề phía trước) là "edge", không tính vào đổi chữ ở thân và không vào mẫu; đoạn dài được rút gọn."""
    assert edit_kind("a b c", "a b c") == "same"
    assert edit_kind("a b c d e f", "a b c") == "edge" and edit_kind("H a b c", "a b c") == "edge" and edit_kind("a b c", "a b c d e") == "edge"
    assert edit_kind("a b c d", "a X c d") == "inner" and edit_kind("H a b c d e", "a X c") == "inner"
    old = [_row(0, "a b c d e f g h"), _row(1, "p q r s t u v w")]
    new = [_row(0, "a b c d"), _row(1, "p q Z s t u v w")]
    res = paired_compare(old, new, samples=5)
    assert (res["text_changed"], res["text_changed_inner"], res["text_changed_edge"]) == (2, 1, 1)
    assert [s["doc_id"] for s in res["samples"]] == ["vjol:1:0#c0"]
    assert "biên" in diff_excerpt("a b c d e f g h", "a b c d") and "…" in diff_excerpt(" ".join(f"w{i}" for i in range(40)) + " end", "end")


# ---------------------------------------------------------------------------------------------------------------- (1c) bộ dò mới


def test_bo_do_chu_roi_rac_dinh_lech_dau():
    """gibberish, dropped_letters, joined_words, tone_shift đo đúng ca ví dụ của J-41 / J-42 và không báo nhầm đơn vị, ký hiệu toán, chữ đúng."""
    assert Df.gibberish_ratio("/, N9 = < 0 e H ; : " * 10)[0] > 0.9 and Df.gibberish_ratio("Nghiên cứu này trình bày kết quả khảo sát " * 5)[0] < 0.05
    assert Df.dropped_letters_runs("cho thấy kh n th và nh ng điều khác") == 2
    assert Df.dropped_letters_runs("liều 5 mm cm kg mg và các biến x y z") == 0
    # J-44: chuỗi sát ký hiệu toán (công thức dàn phẳng) không phải chữ rơi; chữ rơi thật trong văn xuôi vẫn đếm
    assert Df.dropped_letters_runs("nhiệt độ trung bình tháng t tb = 20,5 – 25,7 độ") == 0
    assert Df.dropped_letters_runs("liên tục theo độ sâu: h dz z w x u ∫ − η và (6)") == 0
    assert Df.dropped_letters_runs("có nh ng s thay đổi về tâm sinh l , s thay đổi này k o theo nh ng") >= 2
    assert Df.joined_words_hits("mối quan hệ tươnghỗ và Tómtắt", VOCAB) == 2 and Df.joined_words_hits("tương hỗ tóm tắt nghiên cứu", VOCAB) == 0
    assert Df.tone_shift_hits("NGHIấN cứu bệnh nhõn Khỏnh", VOCAB) == 3 and Df.tone_shift_hits("nghiên cứu bệnh nhân", VOCAB) == 0
    assert Df.first_dropped_span("cho thấy kh n th ok") == (9, 16) and Df.first_gibberish_span("abc def = ghi")[0] == 8


def test_hoc_bang_doi_lech_dau_tu_du_lieu():
    """learn_tone_shift_map học đúng cặp ỏ -> á từ dữ liệu có lỗi lặp lại, bỏ cặp rải rác ít lần; load_tone_shift_map đọc lại bảng đã ghi."""
    vocab = frozenset("phát triển thái nhân khánh".split())
    texts = ["phỏt triển nhõn thỏi khỏnh " * 12, "phát triển xyz abc"]
    res = Df.learn_tone_shift_map(texts, vocab, min_count=10, min_share=0.2)
    assert res["map"].get("ỏ") == "á" and res["map"].get("õ") == "â" and res["unknown_tokens"] > 0


def test_bang_lech_dau_chi_hoc_chu_co_dau_va_bang_rong_dung_bang_mac_dinh(tmp_path):
    """J-44: lỗi gõ ở chữ không dấu ("d" thay "r") không vào bảng học; file bảng rỗng được đọc thành None để tone_shift dùng TONE_SHIFT_MAP chứ không tắt hẳn."""
    vocab = frozenset("rừng đàn".split())
    res = Df.learn_tone_shift_map(["dừng " * 80], vocab, min_count=10, min_share=0.1)
    assert res["map"] == {} and all(p["wrong"].isascii() is False for p in res["pairs"])
    f = tmp_path / "tone_shift.json"
    f.write_text(json.dumps({"map": {}, "pairs": []}), encoding="utf-8")
    assert Df.load_tone_shift_map(f) is None and Df.load_tone_shift_map(tmp_path / "none.json") is None
    f.write_text(json.dumps({"map": {"ỏ": "á"}, "pairs": []}), encoding="utf-8")
    assert Df.load_tone_shift_map(f) == {"ỏ": "á"}


def test_tieu_de_giua_doan_khong_dem_trich_dan():
    """find_journal_headers thấy tiêu đề giữa dòng (số + năm / ISSN) nhưng không thấy câu trích dẫn có dấu phẩy trước "số", cụm viết thường giữa câu, hay tiêu đề không có năm."""
    assert find_journal_headers("Nội dung. Tạp chí Dân tộc học số 3 - 2004 tiếp.")
    assert find_journal_headers("Khoa học Tạp chí Khoa học Xã hội ISSN 1859-0136 tiếp")
    for keep in ("Tạp chí Nghiên cứu Tôn giáo, số 8, 2008, tr. 5", "đăng trên tạp chí Kinh tế số 12 - 2015", "Tạp chí Kinh tế số 12 không năm"):
        assert not find_journal_headers(keep), keep
    cut, n = strip_mid_headers(["Câu đầu. Tạp chí Kinh tế số 12 - 2015 câu sau.\nTạp chí Dân tộc học số 3 - 2004\n\nXem Tạp chí Tôn giáo, số 8, 2008, tr. 5."])
    assert n == 2 and cut == ["Câu đầu. câu sau.\n\nXem Tạp chí Tôn giáo, số 8, 2008, tr. 5."]
    assert "mid_header" in A.error_flags(_quality_row(0, "Mở " * 100 + "Tạp chí Dân tộc học số 3 - 2004 nữa"))


def test_audit_do_cac_bo_do_moi_va_bo_journal_header():
    """audit_rows đếm mid_header / gibberish / dropped_letters / tone_shift / joined_words trong đoạn giữ; journal_header không còn; excerpt trích quanh chỗ rác."""
    rows = [_quality_row(i, vi_text(3, i)) for i in range(10)]
    rows[0]["text"] = "/, N9 = < 0 e H ; : " * 10
    rows[1]["text"] = "kh n th " + vi_text(2, 1) + " nh ng điều " + vi_text(1, 2) + " ph ng v n"
    rows[2]["text"] = vi_text(2, 3) + " nghiờn cứu bệnh nhõn"
    rows[3]["text"] = vi_text(2, 4) + " tươnghỗ và Tómtắt"
    res = A.audit_rows(rows, vocab=VOCAB)
    for name in ("gibberish", "dropped_letters", "tone_shift", "joined_words"):
        assert res["error_counts"].get(name) == 1, (name, res["error_counts"])
    assert "journal_header" not in res["error_counts"] and "journal_header" not in A.ERROR_CHECKS
    assert "kh n th" in A.excerpt_of("x " * 300 + "kh n th ok", "dropped_letters")


# ---------------------------------------------------------------------------------------------------------------- (1d) nhãn chuẩn


def test_nhan_other_chia_nho_va_cong_precision():
    """refine_label chia `other` theo cột note; evaluate_labels báo precision / recall từng bộ dò (gộp "all"), cờ gate_ok theo precision >= 0,9 và số đoạn tối thiểu, mẫu other để kiểm tay."""
    assert A.refine_label({"label": "other", "note": "mất chữ có dấu kh n th"}) == "dropped_letters"
    assert A.refine_label({"label": "other", "note": "chữ rác"}) == "garbage" and A.refine_label({"label": "other", "note": "công thức dàn phẳng"}) == "formula"
    assert A.refine_label({"label": "other", "note": "không rõ"}) == "other" and A.refine_label({"label": "clean", "note": "rác"}) == "clean"
    items = []
    for i in range(12):
        items.append({"doc_id": f"d{i}", "split": "tune" if i % 2 else "holdout", "label": "other", "note": "chữ rơi mất kh n", "quality_metrics": "{}", "text": f"Câu {i}: kh n th nh ng điều ph ng v n"})
    for i in range(12, 20):
        items.append({"doc_id": f"d{i}", "split": "tune", "label": "clean", "note": "", "quality_metrics": "{}", "text": vi_text(3, i)})
    res = A.evaluate_labels(items, VOCAB)
    d = res["detectors"]["dropped_letters"]
    assert d["all"]["tp"] == 12 and d["all"]["fp"] == 0 and d["all"]["precision"] == 1.0 and d["all"]["recall"] == 1.0 and d["gate_ok"] is True
    assert res["other_breakdown"] == {"dropped_letters": 12} and len(res["other_samples"]) == min(12, A.OTHER_SAMPLES)
    assert res["detectors"]["gibberish"]["gate_ok"] is False and res["detectors"]["tone_shift"]["targets"] == ["font_misread", "font_shift"]
    assert res["clean_false_flags"]["tune"]["dropped_letters"] == 0
    few = A.evaluate_labels(items[:2], VOCAB)["detectors"]["dropped_letters"]
    assert few["gate_ok"] is False  # gắn cờ ít hơn GATE_MIN_FLAGGED đoạn thì chưa qua cổng dù precision 1,0


def test_detect_all_dung_cho_ca_thu_vang():
    """detect_all trả tên bộ dò gắn cờ trên văn bản đơn lẻ (kể cả u_split, tone_shift, joined_words khi có tập âm tiết)."""
    assert A.detect_all("mối quan hệ tươnghỗ và Tómtắt", VOCAB) == {"joined_words"}
    assert "u_split" in A.detect_all("nước xư a và " + vi_text(2, 1)) and not A.detect_all(vi_text(3, 1), VOCAB)


# ---------------------------------------------------------------------------------------------------------------- (5) sửa chữ chỉ khi xác định


def test_hyphen_fix_bo_qua_cho_mo_ho():
    """J-42 P6: "M- ơng" (Mương hay Mường) và "h- ơng" không sửa và được đếm vào stats["ambiguous"]; dạng có dấu thanh rõ vẫn sửa."""
    stats: dict = {}
    assert restore_u_in_text("ông M- ơng đã", VOCAB, stats=stats) == ("ông M- ơng đã", 0) and stats["ambiguous"] == 1
    assert restore_u_in_text("ông tr-ờng ng-ời", VOCAB) == ("ông trường người", 2)
    st: dict = {}
    pages, n = restore_u_in_pages(["ông M- ơng đã", "tr-ờng ng-ời"], VOCAB, stats=st)
    assert pages == ["ông M- ơng đã", "trường người"] and n == 2 and st["ambiguous"] == 1  # lượt đủ dạng được dùng (>= min_hits), "M- ơng" bị bỏ vì mơ hồ
    st2: dict = {}
    restore_u_in_pages(["ông M- ơng đã", "ông tr-ờng", "ông ng-ời nữa", "t- ơng"], VOCAB, min_hits=2, stats=st2)
    assert st2["ambiguous"] >= 1


def test_repairs_trong_meta_cua_ban_ghi():
    """repairs_of giữ các quy tắc có số > 0 (kể cả hyphen_ambiguous); clean_paper_pages trả thêm số mid_headers / junk_lines / hyphen_ambiguous."""
    assert repairs_of({"hyphen_restored": 3, "hyphen_ambiguous": 1, "junk_lines": 0, "mid_headers": 2, "toc_pages": 0}) == {"hyphen_restored": 3, "hyphen_ambiguous": 1, "mid_headers": 2}
    pages = ["Tr-ờng học. Ng-ời dân. M- ơng đã. " + vi_text(8, 1) + " Tạp chí Dân tộc học số 3 - 2004 " + vi_text(4, 2)]
    _, st = clean_paper_pages(pages, None, VOCAB)
    assert st["hyphen_restored"] >= 1 and st["hyphen_ambiguous"] >= 1 and st["mid_headers"] == 1 and st["junk_lines"] == 0


# ---------------------------------------------------------------------------------------------------------------- (6) cắt mức dòng


def test_bo_dong_rac_khong_loai_ca_doan():
    """Dòng rác ("/, N9 = < 0 e H ; : ...") bị bỏ, dòng văn thường và dòng công thức có từ dài giữ nguyên; trang giữ số dòng trống."""
    junk = "/, N9 = < 0 e H ; : a b c d e f"
    assert is_junk_line(junk) and not is_junk_line("Đây là một câu tiếng Việt bình thường có nhiều từ.") and not is_junk_line("a = b + c nhưng giá trị thay đổi theo thời gian dài")
    cut, n = strip_junk_lines(["Câu một.\n" + junk + "\n\nCâu hai."])
    assert n == 1 and cut == ["Câu một.\n\nCâu hai."]


# ---------------------------------------------------------------------------------------------------------------- quality: mã loại cứng chỉ bật khi đặt ngưỡng


def test_so_do_moi_o_quality_chi_do_khi_chua_dat_nguong(monkeypatch):
    """compute_metrics có bốn số đo mới; HARD_DETECTOR_THRESHOLDS mặc định None nên không đoạn nào bị loại; đặt ngưỡng thì mã cùng tên xuất hiện và đoạn thành band D."""
    text = "kh n th " + vi_text(8, 1) + " nh ng ph ng v n " + vi_text(8, 2)
    m = Q.compute_metrics(text, VOCAB)
    assert m["dropped_letters"] == 2 and {"gibberish", "joined_words", "tone_shift"} <= set(m)
    profile = make_cfg().profile("vjol")
    _, codes = Q.score_metrics(m, "vi", profile)
    assert "dropped_letters" not in codes and all(v is None for v in Q.HARD_DETECTOR_THRESHOLDS.values())
    monkeypatch.setitem(Q.HARD_DETECTOR_THRESHOLDS, "dropped_letters", 2)
    score, codes = Q.score_metrics(m, "vi", profile)
    assert "dropped_letters" in codes and Q.band_of(score) == "D" and "dropped_letters" in Q.HARD_REASONS


# ---------------------------------------------------------------------------------------------------------------- (3), (4) OCR lại cả bài, nhóm font xấu


class _EchoOcr:
    """OCR giả: trả lại chính lớp chữ của trang (độ giống ~1), hoặc một câu khác hẳn khi `wrong`."""

    def __init__(self, wrong: bool = False) -> None:
        """`wrong` True thì trả câu không liên quan (độ giống thấp)."""
        self.wrong, self.calls = wrong, 0

    def ocr_pages(self, pages, render=None):
        """Kết quả OCR giả cho từng trang."""
        self.calls += 1
        return [{"text": "Một câu hoàn toàn khác không liên quan gì đến trang này " * 8 if self.wrong else p.get_text(), "method": "ocr_vlm", "tier": "fake", "status": "ok"} for p in pages]


def _pdf(path, pages, producer: str = "", creator: str = ""):
    """Tạo PDF chữ với metadata producer / creator."""
    mixed_pdf(path, pages)
    if producer or creator:
        with pymupdf.open(path) as d:
            d.set_metadata({"producer": producer, "creator": creator})
            d.saveIncr()
    return path


def test_bai_lech_dau_roi_chu_di_ocr_ca_bai(tmp_path):
    """J-42 (3): reocr_reason trả defect_doc khi đủ trang trúng tín hiệu; extract_pdf đưa cả bài đi OCR với lý do đó; bài sạch giữ lớp chữ; hyphen / gibberish không nằm trong tín hiệu mặc định."""
    bad = "Kết quả nghiờn cứu trên bệnh nhõn Khỏnh cho thấy nghiờn cứu mới. " + vi_text(3, 1)
    good = vi_text(4, 2)
    assert R.reocr_reason([bad, bad, good, None], VOCAB) == g.REASON_DEFECT and R.reocr_reason([good, good, good], VOCAB) is None
    assert R.reocr_reason([bad, good, good, good, good, good, good], VOCAB) is None  # 1 trang < REOCR_MIN_PAGES
    assert R.reocr_reason(["Nội dung tr-ờng ng-ời đ-ợc nh-ng " * 6] * 4, VOCAB) is None  # gạch nối thay ư: OCR không sửa được (J-32)
    pdf = _pdf(tmp_path / "bad.pdf", [bad, bad, good])
    ocr = _EchoOcr()
    pages = extract_pdf(pdf, lambda: ocr, detect_garbled=True, ocr_all_if=lambda layer: R.reocr_reason(layer, VOCAB))
    assert [p["reason"] for p in pages] == [g.REASON_DEFECT] * 3 and ocr.calls == 1 and g.REASON_DEFECT in g.GARBLED_REASONS
    clean = extract_pdf(_pdf(tmp_path / "ok.pdf", [good, good, good]), lambda: _EchoOcr(), detect_garbled=True, ocr_all_if=lambda layer: R.reocr_reason(layer, VOCAB))
    assert all(p["method"] == "text" and "reason" not in p for p in clean)


def test_nhom_font_xau_ocr_ca_bai_va_phep_thu_canary(tmp_path):
    """J-42 (4): extract_pdf OCR cả bài của nhóm (producer, creator) khớp NGUYÊN; nhóm rỗng không khớp gì; canary gom nhóm, chọn bài theo băm, đo độ giống và chỉ ghi font_groups_bad.json khi có ngưỡng."""
    good = vi_text(6, 3)
    a = _pdf(tmp_path / "a.pdf", [good, good, good], "Prod X", "Creator Y")
    b = _pdf(tmp_path / "b.pdf", [good, good, good], "Prod X", "Creator Z")
    ocr = _EchoOcr()
    hit = extract_pdf(a, lambda: ocr, detect_garbled=True, ocr_all_groups=[("Prod X", "Creator Y")])
    assert [p["reason"] for p in hit] == [g.REASON_FONT_GROUP] * 3
    assert all("reason" not in p for p in extract_pdf(b, lambda: ocr, detect_garbled=True, ocr_all_groups=[("Prod X", "Creator Y")]))
    assert all("reason" not in p for p in extract_pdf(b, lambda: ocr, detect_garbled=True, ocr_all_groups=[("", "")]))
    assert FC.char_similarity("Hà Nội", "ha  nội") >= 0.8 > FC.char_similarity("Hà Nội", "xyz") and FC.char_similarity("", "x") == 0.0
    docs = [{"sha256": f"s{i}", "path": str(a if i % 2 else b), "article_id": str(i), "producer": "Prod X", "creator": "Creator Y" if i % 2 else "Creator Z"} for i in range(6)]
    docs.append({"sha256": "none", "path": str(a), "article_id": "9", "producer": "", "creator": ""})
    assert {k: len(v) for k, v in FC.pick_docs(docs, per_group=2, seed=0).items()} == {("Prod X", "Creator Y"): 2, ("Prod X", "Creator Z"): 2}
    assert FC.pick_docs(docs, 2, 0) == FC.pick_docs(list(reversed(docs)), 2, 0)
    ok = FC.run_canary(docs, lambda: _EchoOcr(), per_group=2, n_pages=2)
    assert {(x["producer"], x["creator"]) for x in ok} == {("Prod X", "Creator Y"), ("Prod X", "Creator Z")} and all(x["similarity_mean"] > 0.95 and x["pages"] >= 2 for x in ok)
    bad = FC.run_canary(docs, lambda: _EchoOcr(wrong=True), per_group=2, n_pages=2)
    assert all(x["similarity_mean"] < 0.5 for x in bad) and len(FC.bad_groups(bad, 0.5)) == 2 and FC.bad_groups(ok, 0.5) == []
    assert all(x["overlap_mean"] > 0.95 and x["pages_overlap_low"] == 0 for x in ok) and all(x["overlap_mean"] < FC.WORD_OVERLAP_LOW and x["pages_overlap_low"] >= 2 for x in bad)  # J-48 (b)
    labeled = FC.label_reference(bad, docs, [{"article_id": "1", "label": "clean"}, {"article_id": "3", "label": "hyphen_font"}, {"article_id": "2", "label": "garbage"}, {"article_id": "7", "label": "clean"}])
    by_creator = {x["creator"]: (x["labeled_clean"], x["labeled_bad"]) for x in labeled}
    assert by_creator == {"Creator Y": (1, 0), "Creator Z": (0, 1)}  # hyphen_font không tính là lỗi font (OCR không sửa được), đếm ở labeled_hyphen
    assert {x["creator"]: x["labeled_hyphen"] for x in labeled} == {"Creator Y": 1, "Creator Z": 0}
    det: list[dict] = []
    only = FC.run_canary(docs, lambda: _EchoOcr(wrong=True), per_group=2, n_pages=2, only_groups=[("Prod X", "Creator Y")], details=det)
    assert [x["creator"] for x in only] == ["Creator Y"] and det and all(d["creator"] == "Creator Y" and d["layer"] and d["ocr"] and "similarity" in d and "overlap" in d for d in det)
    FC.write_canary(labeled, tmp_path / "out")
    assert (tmp_path / "out" / "canary.csv").is_file() and not (tmp_path / "out" / "font_groups_bad.json").exists()
    FC.write_canary(labeled, tmp_path / "out", threshold=0.5)
    groups = R.load_font_groups(tmp_path / "out" / "font_groups_bad.json")
    assert set(groups) == {("Prod X", "Creator Y"), ("Prod X", "Creator Z")} and R.font_groups_tag(groups) != "" and R.font_groups_tag([]) == ""
    assert R.load_font_groups(tmp_path / "khong_co.json") == []


def test_joined_words_khong_con_la_tin_hieu_ocr_lai_ca_bai():
    """J-48 (a): bài toàn tên riêng / từ dính liền KHÔNG bị OCR lại; joined_words vẫn được đo (ALL_SIGNALS) và quét trong plan_reocr; EXTRACT_VERSION đã tăng."""
    from vi_corpus.vjol.extract import EXTRACT_VERSION

    joined = "mối quan hệ tươnghỗ và Tómtắt " * 8
    assert "joined_words" in R.page_flags(R.page_defects(joined, VOCAB))
    assert R.reocr_reason([joined] * 4, VOCAB) is None and "joined_words" not in R.REOCR_SIGNALS and "joined_words" in R.ALL_SIGNALS
    assert R.reocr_reason([joined] * 4, VOCAB, signals=R.ALL_SIGNALS) == g.REASON_DEFECT  # vẫn bật lại được khi muốn đo
    assert any("joined_words" in name for name in PR.SIGNAL_SETS) and int(EXTRACT_VERSION) >= 7


def test_do_trung_tu_bat_chu_mat_dau_ma_do_giong_ky_tu_bo_qua():
    """J-48 (b): word_overlap không xét thứ tự từ, thấp khi mất dấu; canary ghi overlap từng trang và nhóm; missed_by_garbled đếm bài có trang trùng thấp mà vẫn là `text` (J-48 c)."""
    ocr = "Ảnh hưởng của tỉ lệ vốn đến luật đầu tư của các doanh nghiệp nhỏ " * 4
    lost = "Anh huong cua ti le von den luat dau tu cua cac doanh nghiep nho " * 4
    assert FC.word_overlap(ocr, ocr) == 1.0 and FC.word_overlap(ocr, "") == 0.0 and FC.word_overlap("a b c d", "d c b a") == 1.0  # không xét thứ tự đọc
    assert FC.word_overlap(ocr, lost) < 0.3 < FC.char_similarity(ocr, lost)  # đúng ca J-47: giống ký tự cao, trùng từ thấp
    rows = [{"path": "/x/a.pdf", "page": 0, "overlap": 0.05, "producer": "P", "creator": "C", "article_id": "1"},
            {"path": "/x/a.pdf", "page": 1, "layer": lost, "ocr": ocr, "producer": "P", "creator": "C", "article_id": "1"},  # dòng cũ: tính từ chữ đã cắt
            {"path": "/x/a.pdf", "page": 2, "overlap": 0.9, "producer": "P", "creator": "C", "article_id": "1"},
            {"path": "/x/b.pdf", "page": 0, "overlap": 0.1, "producer": "P", "creator": "C", "article_id": "2"},
            {"path": "/x/c.pdf", "page": 0, "overlap": 0.1, "producer": "P", "creator": "C", "article_id": "3"}]
    res = FC.missed_by_garbled(rows, {"/x/a.pdf": ["text", "ocr_vlm", "text"], "/x/b.pdf": ["ocr_vlm"]})
    assert (res["pages_measured"], res["pages_low"], res["pages_low_text"], res["docs_measured"], res["docs_low"], res["docs_missed"], res["no_checkpoint"]) == (4, 3, 1, 2, 2, 1, 1)
    assert res["missed"][0]["article_id"] == "1" and res["missed"][0]["pages"] == [[0, 0.05]] and sum(res["histogram"].values()) == 5


def test_check_garbled_missed_doc_jsonl_co_ky_tu_ngat_dong_unicode(tmp_path):
    """Hồi quy J-49: canary_pages.jsonl ghi ensure_ascii=False nên lớp chữ có thể chứa \\u2028 / \\x85 trong chuỗi JSON; script đọc theo "\\n" chứ không dùng splitlines()."""
    import subprocess
    import sys
    from pathlib import Path

    pdf = tmp_path / "raw" / "a.pdf"
    pdf.parent.mkdir()
    pdf.touch()
    (tmp_path / "interim" / "vjol_text").mkdir(parents=True)
    (tmp_path / "interim" / "vjol_text" / "s.json").write_text(json.dumps({"sha256": "s", "paths": [str(pdf)], "error": None, "skipped": None, "pages": [{"method": "text"}]}), encoding="utf-8")
    out = tmp_path / "processed" / "vjol_font_canary"
    out.mkdir(parents=True)
    row = {"path": str(pdf), "page": 0, "overlap": 0.1, "layer": "chữ\u2028hỏng\x85\x0b", "ocr": "x", "producer": "P", "creator": "C", "article_id": "7"}
    (out / "canary_pages.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "vjol" / "check_garbled_missed.py"
    run = subprocess.run([sys.executable, str(script), "--data-root", str(tmp_path)], capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    assert json.loads((out / "missed_by_garbled.json").read_text(encoding="utf-8"))["docs_missed"] == 1


def test_extract_version_them_dau_nhom_font(tmp_path):
    """extract_version đổi khi danh sách nhóm font xấu đổi (checkpoint cũ bị làm lại) và vẫn được version_is_current nhận ra."""
    from vi_corpus.vjol import extract as vx

    v0, v1 = vx.version_for(VOCAB), vx.version_for(VOCAB, None, [("P", "C")])
    assert v0 != v1 and "+fg" in v1 and vx.version_is_current(v1) and vx.version_is_current(v0) and v1 in vx.accepted_versions(VOCAB, None, [("P", "C")])
    assert v0 not in vx.accepted_versions(VOCAB, None, [("P", "C")])


# ---------------------------------------------------------------------------------------------------------------- (2) kế hoạch OCR lại


def test_plan_reocr_dem_trang_gio_va_canh_bao(tmp_path):
    """measure_docs / summarize đếm bài và trang theo tín hiệu, tính giờ OCR cho mẫu và cả kho, tách bài chỉ có gạch nối thay ư, cảnh báo khi vượt 15% số trang; write_plan ghi csv."""
    bad = "Kết quả nghiờn cứu trên bệnh nhõn Khỏnh cho thấy nghiờn cứu mới. " + vi_text(3, 1)
    good = vi_text(4, 2)
    hyph = "Nội dung tr-ờng ng-ời đ-ợc nh-ng " * 4
    text_dir = tmp_path / "interim" / "vjol_text"
    text_dir.mkdir(parents=True)

    def ckpt(sha: str, texts: list[str], method: str = "text") -> None:
        """Ghi một checkpoint giả."""
        (text_dir / f"{sha}.json").write_text(json.dumps({"sha256": sha, "paths": [f"raw/vjol/2010/{sha}_1.pdf"], "article_id": sha, "pages": [{"text": t, "method": method} for t in texts]}), encoding="utf-8")

    ckpt("a1", [bad, bad, good])
    ckpt("b2", [good, good, good, good])
    ckpt("c3", [hyph, hyph, good])
    ckpt("d4", [bad, good, good, good, good, good, good])
    docs = PR.measure_docs(tmp_path, VOCAB, with_pdf_meta=False, metadata={"a1_1": {"journal_slug": "tcA"}})
    assert [d["sha256"] for d in docs] == ["a1", "b2", "c3", "d4"] and docs[0]["journal"] == "tcA"
    assert PR.decide(docs[0]) == (True, 3) and PR.decide(docs[1]) == (False, 0) and PR.decide(docs[3]) == (False, 0)
    summary = PR.summarize(docs, corpus_pdfs=400)
    assert summary["docs"] == 4 and summary["default"]["docs"] == 1 and summary["default"]["pages"] == 3 and summary["docs_with_signal"]["tone_shift"] == 2
    assert summary["hyphen_only_docs"] == 1 and summary["default"]["pages_corpus"] == 300 and summary["warning"] is not None
    assert abs(summary["default"]["hours"] - PR.hours(3)) < 0.01 and len(summary["sweep"]) == len(PR.SIGNAL_SETS) * len(PR.SHARE_SWEEP)
    assert PR.write_plan(docs, tmp_path / "plan") == 3 and "tone_shift" in (tmp_path / "plan" / "reocr_plan.csv").read_text(encoding="utf-8")
    out = PR.plan(tmp_path, root=tmp_path / "raw" / "vjol", syllables=tmp_path / "khong_co.json", with_pdf_meta=False)
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["docs"] == 4


# ---------------------------------------------------------------------------------------------------------------- (7) mẫu thi cuối


def test_mau_thi_cuoi_ngau_nhien_loai_bai_cu_va_bao_cao_wilson():
    """make_final_sample: đúng n đoạn, ngẫu nhiên theo seed, loại bài đã nằm trong mẫu cũ, tối đa 3 đoạn mỗi bài; final_report tính tỷ lệ, Wilson 95%, ngân sách và độ đồng ý xem tay."""
    rows = [_quality_row(i, vi_text(3, i), band="A" if i % 3 else "B", journal="j1") for i in range(300)]
    rows += [{**_quality_row(i, vi_text(3, i)), "doc_id": f"vjol:{i}:0#c1"} for i in range(50)]
    sample = A.make_final_sample(rows, n=100, seed=A.FINAL_SEED, exclude_articles=frozenset(str(i) for i in range(20)))
    assert len(sample) == 100 and all(int(it["article_id"]) >= 20 for it in sample) and all(it["label"] == "" and it["group"] == "thi_cuối" for it in sample)
    assert sample == A.make_final_sample(rows, n=100, seed=A.FINAL_SEED, exclude_articles=frozenset(str(i) for i in range(20)))
    assert {it["doc_id"] for it in sample} != {it["doc_id"] for it in A.make_final_sample(rows, n=100, seed=1, exclude_articles=frozenset())}
    assert A.FINAL_SEED == 20261026 and A.FINAL_MIN_N == 600
    lo, hi = A.wilson_interval(0, 300)
    assert lo == 0.0 and 0.01 < hi < 0.014 and A.wilson_interval(0, 0) == (0.0, 1.0)  # 0 lỗi trong 300 chỉ chứng minh < 1,3%
    items = [{"doc_id": f"d{i}", "band": "A", "label": "clean", "note": "", "text": "t"} for i in range(594)]
    items += [{"doc_id": "x1", "band": "A", "label": "hyphen_font", "note": "", "text": "t"}, {"doc_id": "x2", "band": "B", "label": "running_header", "note": "", "text": "t"},
              {"doc_id": "x3", "band": "A", "label": "other", "note": "chữ rác", "text": "t"}, {"doc_id": "x4", "band": "C", "label": "reference", "note": "", "text": "t"},
              {"doc_id": "x5", "band": "A", "label": "clean", "note": "", "text": "t"}, {"doc_id": "x6", "band": "A", "label": "clean", "note": "", "text": "t"}]
    review = A.make_review_sample(items, 50)
    assert len(review) == 50 and all(r["user_label"] == "" for r in review)
    for r in review[:40]:
        r["user_label"] = r["label"]
    for r in review[40:44]:
        r["user_label"] = "other"
    rep = A.final_report(items, review)
    b = rep["budgets"]
    assert rep["n"] == 600 and b["chu_hong_sai_nghia"]["k"] == 2 and b["tieu_de_hoac_tham_khao"]["k"] == 2 and b["rac_hoan_toan_dai_A"]["k"] == 1
    assert b["chu_hong_sai_nghia"]["within_budget"] and not b["chu_hong_sai_nghia"]["proven"] and b["tieu_de_hoac_tham_khao"]["proven"] and not b["rac_hoan_toan_dai_A"]["within_budget"]
    assert rep["review"]["reviewed"] == 44 and rep["review"]["agree"] in range(40, 45)
