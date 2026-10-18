"""
Test xử lý chữ hỏng do font PDF (J-10): bộ dò theo trang và theo bài, đưa trang lỗi font qua chuỗi OCR VLM (vẽ lại cả trang),
giữ chữ hỏng để kiểm tra, làm lại checkpoint cũ khi đổi phiên bản trích text, luật chất lượng garbled_font và script đo
scripts/vjol/check_garbled.py. Không cần mạng, GPU hay mô hình (chuỗi OCR được thay bằng giả).
"""

import csv
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from tests.test_ocr_vlm import GOOD, make_cascade, ok
from tests.test_pipeline import make_cfg, vi_text
from tests.test_vjol import REPO, CountingOcr, scan_pdf, text_pdf, write_ckpt
from vi_corpus.common import garbled as g
from vi_corpus.common.ocr_cascade import CascadePageOcr, page_image_for_vlm
from vi_corpus.common.pdf_text import extract_pdf, tcvn3_to_unicode
from vi_corpus.common.registry import SOURCES
from vi_corpus.pipeline.quality import HARD_REASONS, band_of, compute_metrics, score_metrics
from vi_corpus.vjol import extract as vx
from vi_corpus.vjol import garbled_check as gc
from vi_corpus.vjol.records import iter_records

# Hai kiểu chữ hỏng thật đã gặp ở VJOL (J-10).
GARBLED = ("Baỷng hieọu ủoọ ngửụứi lửợi naõy Giuỏp hoồc sinh traỏnh nửụực tửứ ngửụừng ủửụùc hoaùt ủoọng nghieọp vuù "
           "Chuực Mửứng Naờm Mụựi")
GOOD_VI = "Phân tích kinh tế vĩ mô cho thấy lạm phát chịu tác động của nhiều yếu tố khác nhau trong nước và quốc tế."
VNI = ("Thöïc traïng xuaát khaåu thuûy saûn cuûa Vieät Nam ñeán thò tröôøng baéc myõ trong boái caûnh hieäp ñònh nguyeãn "
       "quoác dung ñaàu tö vaøo noâng nghieäp coâng ngheä cao ")  # họ VNI: nhiều ñ ö ï ø å, ít dấu thanh
CASEMIX = "TRoNg Boái cAûNh xuAáT khAåu Thuûy sAûN cuûA VieäT NAm ñeáN Thò TRöôøNg BAéc myõ hieäp ñòNh "
SHIFTED = "7K{QJWLQYjJLảLSKiSNKRDKọFF{QJQJKệ 7ạSFKt9ậWOLệXYj;k\\GựQJ7ậSVố +uQKLưu đồFủDP{KuQKGựa trên đạLOमGjLKạQ "
# Họ sai dấu kiểu Windows-1258: ỏ ờ õ thay cho á ê â, mỗi âm tiết vẫn chỉ một dấu thanh nên tín hiệu tone không thấy.
WIN1258 = "Phỏt triển sõn xuỗt cam trờn bàn Sõu bệnh trờn lỏ cam nhỏ "
WIN1258_OK = "Phát triển sản xuất cam trên bàn Sâu bệnh trên lá cam nhỏ"
ENGLISH = "This paper studies the effect of inflation on household consumption across several developing economies in Asia."


def base_vocab() -> frozenset[str]:
    """Tập âm tiết giả đủ để các trang tiếng Việt thường trong test không bị tín hiệu unknown bắt."""
    return make_vocab(GOOD_VI, vi_text(40, 1), vi_text(40, 2), WIN1258_OK)


def make_vocab(*texts: str) -> frozenset[str]:
    """Tập âm tiết giả: mọi từ chữ cái (chữ thường) của các văn bản đưa vào."""
    import re

    return frozenset(w.lower() for t in texts for w in re.findall(r"[^\W\d_]+", t))


def png(data: bytes) -> bool:
    """Byte ảnh có phải PNG (ảnh vẽ lại cả trang) hay không; ảnh nhúng trong test là JPEG."""
    return data[:4] == b"\x89PNG"


def mixed_pdf(path: Path, pages: list[str | None], image_on: tuple[int, ...] = ()) -> None:
    """PDF mỗi phần tử của `pages` là một trang chữ (None = trang trắng); trang có chỉ số trong image_on có thêm một ảnh JPEG nhỏ."""
    doc = pymupdf.open()
    for i, text in enumerate(pages):
        page = doc.new_page()
        if text:
            page.insert_htmlbox(pymupdf.Rect(30, 30, 565, 810), f"<p style='font-size:9pt'>{text}</p>")
        if i in image_on:
            import io

            buf = io.BytesIO()
            Image.new("RGB", (40, 40), (200, 30, 30)).save(buf, format="JPEG")
            page.insert_image(pymupdf.Rect(400, 700, 440, 740), stream=buf.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


def ocr_engine(seen: list):
    """Chuỗi OCR giả: ghi lại ảnh nhận được, trả một câu cố định (đủ dài để qua guard)."""
    def script(im):
        """Ghi ảnh rồi trả văn bản OCR."""
        seen.append(im)
        return ok("Bảng số liệu đã được OCR đúng chính tả. " + GOOD)

    return lambda: CascadePageOcr(make_cascade(script), concurrency=2)


# --------------------------------------------------------------------------------------------------------------------
# Bộ dò
# --------------------------------------------------------------------------------------------------------------------


def test_bo_do_phan_biet_chu_hong_voi_tieng_viet_dung_va_tieng_anh():
    """Chữ hỏng bị nghi; tiếng Việt đúng, tiếng Anh, tiếng Việt viết rời dấu (NFD) không bị nghi; văn bản quá ngắn không kết luận."""
    assert g.is_garbled(GARBLED)
    assert not g.is_garbled(GOOD_VI) and not g.is_garbled(ENGLISH) and not g.is_garbled(vi_text(8, 3))
    assert g.garbled_ratio(unicodedata.normalize("NFD", GOOD_VI)) == g.garbled_ratio(GOOD_VI)
    assert g.garbled_ratio(unicodedata.normalize("NFD", GARBLED)) == g.garbled_ratio(GARBLED)
    assert not g.is_garbled("Baỷng ngửụứi lửợi ủửụùc")  # chỉ 4 từ: chưa đủ để kết luận
    assert g.garbled_ratio("") == (0.0, 0) and g.garbled_ratio("12 3 4") == (0.0, 0)


def test_nguong_trang_5_phan_tram_nhung_nguong_cung_quality_van_8():
    """J-13: trang có 1/20 từ nhiều dấu thanh (5%) bị nghi để OCR lại, nhưng đoạn văn đó chưa bị mã cứng garbled_font (ngưỡng 8%)."""
    text = " ".join(["kinh"] * 19 + ["ngửụứi"])
    assert g.GARBLED_RATIO == 0.05 and g.GARBLED_HARD_RATIO == 0.08
    assert g.is_garbled(text) and g.plan_garbled([text]) == [g.REASON_PAGE]
    m = compute_metrics(text)
    assert m["garbled"] == pytest.approx(0.05)
    assert "garbled_font" not in score_metrics(m, "vi", make_cfg().profile("vjol"), 0.99, "{}")[1]


def test_tin_hieu_chu_hoa_chen_giua_tu():
    """Họ chữ lệch mã ASCII và chữ hoa lộn xộn ("xuAáT", "TRoNg") bị tín hiệu case bắt; tiếng Việt, tiếng Anh, tên riêng thường không."""
    assert g.case_mix_ratio(CASEMIX)[0] > 0.2 and g.case_mix_ratio(SHIFTED * 3)[0] > 0.2
    for ok_text in (GOOD_VI, ENGLISH, vi_text(8, 3), GOOD_VI * 12 + " Công ty McDonald bán iPhone và dung dịch NaCl."):  # vài tên riêng lẫn chữ hoa trong trang bình thường
        assert g.case_mix_ratio(ok_text)[0] < 0.05, ok_text
    assert g.case_mix_ratio("") == (0.0, 0) and g.case_mix_ratio("a b c") == (0.0, 0)  # từ < 4 chữ cái không tính


def test_tin_hieu_chu_cai_latin_1_la():
    """Họ VNI (ñ ö ï ø å thay cho đ ư ị ờ ả) bị tín hiệu foreign bắt; tiếng Việt đúng thì không; trang quá ít chữ cái không kết luận."""
    assert g.foreign_letter_ratio(VNI)[0] > 0.05
    assert g.foreign_letter_ratio(GOOD_VI * 3)[0] == 0.0
    assert g.page_signals("Thöïc ñeáN") == [] and "foreign" in g.page_signals(VNI)  # dưới 100 chữ cái thì không xét


def test_tin_hieu_am_tiet_la_chi_tinh_tu_latinh_va_can_vocab():
    """Họ Windows-1258 có nhiều âm tiết lạ nên bị bắt; trang chữ Hán không bị phạt; không có vocab thì bỏ tín hiệu; load_vocab đọc đúng file."""
    vocab = base_vocab()
    assert g.unknown_syllable_rate(WIN1258 * 3, vocab)[0] > 0.4
    assert g.unknown_syllable_rate(GOOD_VI * 2, vocab)[0] < 0.1
    chinese = "越语 这是一个关于称谓语的研究 " * 20 + GOOD_VI * 2
    assert g.unknown_syllable_rate(chinese, vocab)[0] < 0.1  # chữ Hán bị bỏ khỏi mẫu số
    assert g.page_signals(WIN1258 * 3, vocab=vocab) == ["unknown"] and g.page_signals(WIN1258 * 3) == []
    assert g.unknown_syllable_rate("12 3", vocab) == (0.0, 0)


def test_load_vocab(tmp_path):
    """load_vocab đọc khoá "syllables" của syllables.json; thiếu file thì None."""
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"syllables": ["phân", "tích"]}, ensure_ascii=False), encoding="utf-8")
    assert g.load_vocab(f) == frozenset({"phân", "tích"}) and g.load_vocab(tmp_path / "khong_co.json") is None


def test_ke_hoach_dung_ca_ba_tin_hieu():
    """plan_garbled: trang chỉ có một tín hiệu mới cũng bị nghi (1/10 trang thì chỉ trang đó); tắt vocab thì họ Windows-1258 lọt."""
    vocab = base_vocab()
    pages = [GOOD_VI * 3] * 9
    for text, sig in ((VNI * 2, "foreign"), (CASEMIX * 3, "case"), (WIN1258 * 3, "unknown")):
        plan = g.plan_garbled([text, *pages], vocab=vocab)
        assert plan == [g.REASON_PAGE] + [None] * 9, sig
        assert sig in g.page_signals(text, vocab=vocab)
    assert g.plan_garbled([WIN1258 * 3, *pages]) == [None] * 10  # thiếu vocab: không thấy


def test_trang_tcvn3_da_doi_sang_unicode_khong_bi_nghi():
    """Chữ TCVN3 đã đổi sang Unicode là tiếng Việt đúng nên không bị coi là chữ hỏng."""
    text = tcvn3_to_unicode("Phè Ngôy Nh− Kon Tum " * 6)
    assert text.startswith("Phố Ngụy Như Kon Tum") and not g.is_garbled(text)


def test_tin_hieu_thu_hai_tu_mot_chu_cai():
    """Chữ mất kiểu "đ u" có tỷ lệ từ một chữ cái cao hơn hẳn tiếng Việt đúng (chỉ đo)."""
    assert g.single_letter_ratio("đ u t ư v à o k inh d oan h") > 0.4
    assert g.single_letter_ratio(GOOD_VI) == 0.0 and g.single_letter_ratio("") == 0.0


def test_ke_hoach_theo_trang_va_nang_len_cap_bai():
    """
    Một trang hỏng trong 10 trang: chỉ trang đó. 4/10 trang hỏng (>= 30%): cả bài, trang bình thường mang lý do garbled_doc, trang
    scan (None), trang trắng, trang quá ngắn bị nghi thì không bị tính. Không có lớp chữ nào thì không làm gì.
    """
    ok_pages = [GOOD_VI] * 9
    plan = g.plan_garbled([GARBLED, *ok_pages])
    assert plan == [g.REASON_PAGE] + [None] * 9
    plan = g.plan_garbled([GARBLED] * 4 + [GOOD_VI] * 5 + [None])
    assert plan == [g.REASON_PAGE] * 4 + [g.REASON_DOC] * 5 + [None]
    assert g.plan_garbled([GARBLED, GOOD_VI, None, "   "])[2:] == [None, None]
    assert g.plan_garbled([None, None]) == [None, None] and g.plan_garbled([]) == []
    short = "Baỷng ngửụứi"
    assert g.plan_garbled([short, GOOD_VI]) == [None, None]  # trang ngắn không kết luận
    # trang ngắn nằm trong bài bị nâng lên cấp bài thì bị kéo theo
    assert g.plan_garbled([GARBLED, GARBLED, short, GOOD_VI]) == [g.REASON_PAGE, g.REASON_PAGE, g.REASON_DOC, g.REASON_DOC]
    # ngưỡng cấp bài chỉnh được
    assert g.plan_garbled([GARBLED] + [GOOD_VI] * 9, doc_share=0.1) == [g.REASON_PAGE] + [g.REASON_DOC] * 9


# --------------------------------------------------------------------------------------------------------------------
# Trích text: trang lỗi font đi qua chuỗi OCR
# --------------------------------------------------------------------------------------------------------------------


def test_trang_loi_font_ocr_ve_lai_ca_trang_va_giu_chu_hong(tmp_path):
    """
    Bài 8 trang, trang 2 lỗi font và có một ảnh nhúng: chỉ trang đó được OCR, ảnh gửi đi là PNG vẽ lại cả trang (không phải ảnh JPEG nhúng),
    mục trang ghi method ocr_vlm + reason + garbled_text (chữ hỏng giữ để kiểm tra), các trang khác giữ nguyên lớp chữ.
    """
    pdf = tmp_path / "a.pdf"
    mixed_pdf(pdf, [vi_text(4, k) for k in range(1)] + [GARBLED * 2] + [vi_text(4, k + 5) for k in range(6)], image_on=(1,))
    seen: list = []
    pages = extract_pdf(pdf, ocr_engine(seen), detect_garbled=True)
    assert [p["method"] for p in pages] == ["text", "ocr_vlm"] + ["text"] * 6
    assert len(seen) == 1 and png(seen[0])
    bad = pages[1]
    assert bad["reason"] == g.REASON_PAGE and "Baỷng" in bad["garbled_text"] and "Baỷng" not in bad["text"]
    assert bad["status"] == "ok" and bad["text"].startswith("Bảng số liệu")
    assert all("reason" not in p for i, p in enumerate(pages) if i != 1)


def test_trang_ho_vni_di_ocr_va_ghi_tin_hieu(tmp_path):
    """Trang chữ họ VNI (ít dấu thanh) bị tín hiệu foreign bắt và đi OCR; mục trang ghi signals; trang tiếng Việt thường thì không."""
    pdf = tmp_path / "v.pdf"
    mixed_pdf(pdf, [VNI * 2] + [vi_text(5, k) for k in range(1, 8)])
    seen: list = []
    pages = extract_pdf(pdf, ocr_engine(seen), detect_garbled=True)
    assert pages[0]["method"] == "ocr_vlm" and pages[0]["reason"] == g.REASON_PAGE and "foreign" in pages[0]["signals"]
    assert "garbled_text" in pages[0] and all(p["method"] == "text" for p in pages[1:]) and len(seen) == 1


def test_bai_co_nhieu_trang_hong_ca_bai_di_ocr(tmp_path):
    """3/5 trang lỗi font: cả 5 trang có chữ đi OCR; trang bình thường mang lý do garbled_doc; trang trắng vẫn trắng."""
    pdf = tmp_path / "b.pdf"
    mixed_pdf(pdf, [GARBLED * 2] * 3 + [vi_text(4, 1), vi_text(4, 2), None])
    seen: list = []
    pages = extract_pdf(pdf, ocr_engine(seen), detect_garbled=True)
    assert [p["method"] for p in pages] == ["ocr_vlm"] * 5 + ["empty"]
    assert [p["reason"] for p in pages[:5]] == [g.REASON_PAGE] * 3 + [g.REASON_DOC] * 2
    assert len(seen) == 5 and all(png(s) for s in seen)
    assert "reason" not in pages[5]


def test_mac_dinh_khong_do_chu_hong_nen_nguon_khac_chay_nhu_cu(tmp_path):
    """Không bật detect_garbled (giáo trình): trang lỗi font vẫn là "text" và không gọi OCR."""
    pdf = tmp_path / "c.pdf"
    mixed_pdf(pdf, [GARBLED * 2])
    ocr = CountingOcr()
    pages = extract_pdf(pdf, lambda: ocr)
    assert [p["method"] for p in pages] == ["text"] and ocr.calls == 0


def test_trang_loi_font_bi_flagged_thi_bo_trang_khong_dung_lai_chu_hong(tmp_path):
    """OCR trang lỗi font bị guard loại: trang rỗng, chữ hỏng chỉ nằm ở garbled_text, bản ghi không chứa chữ hỏng."""
    raw = tmp_path / "raw/vjol"
    mixed_pdf(raw / "2019/7_1.pdf", [GARBLED * 2] + [vi_text(6, k) for k in range(1, 8)])
    loop = "Đọc đúng phần đầu\n" + " . " * 400
    cascade = make_cascade(lambda im: ok(loop), lambda im: ok(loop))
    assert vx.run_extract(raw, tmp_path, lambda: CascadePageOcr(cascade, 2)) == 0
    page = json.loads(next((tmp_path / vx.TEXT_DIR).glob("*.json")).read_text(encoding="utf-8"))["pages"][0]
    assert page["status"] == "flagged" and page["text"] == "" and "Baỷng" in page["garbled_text"] and page["reason"] == g.REASON_PAGE
    rec = next(iter_records(SOURCES["vjol"], tmp_path, with_meta=True))
    assert "Baỷng" not in rec["text"] and "ủoọ" not in rec["text"]
    assert json.loads(rec["meta"])["garbled_pages"] == 1


def test_tat_ocr_trang_loi_font_thanh_needs_ocr_va_khong_vao_ban_ghi(tmp_path):
    """--no-ocr: trang lỗi font thành needs_ocr kèm reason, record-time thành trang rỗng; bật OCR lại thì làm lại đúng bài đó."""
    raw = tmp_path / "raw/vjol"
    mixed_pdf(raw / "2019/7_1.pdf", [GARBLED * 2] + [vi_text(6, k) for k in range(1, 8)])
    assert vx.run_extract(raw, tmp_path, lambda: None, ocr_enabled=False) == 0
    rep = vx.report(raw, tmp_path)
    assert rep["pages"]["needs_ocr"] == 1 and rep["garbled_pages"] == {g.REASON_PAGE: 1} and rep["garbled_docs"] == 1
    assert "Baỷng" not in next(iter_records(SOURCES["vjol"], tmp_path))["text"]
    seen: list = []
    assert vx.run_extract(raw, tmp_path, ocr_engine(seen)) == 0
    assert vx.report(raw, tmp_path)["pages"] == {"ocr_vlm": 1, "text": 7} and len(seen) == 1


def test_ocr_kieu_cu_cho_trang_loi_font_cung_ve_lai_trang(tmp_path, monkeypatch):
    """Bộ OCR chỉ có .page_text (kiểu cũ): trang lỗi font vẫn được OCR và ảnh là ảnh vẽ lại cả trang (render=True)."""
    pdf = tmp_path / "d.pdf"
    mixed_pdf(pdf, [GARBLED * 2, vi_text(4, 1), vi_text(4, 2), vi_text(4, 3)], image_on=(0,))
    flags = []
    import vi_corpus.common.ocr as ocr_mod

    real = ocr_mod.page_image
    monkeypatch.setattr(ocr_mod, "page_image", lambda page, render=False: (flags.append(render), real(page, render))[1])
    ocr = CountingOcr()
    pages = extract_pdf(pdf, lambda: ocr, detect_garbled=True)
    assert [p["method"] for p in pages] == ["ocr", "text", "text", "text"] and ocr.calls == 1 and flags == [True]


def test_anh_cho_vlm_ve_lai_trang_khi_render(tmp_path):
    """page_image_for_vlm lấy ảnh nhúng gốc (JPEG) cho trang scan, nhưng render=True luôn vẽ lại thành PNG cả trang."""
    scan = tmp_path / "scan.pdf"
    scan_pdf(scan, 1)
    with pymupdf.open(scan) as doc:
        assert page_image_for_vlm(doc[0])[1] == "jpg"
        data, ext = page_image_for_vlm(doc[0], render=True)
        assert ext == "png" and png(data)


# --------------------------------------------------------------------------------------------------------------------
# Làm lại checkpoint cũ
# --------------------------------------------------------------------------------------------------------------------


def test_checkpoint_cu_khong_co_extract_version_bi_lam_lai_mot_lan(tmp_path):
    """Checkpoint do bản trích text cũ (chưa có extract_version) bị làm lại ở lần chạy sau, rồi không làm lại nữa; report đếm được bản cũ."""
    raw = tmp_path / "raw/vjol"
    text_pdf(raw / "2019/7_1.pdf", [vi_text(6, 1)])
    ocr = CountingOcr()
    assert vx.run_extract(raw, tmp_path, lambda: ocr) == 0
    ckpt = next((tmp_path / vx.TEXT_DIR).glob("*.json"))
    data = json.loads(ckpt.read_text(encoding="utf-8"))
    assert data["extract_version"] == vx.EXTRACT_VERSION
    old = {k: v for k, v in data.items() if k != "extract_version"}
    old["pages"] = [{"text": "chữ cũ", "method": "text"}]
    ckpt.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    assert vx.report(raw, tmp_path)["old_version"] == 1
    assert vx.run_extract(raw, tmp_path, lambda: ocr) == 0
    new = json.loads(ckpt.read_text(encoding="utf-8"))
    assert new["extract_version"] == vx.EXTRACT_VERSION and new["pages"][0]["text"] != "chữ cũ"
    assert vx.report(raw, tmp_path)["old_version"] == 0
    stamp = ckpt.stat().st_mtime_ns
    assert vx.run_extract(raw, tmp_path, lambda: ocr) == 0 and ckpt.stat().st_mtime_ns == stamp


def test_phien_ban_co_vocab_checkpoint_thieu_vocab_duoc_lam_lai_khi_co_file(tmp_path):
    """Chạy khi thiếu syllables.json: checkpoint không có +vocab, report cảnh báo; thêm file thì làm lại; sau đó mất file cũng không làm lại."""
    raw = tmp_path / "raw/vjol"
    text_pdf(raw / "2019/7_1.pdf", [vi_text(6, 1)])
    syl = tmp_path / vx.SYLLABLES_REL
    assert vx.run_extract(raw, tmp_path, lambda: None) == 0
    ckpt = next((tmp_path / vx.TEXT_DIR).glob("*.json"))
    assert json.loads(ckpt.read_text(encoding="utf-8"))["extract_version"] == vx.EXTRACT_VERSION
    rep = vx.report(raw, tmp_path)
    assert rep["without_vocab"] == 1 and rep["old_version"] == 0
    syl.parent.mkdir(parents=True)
    syl.write_text(json.dumps({"syllables": sorted(make_vocab(vi_text(6, 1)))}, ensure_ascii=False), encoding="utf-8")
    assert vx.run_extract(raw, tmp_path, lambda: None) == 0
    assert json.loads(ckpt.read_text(encoding="utf-8"))["extract_version"] == vx.EXTRACT_VERSION + vx.VOCAB_TAG
    assert vx.report(raw, tmp_path)["without_vocab"] == 0
    stamp = ckpt.stat().st_mtime_ns
    syl.unlink()
    assert vx.run_extract(raw, tmp_path, lambda: None) == 0 and ckpt.stat().st_mtime_ns == stamp
    assert vx.accepted_versions(None) == {"2", "2+vocab"} and vx.accepted_versions(frozenset({"a"})) == {"2+vocab"}


def test_dung_vocab_trong_trich_text_bat_ho_windows_1258(tmp_path):
    """E2E trích text: có syllables.json thì trang họ Windows-1258 đi OCR (signals có unknown); thiếu file thì trang đó vẫn là text."""
    raw = tmp_path / "raw/vjol"
    mixed_pdf(raw / "2019/7_1.pdf", [WIN1258 * 4] + [vi_text(5, k) for k in range(1, 8)])
    vocab = sorted(base_vocab())
    ocr = ocr_engine([])
    assert vx.run_extract(raw, tmp_path, ocr) == 0
    ckpt = next((tmp_path / vx.TEXT_DIR).glob("*.json"))
    assert json.loads(ckpt.read_text(encoding="utf-8"))["pages"][0]["method"] == "text"
    (tmp_path / vx.SYLLABLES_REL).parent.mkdir(parents=True)
    (tmp_path / vx.SYLLABLES_REL).write_text(json.dumps({"syllables": vocab}, ensure_ascii=False), encoding="utf-8")
    assert vx.run_extract(raw, tmp_path, ocr) == 0
    page = json.loads(ckpt.read_text(encoding="utf-8"))["pages"][0]
    assert page["method"] == "ocr_vlm" and "unknown" in page["signals"]
    rep = vx.report(raw, tmp_path)
    assert rep["garbled_signals"].get("unknown") == 1 and rep["garbled_pages"] == {g.REASON_PAGE: 1}


# --------------------------------------------------------------------------------------------------------------------
# Luật chất lượng
# --------------------------------------------------------------------------------------------------------------------


def test_luat_chat_luong_garbled_font_la_loi_cung():
    """Văn bản chữ hỏng bị band D với mã garbled_font dù fastText/điểm khác bình thường; tiếng Việt đúng không bị; số đo cũ thiếu khoá garbled không lỗi."""
    prof = make_cfg().profile("vjol")
    bad = compute_metrics(" ".join([GARBLED] * 4))
    good = compute_metrics(vi_text(12, 5))
    assert bad["garbled"] > 0.08 and good["garbled"] < 0.02
    score, codes = score_metrics(bad, "vi", prof, 0.99, "{}")
    assert "garbled_font" in codes and "garbled_font" in HARD_REASONS and band_of(score) == "D"
    assert "garbled_font" not in score_metrics(good, "vi", prof, 0.99, "{}")[1]
    legacy = {k: v for k, v in bad.items() if k != "garbled"}
    assert "garbled_font" not in score_metrics(legacy, "vi", prof, 0.99, "{}")[1]
    assert compute_metrics("Baỷng ngửụứi")["garbled"] == 0.0  # văn bản ngắn không kết luận


# --------------------------------------------------------------------------------------------------------------------
# Script đo
# --------------------------------------------------------------------------------------------------------------------


def make_ckpts(root: Path) -> None:
    """Hai bài: 2018 có 1 trang hỏng / 10 trang chữ; 2019 có 4 hỏng / 10 trang chữ (bị nâng lên cấp bài) + 1 trang scan; một bài lỗi."""
    write_ckpt(root, "a" * 64, "raw/vjol/2018/1_1.pdf", [GARBLED * 2] + [GOOD_VI] * 9)
    write_ckpt(root, "b" * 64, "raw/vjol/2019/2_2.pdf", [GARBLED * 2] * 4 + [GOOD_VI] * 6)
    write_ckpt(root, "c" * 64, "raw/vjol/2019/3_3.pdf", [], error="hỏng")
    ckpt = json.loads(vx.checkpoint_path(root, "b" * 64).read_text(encoding="utf-8"))
    ckpt["pages"].append({"text": "Nội dung OCR", "method": "ocr_vlm", "tier": "t0", "status": "ok"})
    vx.checkpoint_path(root, "b" * 64).write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")


def test_do_chu_hong_tren_checkpoint_co_san(tmp_path):
    """check() đếm đúng trang / bài bị nghi, bài bị nâng lên cấp bài, phân bố theo năm, quét ngưỡng, và ghi đủ file kết quả (thiếu vocab)."""
    make_ckpts(tmp_path)
    out = gc.check(tmp_path)
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["docs"] == 2 and s["text_layer_pages"] == 20 and s["suspect_pages"] == 5 and s["vocab"] is False
    assert s["suspect_pages_by_signal"]["tone"] == 5 and "unknown" not in s["suspect_pages_by_signal"]
    assert s["docs_with_suspect_pages"] == 2 and s["docs_whole_upgraded"] == 1 and s["docs_to_ocr"] == 2
    assert s["pages_to_ocr_again"] == 1 + 10 and s["ocr_minutes_estimate"] > 0
    assert s["by_year"]["2018"]["suspect_pages"] == 1 and s["by_year"]["2019"]["docs_whole"] == 1
    assert sum(s["tone_histogram_pages"].values()) == 20
    grid = s["sweep"]["tone_x_doc_share"]
    hit = next(r for r in grid if r["tone"] == 0.08 and r["doc_share"] == 0.30)
    assert hit["pages_to_ocr"] == 11 and hit["docs_whole"] == 1
    strict = next(r for r in grid if r["tone"] == 0.08 and r["doc_share"] == 0.50)
    assert strict["docs_whole"] == 0 and strict["pages_to_ocr"] == 5
    tone = {r["threshold"]: r for r in s["sweep"]["by_signal"]["tone"]}
    assert tone[0.05]["pages_flagged"] == 5 and tone[0.05]["pages_only_this_signal"] == 5
    pages = list(csv.DictReader(open(out / "pages.csv", encoding="utf-8")))
    assert len(pages) == 5 and {r["year"] for r in pages} == {"2018", "2019"} and pages[0]["signals"] == "tone"
    docs = {r["sha256"][0]: r for r in csv.DictReader(open(out / "docs.csv", encoding="utf-8"))}
    assert docs["b"]["whole_doc"] == "True" and docs["a"]["whole_doc"] == "False"
    assert "Baỷng" in (out / "examples.txt").read_text(encoding="utf-8")
    assert "unknown: thiếu syllables.json" in (out / "new_signals.txt").read_text(encoding="utf-8")


def test_do_chu_hong_ba_tin_hieu_moi_va_trang_can_xem(tmp_path):
    """
    Có syllables.json và ba họ chữ hỏng ít dấu thanh: mỗi tín hiệu mới bắt đúng trang của nó (và chỉ nó), new_signals.txt liệt kê trang chỉ
    tín hiệu đó bắt, blind_spots.txt liệt kê trang quá ngắn và trang không bị nghi nhiều âm tiết lạ nhất.
    """
    vocab = base_vocab()
    (tmp_path / "processed/ocr_bakeoff").mkdir(parents=True)
    (tmp_path / gc.SYLLABLES_REL).write_text(json.dumps({"syllables": sorted(vocab)}, ensure_ascii=False), encoding="utf-8")
    odd = "Xkcd zqwx plmn vrtk " * 3 + GOOD_VI * 2  # nhiều âm tiết lạ nhưng dưới ngưỡng: không bị nghi
    write_ckpt(tmp_path, "d" * 64, "raw/vjol/2020/9_9.pdf",
               [VNI * 2, SHIFTED * 3, WIN1258 * 3, *[GOOD_VI * 3] * 5, odd, "Ngắn quá"])
    out = gc.check(tmp_path)
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["vocab"] is True and s["suspect_pages"] == 3 and s["short_text_pages"] == 1
    assert s["suspect_pages_by_signal"]["foreign"] >= 1 and s["suspect_pages_by_signal"]["case"] >= 1
    assert s["suspect_pages_by_signal"]["unknown"] >= 1
    by = {r["threshold"]: r for r in s["sweep"]["by_signal"]["unknown"]}
    assert by[0.4]["pages_flagged"] >= 1 and by[0.25]["pages_flagged"] >= by[0.6]["pages_flagged"]
    new = (out / "new_signals.txt").read_text(encoding="utf-8")
    assert all(f"=== {sig}:" in new for sig in ("case", "foreign", "unknown")) and "sát dưới ngưỡng" in new
    blind = (out / "blind_spots.txt").read_text(encoding="utf-8")
    assert "Xkcd" in blind and "Ngắn quá" in blind


def test_do_chu_hong_chay_duoc_sau_khi_da_ocr_lai(tmp_path):
    """Sau khi chạy lại mẫu, trang đã OCR lại vì lỗi font vẫn được tính nhờ garbled_text (script đo dùng được cả trước lẫn sau)."""
    raw = tmp_path / "raw/vjol"
    mixed_pdf(raw / "2019/7_1.pdf", [GARBLED * 2] + [vi_text(6, k) for k in range(1, 8)])
    assert vx.run_extract(raw, tmp_path, ocr_engine([])) == 0
    s = json.loads((gc.check(tmp_path) / "summary.json").read_text(encoding="utf-8"))
    assert s["suspect_pages"] == 1 and s["text_layer_pages"] == 8


def test_dong_lenh_check_garbled(tmp_path):
    """E2E dòng lệnh: scripts/vjol/check_garbled.py chạy được trên checkpoint có sẵn và in đường dẫn kết quả."""
    make_ckpts(tmp_path)
    r = subprocess.run([sys.executable, str(REPO / "scripts/vjol/check_garbled.py"), "--data-root", str(tmp_path)],
                       capture_output=True, text=True, timeout=300, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / gc.CHECK_DIR / "summary.json").is_file() and "vjol_garbled_check" in r.stdout


def test_dong_lenh_trich_in_loi_font_va_checkpoint_cu(tmp_path):
    """--status của scripts/vjol/extract.py báo số bài lỗi font và số checkpoint cũ sẽ bị làm lại."""
    raw = tmp_path / "raw/vjol"
    mixed_pdf(raw / "2019/7_1.pdf", [GARBLED * 2] + [vi_text(6, k) for k in range(1, 8)])
    r = subprocess.run([sys.executable, str(REPO / "scripts/vjol/extract.py"), "--data-root", str(tmp_path), "--no-ocr"],
                       capture_output=True, text=True, timeout=300, check=False)
    assert r.returncode == 0 and "Lỗi font (J-10): 1 bài" in r.stdout, r.stdout + r.stderr
    ckpt = next((tmp_path / vx.TEXT_DIR).glob("*.json"))
    data = json.loads(ckpt.read_text(encoding="utf-8"))
    del data["extract_version"]
    ckpt.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([sys.executable, str(REPO / "scripts/vjol/extract.py"), "--data-root", str(tmp_path), "--status"],
                       capture_output=True, text=True, timeout=300, check=False)
    assert "1 checkpoint do bản trích text cũ" in r.stdout
