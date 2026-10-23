"""
Test các sửa của J-22 / J-23: ký tự font Symbol, mã chất lượng mới (private_use_chars, math_font_junk, lost_diacritics, reversed_text, odd_words,
reference_list), tín hiệu mất dấu của bộ dò chữ hỏng, làm sạch khuôn mẫu tạp chí (email, điện thoại, tiêu đề trang, tham khảo có hậu tố),
đổi tên trường meta, embed không rơi xuống CPU lặng lẽ và công cụ chẩn đoán. Không cần mạng, GPU hay mô hình. Các câu mẫu mô phỏng lỗi thật của vjol_10k_v2
(J-22), không phải đoạn trích nguyên văn.
"""

import json

import pytest

from tests.test_garbled import make_vocab
from tests.test_pipeline import make_cfg, vi_text
from tests.test_vjol import write_ckpt
from vi_corpus.common import garbled as g
from vi_corpus.pipeline.normalize import normalize_rows, normalize_text
from vi_corpus.pipeline.quality import (annotate_odd_words, annotate_quality, compute_metrics, odd_tokens, score_metrics,
                                        vocab_identity)
from vi_corpus.pipeline.symbol_font import SYMBOL_MAP, map_symbol_font
from vi_corpus.vjol import clean
from vi_corpus.vjol import diagnose as dg
from vi_corpus.vjol.records import load_metadata

GOOD_VI = vi_text(6, 3)
UNACCENTED = ("Nghien cuu cua nhom tac gia cho thay nhung nguoi dan duoc huong loi nhieu hon voi mot chinh sach hoac mot chuong trinh "
              "khong chi cua nha nuoc ma cung cua doanh nghiep voi hoat dong thuc te ") * 2


def metrics_of(text: str, vocab=None) -> dict:
    """Số đo quality của text (profile mặc định)."""
    return compute_metrics(text, vocab)


def codes_of(text: str, vocab=None, lang: str = "vi") -> list[str]:
    """Reason code của text với hồ sơ nguồn mặc định (min_words hạ xuống để câu mẫu ngắn không bị too_short)."""
    prof = make_cfg().profile("x")
    prof = type(prof)(min_words=1)
    return score_metrics(metrics_of(text, vocab), lang, prof)[1]


# ---- (4) font Symbol


def test_font_symbol_doi_ve_unicode_va_bo_manh_ghep():
    """U+F0xx của font Symbol thành ký tự Unicode đúng; gạch đầu dòng Wingdings thành '-'; mảnh ngoặc nhiều dòng bị bỏ; chữ khác giữ nguyên."""
    text, n = map_symbol_font("p  005; ;  mục;  x a")
    assert text == "p ≤ 0.05; α±β; - mục;  x (a)" and n == 9
    assert map_symbol_font("Việt Nam") == ("Việt Nam", 0)
    assert len(SYMBOL_MAP) >= 190


def test_normalize_doi_font_symbol_va_dem():
    """normalize_text và normalize_rows đổi ký tự Symbol (bản ghi thống kê symbol_chars), kể cả khi có code (thụt lề giữ)."""
    assert normalize_text("x 5 ") == "x≥ 5 λ"
    rows, st = normalize_rows([{"text": "ab"}, {"text": "không có"}])
    assert rows[0]["text"] == "a±b" and st["symbol_chars"] == 1 and st["normalizer_version"] == "3"


# ---- (5) rác font toán, ký tự vùng riêng; (2) mất dấu; (1) đảo ngược


def test_rac_font_toan_va_ky_tu_vung_rieng_la_ma_cung():
    """>= 5 chữ Tamil / Ả Rập / Thái hoặc >= 5 ký tự vùng riêng còn sót thì mã cứng; ít hơn thì không; chữ Hán không bao giờ tính."""
    junk = GOOD_VI + " " + "ரமவக்ஷ"
    assert "math_font_junk" in codes_of(junk) and "math_font_junk" not in codes_of(GOOD_VI + " ரம")
    assert "math_font_junk" not in codes_of(GOOD_VI + " 漢字漢字漢字漢字漢字漢字")
    assert "private_use_chars" in codes_of(GOOD_VI + " ") and "private_use_chars" not in codes_of(GOOD_VI + " ")
    assert {"math_font_junk", "private_use_chars"} <= set(__import__("vi_corpus.pipeline.quality", fromlist=["x"]).HARD_REASONS)


def test_mat_dau_tieng_viet_bi_bat_con_chu_co_dau_dung_thi_khong():
    """Văn bản mất dấu bị lost_diacritics; chữ có dấu đúng và văn bản ngắn thì không; trang mất dấu bị nghi bởi tín hiệu nodiac."""
    assert "lost_diacritics" in codes_of(UNACCENTED)
    assert "lost_diacritics" not in codes_of(GOOD_VI * 3)
    assert "lost_diacritics" not in codes_of("Nghien cuu cua nhom tac gia")  # dưới 50 từ Latinh
    assert g.page_signals(UNACCENTED) == ["nodiac"] and g.page_signals(GOOD_VI * 3) == []
    assert "nodiac" in g.SIGNALS


def test_chu_dao_nguoc_can_tap_am_tiet():
    """Văn bản đảo từng từ bị reversed_text khi có tập âm tiết; văn bản thường và English thì không; thiếu tập âm tiết thì bỏ."""
    vocab = make_vocab(GOOD_VI)
    flipped = " ".join(w[::-1] for w in (GOOD_VI * 2).split())
    assert "reversed_text" in codes_of(flipped, vocab)
    assert "reversed_text" not in codes_of(GOOD_VI * 2, vocab)
    assert "reversed_text" not in codes_of(flipped)  # không vocab: bỏ tín hiệu
    english = "The committee reviewed the proposal and approved the budget for the following year after long discussion " * 3
    assert "reversed_text" not in codes_of(english, vocab)
    gain, rev, n = g.reversed_gain(flipped, vocab)
    assert gain > 0.5 and rev > 0.8 and n > 30


# ---- (3) C: từ lạ


def test_tu_la_cap_bai_cho_rieng_vjol_va_mien_viet_tat_ten_khoa_hoc():
    """Từ lạ tính theo số bài trong lần chạy: từ lặp ở >= 3 bài là thuật ngữ; viết tắt và tên khoa học được miễn; nguồn không bật odd_words không bị đụng."""
    base = vi_text(8, 5)
    vocab = make_vocab(base)
    cfg = make_cfg(syllables=None)
    cfg_v = type(cfg)(**{**cfg.__dict__, "syllables": None})
    toks, n = odd_tokens(base + " ADN Escherichia coli blorf", vocab)
    assert toks == ["blorf"] and n > 50

    def row(i, key, text, parent):
        """Một đoạn giả đã qua language."""
        return {"doc_id": f"{key}:{i}#c0", "parent_doc_id": parent, "source_key": key, "text": text, "language": "vi",
                "lang_score": 1.0, "lang_mix": "{}"}

    garbage = " ".join(f"xqz{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(60))  # 60 từ lạ, mỗi từ chỉ ở một bài
    shared = "mekong"  # từ lạ lặp ở 3 bài: thuật ngữ thật
    rows = [row(1, "vjol", base + " " + garbage, "A"), row(2, "vjol", base + f" {shared}", "B"), row(3, "vjol", base + f" {shared}", "C"),
            row(4, "vjol", base + f" {shared}", "D"), row(5, "sea_pile_v2", base + " " + garbage, "E")]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        syl = Path(d) / "syllables.json"
        syl.write_text(json.dumps({"syllables": sorted(vocab)}), encoding="utf-8")
        cfg2 = make_cfg(syllables=str(syl))
        annotate_quality(rows, cfg2)
        stats = annotate_odd_words(rows, cfg2)
        assert vocab_identity(cfg2) == [str(syl), syl.stat().st_size]
    assert stats == {"checked": 4, "flagged": 1, "skipped": None, "unit": "doc", "journals": 0}
    assert "odd_words" in rows[0]["reason_codes"] and rows[0]["quality_band"] == "D"
    assert all("odd_words" not in r["reason_codes"] for r in rows[1:])  # "mekong" lặp ở 3 bài; nguồn web không bật
    assert json.loads(rows[1]["quality_metrics"])["odd_word"] < 0.15
    assert annotate_odd_words(rows, make_cfg())["skipped"].startswith("thiếu tập âm tiết")
    assert cfg_v.syllables is None


def test_danh_muc_tham_khao_con_sot_duoc_gan_nhan_khong_tru_diem():
    """Đoạn phần lớn là mục tài liệu tham khảo (năm + tên viết tắt / nhà xuất bản / tạp chí) có nhãn reference_list; văn xuôi thì không."""
    refs = "\n".join(f"Nguyen V.A., Tran T.B. ({2000 + i}). Kinh tế vĩ mô. NXB Chính trị quốc gia, tr. {i + 10}." for i in range(8))
    codes = codes_of(refs)
    assert "reference_list" in codes
    prof = type(make_cfg().profile("x"))(min_words=1)
    m = metrics_of(refs)
    assert score_metrics(m, "vi", prof)[0] == score_metrics({**m, "ref_line": 0.0}, "vi", prof)[0]  # chỉ gắn nhãn, không trừ điểm
    assert "reference_list" not in codes_of(GOOD_VI)


# ---- (6) khuôn mẫu và PII


def test_tieu_de_tham_khao_co_hau_to_hoac_dinh_lien_muc_1():
    """Tiêu đề 'Tài liệu tham khảo chính', 'Tài liệu tham khảo 1. ...' được nhận; câu văn nhắc tới tham khảo thì không; strip_references bỏ phần đó."""
    for ok in ("tài liệu tham khảo chính", "tài liệu tham khảo 1. nguyễn văn a (2010)", "references", "tài liệu tham khảo tiếng việt"):
        assert clean.is_ref_heading(ok), ok
    for bad in ("tham khảo 2 nguồn khác nhau", "ta tham khảo tài liệu", "tài liệu tham khảo của bài này rất nhiều"):
        assert not clean.is_ref_heading(bad), bad
    body = "\n".join(f"Dòng nội dung số {i} của bài." for i in range(10))
    page = body + "\nTÀI LIỆU THAM KHẢO CHÍNH\n1. Nguyễn Văn A (2010)\n2. Trần B (2012)"
    out, dropped = clean.strip_references([page])
    assert dropped == 3 and "Nguyễn Văn A" not in out[0] and "Dòng nội dung số 9" in out[0]
    stuck, d2 = clean.strip_references([body + "\nTài liệu tham khảo 1. Nguyễn Văn A (2010)\n2. Trần B"])
    assert d2 == 2 and "Trần" not in stuck[0]


def test_khuon_mau_tap_chi_va_che_pii():
    """Bỏ 'Ngày nhận bài', khối tác giả có email, tiêu đề trang tạp chí (cả dính giữa dòng); che email / điện thoại còn lại; câu văn dài giữ nguyên."""
    pages = ["Nguyễn Văn An\nĐại học Quốc gia\nEmail: an@vnu.edu.vn\nNgày nhận bài: 01/02/2015\nNgày duyệt đăng: 03/04/2015\n"
             "Tóm tắt\nNghiên cứu này liên hệ qua thư an@vnu.edu.vn hoặc số 0912 345 678 để biết thêm chi tiết về mẫu khảo sát.\n"
             "Tạp chí Nghiên cứu Tôn giáo, Số 3 - 2015\nNội dung tiếp theo của bài Tạp chí Nghiên cứu Tôn giáo, Số 3 - 2015 vẫn liền mạch.",
             "Số điện thoại: +84 28 3822 1234\nHết bài."]
    out, st = clean.strip_boilerplate(pages, "Tạp chí Nghiên cứu Tôn giáo")
    text = "\n".join(out)
    assert "Ngày nhận" not in text and "Email:" not in text and "Số điện thoại" not in text
    assert "<EMAIL>" in text and "<PHONE>" in text and "an@vnu" not in text and "0912" not in text
    assert "Tạp chí Nghiên cứu Tôn giáo" not in text and "Nội dung tiếp theo của bài vẫn liền mạch." in text
    assert st["boilerplate_lines"] == 4 and st["author_block_lines"] == 0 and st["journal_headers"] == 2
    assert st["emails_masked"] == 1 and st["phones_masked"] == 1
    assert clean.journal_header_pattern(None) is None and clean.journal_header_pattern("Ab") is None


def test_clean_paper_pages_nhan_ten_tap_chi_va_meta_doi_ten_language(tmp_path):
    """clean_paper_pages nhận tên tạp chí; metadata ghi nhãn ngôn ngữ của tạp chí dưới khoá journal_language_raw, không còn khoá language."""
    out, st = clean.clean_paper_pages(["Hà Nội 2015 Tạp chí Kinh tế, Số 2 - 2015\nĐoạn văn."], "Tạp chí Kinh tế")
    assert st["journal_headers"] == 1 and "Tạp chí" not in out[0]
    meta = tmp_path / "m.jsonl"
    meta.write_text(json.dumps({"article_id": 7, "journal_title": "Tạp chí A", "language": "en", "year": 2019, "galleys": [{"galley_id": 1}]}) + "\n",
                    encoding="utf-8")
    md = load_metadata(meta, {"7_1"})["7_1"]
    assert md["journal_language_raw"] == "en" and "language" not in md


def test_iter_records_dung_ten_tap_chi_va_meta_journal_language_raw(tmp_path):
    """iter_records bỏ tiêu đề trang tạp chí theo metadata, che email và ghi journal_language_raw vào meta."""
    from vi_corpus.common.registry import SOURCES
    from vi_corpus.vjol.records import iter_records

    raw = tmp_path / "raw/vjol"
    (raw / "metadata").mkdir(parents=True)
    (raw / "metadata/vjol_merged.jsonl").write_text(json.dumps({"article_id": 9, "journal_title": "Tạp chí Thử Nghiệm", "language": "en", "year": 2019,
                                                              "title": "T", "galleys": [{"galley_id": 1}]}) + "\n", encoding="utf-8")
    pages = [vi_text(4, 1) + "\nTạp chí Thử Nghiệm, Số 1 - 2019\nChi tiết xin liên hệ qua thư ai@x.vn của nhóm tác giả để biết thêm về bộ số liệu đã dùng.\n" + vi_text(3, 2), vi_text(4, 3)]
    write_ckpt(tmp_path, "a" * 64, "raw/vjol/2019/9_1.pdf", pages)
    from vi_corpus.common.registry import SourceSpec

    spec = SOURCES["vjol"]
    recs = list(iter_records(spec, tmp_path, with_meta=True))
    assert len(recs) == 1 and "Tạp chí Thử Nghiệm" not in recs[0]["text"] and "ai@x.vn" not in recs[0]["text"]
    meta = json.loads(recs[0]["meta"])
    assert meta["journal_language_raw"] == "en" and "language" not in meta and meta["journal_headers"] == 1 and meta["emails_masked"] == 1
    assert isinstance(spec, SourceSpec)


# ---- (8) embed


def test_embed_cuda_khong_co_gpu_bao_loi_thay_vi_chay_cpu(monkeypatch):
    """HfEmbedder(device='cuda') khi torch không thấy CUDA phải ném RuntimeError ngay (chưa nạp mô hình); device lạ bị từ chối."""
    torch = pytest.importorskip("torch")
    from vi_corpus.pipeline.embed import HfEmbedder

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU"):
        HfEmbedder("khong-ton-tai/mo-hinh", "cuda")
    with pytest.raises(ValueError):
        HfEmbedder("khong-ton-tai/mo-hinh", "tpu")


# ---- chẩn đoán


def test_diagnose_inspect_in_nguon_chu_dao_nguoc(tmp_path):
    """inspect_articles in method, tín hiệu và độ lợi đảo của từng trang theo article_id và tiền tố sha; bài lạ được ghi rõ."""
    vocab = make_vocab(GOOD_VI)
    syl = tmp_path / "processed/ocr_bakeoff/syllables.json"
    syl.parent.mkdir(parents=True)
    syl.write_text(json.dumps({"syllables": sorted(vocab)}), encoding="utf-8")
    flipped = " ".join(w[::-1] for w in (GOOD_VI * 2).split())
    write_ckpt(tmp_path, "b" * 64, "raw/vjol/2019/55_1.pdf", [GOOD_VI * 2, flipped])
    report = dg.inspect_articles(tmp_path, ["55", "bbbbbb", "999"])
    assert "tập âm tiết: có" in report and report.count("=== bài 55") == 2
    assert "trang 1: method=text" in report and "độ lợi đảo=-" in report and "độ lợi đảo=+" in report
    assert "=== 999: không thấy checkpoint" in report
    import hashlib
    by_doc_id = hashlib.sha1(("b" * 64).encode()).hexdigest()[:10]  # đoạn giữa của doc_id "vjol:<băm>:0"
    assert "=== bài 55" in dg.inspect_articles(tmp_path, [by_doc_id])


# ---- J-24: trang scan bị xoay


class RotatingStubCascade:
    """
    Chuỗi engine giả cho ảnh 60x90 đen có khối trắng ở góc trên-trái: ảnh đúng chiều thì VLM bị bỏ và baseline đọc ra chữ ngược rác (status ok);
    ảnh xoay 180 độ (khối ở góc dưới-phải) thì VLM đọc tốt; mọi ảnh khác (xoay 90 / 270) bị loại.
    """

    def __init__(self, good: str, junk: str) -> None:
        """good: văn bản đọc đúng; junk: chữ rác của baseline."""
        from vi_corpus.ocr_bakeoff import cascade as cas

        self.tiers = [cas.Tier("vlm", {"type": "openai"}, factory=lambda: None), cas.Tier("baseline", {"type": "baseline"}, factory=lambda: None, serial=True)]
        self.good, self.junk = good, junk

    def page(self, image: bytes, ext: str = "jpg") -> dict:
        """Trả kết quả theo hướng của ảnh (xem docstring lớp)."""
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(image)).convert("L")
        if img.size == (60, 90) and img.getpixel((3, 3)) > 128:
            return {"text": self.junk, "status": "ok", "tier": "baseline", "finish_reason": None, "reasons": []}
        if img.size == (60, 90) and img.getpixel((56, 86)) > 128:
            return {"text": self.good, "status": "ok", "tier": "vlm", "finish_reason": "stop", "reasons": []}
        return {"text": "", "status": "flagged", "tier": "baseline", "finish_reason": None, "reasons": ["loop"], "rejected_text": "xx"}


def test_trang_scan_xoay_180_duoc_xoay_lai_va_chon_ban_doc_duoc():
    """Trang baseline đọc ra chữ ngược rác được xoay thử; bản xoay 180 độ đọc tốt hơn rõ rệt thì thắng và ghi rotation=180; không có vocab thì không xoay."""
    import io

    import pymupdf
    from PIL import Image

    from vi_corpus.common.ocr_cascade import CascadePageOcr

    img = Image.new("RGB", (60, 90), (0, 0, 0))
    img.paste((255, 255, 255), (0, 0, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    doc = pymupdf.open()
    doc.new_page(width=60, height=90).insert_image(pymupdf.Rect(0, 0, 60, 90), stream=buf.getvalue())
    good = vi_text(6, 11)
    junk = " ".join(w[::-1] for w in good.split())
    out = CascadePageOcr(RotatingStubCascade(good, junk), concurrency=2, vocab=make_vocab(good)).ocr_pages([doc[0]])
    assert out[0]["rotation"] == 180 and out[0]["tier"] == "vlm" and out[0]["text"] == good
    out2 = CascadePageOcr(RotatingStubCascade(good, junk), concurrency=2).ocr_pages([doc[0]])
    assert "rotation" not in out2[0] and out2[0]["tier"] == "baseline" and out2[0]["text"] == junk
    # trang ngắn (không đủ từ để so) giữ nguyên bản gốc dù có xoay được
    short = CascadePageOcr(RotatingStubCascade("Xin chào", "oàhc niX"), concurrency=2, vocab=make_vocab(good)).ocr_pages([doc[0]])
    assert "rotation" not in short[0]
