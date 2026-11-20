"""
Test công cụ kiểm chứng trang bị bộ dò chữ hỏng gắn cờ bằng OCR (GT-06, vi_corpus/giao_trinh/verify.py): chia tầng, chọn mẫu, thống kê.
Không cần GPU hay mô hình.
"""

import json

from vi_corpus.giao_trinh.verify import STRATA, collect_pages, pick_sample, stratum_of, summarize, vi_density


def test_vi_density_phan_biet_tieng_viet_voi_tieng_anh_va_code():
    """Tiếng Việt có dấu cho mật độ cao; tiếng Anh, code, văn bản rỗng cho 0."""
    assert vi_density("Phân tích kinh tế vĩ mô cho thấy lạm phát tăng") > 0.2
    assert vi_density("int add_word(Dictionary *dict, const char *key)") == 0.0
    assert vi_density("") == 0.0 and vi_density("123 +- 45") == 0.0


def test_stratum_of_chia_tang_theo_nhan_cu_co_gan_co_va_mat_do_chu_viet():
    """Trang gắn cờ chia theo nhãn lần trước (tcvn3 / text) và mật độ chữ Việt; trang không gắn cờ chỉ lấy loại text làm đối chứng."""
    vi, en = "Phân tích kinh tế vĩ mô", "Web Design and Programming"
    assert stratum_of("tcvn3", True, vi) == "flag_tcvn3"
    assert stratum_of("text", True, vi) == "flag_text_vi" and stratum_of("text", True, en) == "flag_text_en"
    assert stratum_of("text", False, vi) == "ctrl_vi" and stratum_of("text", False, en) == "ctrl_en"
    assert stratum_of("tcvn3", False, vi) is None and stratum_of("empty", True, vi) is None


def _ckpt(path, pages, fmt="pdf"):
    """Ghi một checkpoint giả với các trang (method, reason, text)."""
    path.write_text(json.dumps({"format": fmt, "paths": ["raw/giao_trinh/a/x.pdf"],
                                "pages": [{"method": m, "reason": r, "text": t} for m, r, t in pages]}, ensure_ascii=False), encoding="utf-8")


def test_collect_pages_ghep_hai_lan_trich_va_bo_checkpoint_lech(tmp_path):
    """Trang gắn cờ lấy nhãn cũ từ lần trước; trang không gắn cờ làm đối chứng; checkpoint lệch số trang hoặc không phải PDF bị bỏ."""
    new, old = tmp_path / "new", tmp_path / "old"
    new.mkdir(), old.mkdir()
    vi = "Phân tích kinh tế vĩ mô cho thấy lạm phát"
    _ckpt(new / "a.json", [("needs_ocr", "garbled_text_layer", "Baỷng hieọu ủoọ"), ("text", None, vi), ("pptx", None, "x")])
    _ckpt(old / "a.json", [("tcvn3", None, ""), ("text", None, vi), ("pptx", None, "x")])
    _ckpt(new / "b.json", [("text", None, vi)])  # lệch số trang với bản trước
    _ckpt(old / "b.json", [("text", None, vi), ("text", None, vi)])
    _ckpt(new / "c.json", [("text", None, vi)], fmt="pptx")
    _ckpt(old / "c.json", [("text", None, vi)], fmt="pptx")
    rows = list(collect_pages(new, old))
    assert [(r["sha256"], r["page"], r["stratum"]) for r in rows] == [("a", 0, "flag_tcvn3"), ("a", 1, "ctrl_vi")]
    assert rows[0]["path"] == "raw/giao_trinh/a/x.pdf"


def test_pick_sample_on_dinh_va_gioi_han_moi_tang():
    """Mỗi tầng tối đa N trang, cùng hạt giống cho cùng mẫu, khác hạt giống thì khác."""
    rows = [{"sha256": f"s{i}", "page": i, "stratum": STRATA[i % len(STRATA)], "text": "", "path": "p", "density": 0} for i in range(100)]
    a = pick_sample(rows, 5, seed=1)
    assert len(a) == 5 * len(STRATA) and all(sum(r["stratum"] == s for r in a) == 5 for s in STRATA)
    assert a == pick_sample(rows, 5, seed=1) and a != pick_sample(rows, 5, seed=2)


def test_summarize_chi_tinh_tren_trang_ocr_ra_chu():
    """Thống kê theo tầng chỉ tính trang OCR ra chữ; tầng không có OCR hợp lệ thì chỉ có n và ocr_ok."""
    res = [{"stratum": "flag_text_en", "status": "ok", "overlap": o} for o in (0.9, 0.85, 0.2)] + [{"stratum": "flag_text_en", "status": "blank", "overlap": None}]
    s = summarize(res)["flag_text_en"]
    assert s["n"] == 4 and s["ocr_ok"] == 3 and s["median"] == 0.85
    assert abs(s["share_ge_0.8"] - 2 / 3) < 0.01 and abs(s["share_le_0.4"] - 1 / 3) < 0.01
    assert summarize(res)["ctrl_vi"] == {"n": 0, "ocr_ok": 0}
