"""
Test các sửa đổi theo rà run 10.000 mẫu (docs/DECISION_LOG.md, F-01, F-03, F-04, F-07): xoá dòng lặp không đụng hội
thoại SFT và dòng dẫn ":", dup_line bỏ dòng không có chữ, số đo chất lượng trên văn xuôi (đã bỏ công thức / code), nhận
diện ngôn ngữ SFT theo lượt người dùng, phiên bản code trong vân tay ingest. Không cần mạng.
"""

import json
from dataclasses import replace

from tests.test_finalize_parsers import _first_changed, _web_root, prose
from tests.test_pipeline import SENT, SOURCES, make_cfg
from vi_corpus.pipeline import ingest as ingest_mod
from vi_corpus.pipeline.language import annotate_language, detect_row, language_stats
from vi_corpus.pipeline.lines import clean_lines_in_doc, clean_rows, is_lead_in
from vi_corpus.pipeline.quality import (
    HARD_REASONS,
    band_of,
    compute_metrics,
    score_metrics,
)
from vi_corpus.pipeline.runner import fingerprints
from vi_corpus.pipeline.spans import math_share, strip_math_code

SFT = make_cfg().profile("sea_instruct_2602")
WEB = make_cfg().profile("sea_pile_v2")

# Bài toán LaTeX kiểu SEA-Instruct: nhiều khối $$ ... $$, dòng "$$" đứng riêng lặp lại.
MATH_ANSWER = "\n".join(
    ["gpt: Ta có:", "$$", r"x^2 + 2x + 1 = 0 \Rightarrow (x + 1)^2 = 0", "$$", "Suy ra:", "$$", r"x = -1", "$$",
     "Kiểm tra lại:", "$$", r"(-1)^2 + 2 \cdot (-1) + 1 = 1 - 2 + 1 = 0", "$$", "Vậy:", "$$", r"x = -1", "$$"])
MATH_ROW_TEXT = "human: Giải phương trình $x^2 + 2x + 1 = 0$.\n" + MATH_ANSWER


def _conv(*turns: tuple[str, str]) -> dict:
    """Bản ghi hội thoại SEA-Instruct giả từ các cặp (vai, nội dung)."""
    ts = [{"role": r, "content": c} for r, c in turns]
    return {"doc_id": "c", "source_key": "sea_instruct_2602", "text": "\n".join(f"{r}: {c}" for r, c in turns),
            "meta": json.dumps({"turns": ts}, ensure_ascii=False)}


class LidByScript:
    """Bộ nhận diện giả: chữ Hán -> zh, chữ Kirin -> ru, có từ tiếng Anh thường gặp -> en, còn lại vi."""

    EN = frozenset({"the", "and", "of", "to", "is", "write", "poem", "def", "return", "frac"})

    def predict(self, segments):
        """(nhãn, độ tin cậy) cho từng đoạn."""
        out = []
        for s in segments:
            words = set(s.lower().split())
            if any("一" <= c <= "鿿" for c in s):
                out.append(("zh", 0.99))
            elif any("Ѐ" <= c <= "ӿ" for c in s):
                out.append(("ru", 0.99))
            elif len(words & self.EN) >= 2:
                out.append(("en", 0.95))
            else:
                out.append(("vi", 0.97))
        return out


# ---------------------------------------------------------------- spans (F-03)

def test_strip_math_code_bo_cong_thuc_va_code_giu_tien():
    """Bỏ ``` ```, $$ $$, \\[ \\], \\( \\) và $...$ có ký hiệu toán; giữ "$5", dấu không cân, chữ hai bên không dính."""
    text = "Giá $5 và $10.\nTa có $x^2$ và \\(a_1\\).\n$$\n\\frac{1}{2}\n$$\n```python\nprint(1)\n```\nHết \\[ y \\]."
    out = strip_math_code(text)
    assert "$5 và $10" in out and "x^2" not in out and "a_1" not in out and "frac" not in out
    assert "print" not in out and "Hết" in out and " y " not in out
    assert strip_math_code("Mở $$ không đóng") == "Mở $$ không đóng"
    assert strip_math_code("a$x=1$b") == "a b"
    assert math_share("abc", "abc") == 0.0 and math_share("", "") == 0.0
    assert 0.5 < math_share("ab $$xxxxxxxxxx$$", strip_math_code("ab $$xxxxxxxxxx$$")) < 1


# ---------------------------------------------------------------- lines (F-01)

def test_sft_khong_qua_muc_a_va_top_removed_theo_nguon():
    """F-01 (1), (4): hội thoại SFT giữ nguyên mọi dòng "$$" / nhãn lặp; thống kê dòng bị xoá tách theo nguồn."""
    sft = {"doc_id": "s", "source_key": "sea_instruct_2602", "text": MATH_ROW_TEXT + "\n12\n12"}
    web = {"doc_id": "w", "source_key": "sea_pile_v2",
           "text": "\n".join(["Xem thêm", SENT[0], "Xem thêm", SENT[1], "Xem thêm", "- 3 -"])}
    rows, st = clean_rows([sft, web], make_cfg())
    assert rows[0]["text"] == MATH_ROW_TEXT + "\n12\n12" and rows[0]["lines_removed"] == 0
    assert st["per_source"]["sea_instruct_2602"]["in_doc_off"] is True
    assert st["per_source"]["sea_instruct_2602"]["top_removed"] == []
    assert st["per_source"]["sea_pile_v2"]["top_removed"] == [["Xem thêm", 2], ["- 3 -", 1]]
    assert "top_removed" not in st


def test_dong_dan_ket_thuc_hai_cham_khong_bi_xoa():
    """F-01 (2): dòng dẫn kết thúc ":" (kể cả "**Lưu ý:**") lặp >= 3 lần vẫn giữ; dòng ngắn thường lặp vẫn bị xoá."""
    lead = "Sau khi điều chỉnh địa giới hành chính:"
    text = "\n".join([lead, SENT[0], "**Lưu ý:**", lead, SENT[1], "**Lưu ý:**", lead, SENT[2], "**Lưu ý:**",
                      "Đọc tiếp", SENT[3], "Đọc tiếp", SENT[4], "Đọc tiếp"])
    out, removed = clean_lines_in_doc(text)
    assert out.count(lead) == 3 and out.count("**Lưu ý:**") == 3
    assert removed == ["Đọc tiếp", "Đọc tiếp"]
    assert is_lead_in("Ưu điểm: ") and is_lead_in("__Mô tả:__") and not is_lead_in("Giá: 5 đồng")


def test_dup_line_bo_dong_khong_co_chu_cai():
    """F-01 (3): các dòng "$$" / "---" / "}" giống hệt nhau không làm tăng dup_line."""
    m = compute_metrics(MATH_ANSWER)
    assert m["dup_line"] < 0.25
    loop = "\n".join(["Tôi xin lỗi vì sự bất tiện này."] * 8 + ["$$"] * 8)
    assert compute_metrics(loop)["dup_line"] > 0.5  # dòng chữ lặp vẫn bị bắt


# ---------------------------------------------------------------- quality (F-03)

def test_bai_toan_latex_khong_bi_tru_diem_hinh_dang():
    """F-03: bài toán LaTeX (SFT) không còn repetitive / symbol_heavy / odd_word_length / duplicate_lines, band A."""
    m = compute_metrics(MATH_ROW_TEXT)
    assert m["math_share"] > 0.3 and m["prose_words"] < 20
    score, codes = score_metrics(m, "vi", SFT, 0.95, json.dumps({"vi": 1.0}))
    assert not {"repetitive", "symbol_heavy", "odd_word_length", "duplicate_lines", "repeated_ngrams",
                "low_alpha"} & set(codes)
    assert band_of(score) == "A"


def test_mien_tru_khong_ap_cho_van_ban_ngan_day_ky_hieu_rac():
    """F-03 (3): văn bản ngắn nhiều ký hiệu nhưng không phải công thức (math_share thấp) vẫn bị symbol_heavy."""
    junk = "!!! ### @@@ %%% ^^^ &&& *** ((( ))) +++ === --- ~~~ ``` ;;; ::: ,,, ... ??? <<< >>> aa"
    m = compute_metrics(junk)
    assert m["math_share"] < 0.3
    _, codes = score_metrics(m, "vi", SFT, 0.9, "{}")
    assert "symbol_heavy" in codes


def test_vong_lap_trong_van_xuoi_va_trong_cong_thuc_van_xuong_d():
    """F-03 (4), (6): vòng lặp trong văn xuôi và vòng lặp bên trong công thức đều xuống band D (repeated_ngrams)."""
    loop_prose = "human: Kể chuyện.\ngpt: " + "tôi đi học về nhà " * 60
    score, codes = score_metrics(compute_metrics(loop_prose), "vi", SFT, 0.95, "{}")
    assert "repeated_ngrams" in codes and band_of(score) == "D"
    loop_math = "human: Tính.\ngpt: $$" + " a + b = c" * 80 + " $$"
    m = compute_metrics(loop_math)
    assert m["math_share"] > 0.3 and m["rep_trigram_raw"] >= 0.8
    score, codes = score_metrics(m, "vi", SFT, 0.95, "{}")
    assert "repeated_ngrams" in codes and band_of(score) == "D"


def test_van_ban_khong_cong_thuc_giu_so_do_cu():
    """F-03 (5): văn bản không có công thức thì số đo hình dạng như trước (prose = text, math_share = 0)."""
    text = prose(10, 1)
    m = compute_metrics(text)
    assert m["math_share"] == 0 and m["prose_words"] == m["words"] and m["rep_trigram"] == m["rep_trigram_raw"]


# ---------------------------------------------------------------- language (F-07)

def test_sft_ngon_ngu_theo_luot_nguoi_dung():
    """F-07 (2), (3), (6): yêu cầu tiếng Việt, trả lời tiếng Anh / Nga / code / LaTeX -> vi; người dùng tiếng Trung -> zh."""
    lid = LidByScript()
    cases = {
        "tho": _conv(("system", "You are a helpful assistant and the best"),
                     ("human", "Sáng tác một bài thơ bằng tiếng Anh về mùa thu Hà Nội"),
                     ("gpt", "Write the poem of the autumn and the leaves " * 5)),
        "nga": _conv(("human", "Dịch câu sau sang tiếng Nga: Tôi yêu Hà Nội"), ("gpt", "Я люблю Ханой " * 5)),
        "code": _conv(("human", "Viết hàm Python tính giai thừa của một số nguyên dương"),
                      ("gpt", "```python\ndef f(n):\n    return 1 if n < 2 else n * f(n - 1)\n```")),
        "toan": {**_conv(("human", "Giải phương trình $x^2 + 2x + 1 = 0$."), ("gpt", MATH_ANSWER[5:]))},
    }
    for name, row in cases.items():
        lang, _, mix, answer = detect_row(row, lid)
        assert lang == "vi", name
        _, codes = score_metrics(compute_metrics(row["text"]), lang, SFT, 0.97, json.dumps(mix))
        assert "lang_not_allowed" not in codes, name
    _, _, mix, answer = detect_row(cases["tho"], lid)
    assert answer == "en" and mix["en"] > 0.5
    zh = _conv(("human", "请把下面这句话翻译成越南语，非常感谢你的耐心帮助和详细解释"), ("gpt", "Tôi rất vui được giúp bạn dịch câu này"))
    lang, _, _, _ = detect_row(zh, lid)
    _, codes = score_metrics(compute_metrics(zh["text"]), lang, SFT, 0.99, "{}")
    assert lang == "zh" and "lang_not_allowed" in codes


def test_luot_nguoi_dung_qua_it_chu_thi_lui_ve_moi_luot():
    """F-07 (2): lượt người dùng chỉ có công thức (< 20 chữ cái văn xuôi) thì nhận diện trên mọi lượt."""
    row = _conv(("human", "$$\\int_0^1 x\\,dx$$ ?"), ("gpt", "The answer is the half of the unit and the rest"))
    lang, _, _, answer = detect_row(row, LidByScript())
    assert lang == "en" and answer == "en"


def test_khong_con_chu_cai_thi_und_va_lang_unknown():
    """F-07 (4): bản ghi chỉ còn công thức -> language "und", quality gắn lang_unknown, không lang_not_allowed."""
    row = {"doc_id": "u", "source_key": "sea_pile_v2", "text": "$$ 1 + 2 = 3 $$\n\n$$ 4 + 5 = 9 $$", "meta": None}
    lang, score, mix, answer = detect_row(row, LidByScript())
    assert (lang, score, mix, answer) == ("und", 0.0, {}, None)
    _, codes = score_metrics(compute_metrics(row["text"]), lang, WEB, score, "{}")
    assert "lang_unknown" in codes and "lang_not_allowed" not in codes and "low_lang_score" not in codes
    assert "lang_unknown" not in HARD_REASONS


def test_language_stats_dem_cau_tra_loi_khong_phai_vi(monkeypatch):
    """F-07 (6): language_stats đếm hội thoại có câu trả lời không phải vi (và có mixed_language) theo nguồn."""
    rows = [_conv(("human", "Viết một bài thơ bằng tiếng Anh về biển"), ("gpt", "Write the poem of the sea and " * 5)),
            _conv(("human", "Thủ đô của Việt Nam là gì vậy bạn"), ("gpt", "Thủ đô của Việt Nam là Hà Nội."))]
    import vi_corpus.pipeline.language as lang_mod

    monkeypatch.setattr(lang_mod, "get_lid", lambda spec: LidByScript())
    annotate_language(rows, "fake")
    st = language_stats(rows)["sft_answers"]["sea_instruct_2602"]
    assert st == {"conversations": 2, "answer_not_vi": 1, "answer_not_vi_mixed": 1}
    assert rows[1]["lang_answer"] == "vi"


# ---------------------------------------------------------------- vân tay (F-04, F-01)

def test_van_tay_ingest_theo_phien_ban_code(tmp_path, monkeypatch):
    """F-04: đổi INGEST_VERSION thì vân tay ingest (và mọi stage sau) đổi."""
    root = _web_root(tmp_path)
    cfg = make_cfg(mix={"sea_pile_v2": 15}, embed_spec=None)
    ref = fingerprints(SOURCES, root, cfg)
    from vi_corpus.pipeline import runner

    monkeypatch.setattr(runner, "INGEST_VERSION", ingest_mod.INGEST_VERSION + "x")
    assert _first_changed(ref, fingerprints(SOURCES, root, cfg)) == "ingest"


def test_van_tay_prepare_theo_in_doc_line_clean(tmp_path):
    """F-01: in_doc_line_clean nằm trong vân tay prepare."""
    root = _web_root(tmp_path)
    base = make_cfg(mix={"sea_pile_v2": 15}, embed_spec=None)
    profiles = dict(base.profiles)
    profiles["sea_pile_v2"] = replace(profiles["sea_pile_v2"], in_doc_line_clean=False)
    other = make_cfg(mix={"sea_pile_v2": 15}, embed_spec=None, profiles=profiles)
    assert _first_changed(fingerprints(SOURCES, root, base), fingerprints(SOURCES, root, other)) == "prepare"
