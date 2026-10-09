"""
Test stage prepare (đếm token D-01, xoá dòng lặp D-04, cắt đoạn theo token D-10), stage language (theo đoạn, lang_mix,
D-02) và các mã chất lượng mới (low_lang_score, mixed_language, low_diacritic). Không cần mạng: tokenizer "words" hoặc
tokenizer.json tự dựng; fastText thật chỉ chạy khi đã có lid.176.bin trong VI_CORPUS_MODEL_DIR.
"""

import json
import os
from pathlib import Path

import pytest

from tests.test_pipeline import SENT, WORDS, make_cfg, vi_text
from vi_corpus.pipeline.chunk import chunk_rows, chunk_text, is_heading
from vi_corpus.pipeline.language import detect_mix, lid_text, segments_of
from vi_corpus.pipeline.lines import (
    clean_lines_cross_doc,
    clean_lines_in_doc,
    clean_rows,
)
from vi_corpus.pipeline.quality import compute_metrics, score_metrics
from vi_corpus.pipeline.tokens import TokenCounter


def _wordlevel_tokenizer(tmp_path: Path) -> str:
    """Dựng tokenizer.json WordLevel nhỏ (không cần mạng) để thử nhánh "hf:" của TokenCounter."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate("một hai ba bốn năm sáu".split())}}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return f"hf:{path}"


def test_token_counter_hf_va_words(tmp_path):
    """Nhánh hf: đếm theo lô và cắt cứng tại offset token; nhánh words đếm số từ; spec lạ báo lỗi."""
    counter = TokenCounter(_wordlevel_tokenizer(tmp_path))
    assert counter.count_batch(["một hai ba", "", "bốn, năm"]) == [3, 0, 3]  # dấu phẩy là một token
    assert counter.hard_split("một hai ba bốn năm", 2) == ["một hai", "ba bốn", "năm"]
    assert WORDS.count("Hà Nội  mùa thu") == 4 and WORDS.hard_split("a b c", 2) == ["a b", "c"]
    with pytest.raises(ValueError):
        TokenCounter("bpe:xyz")


def test_clean_lines_trong_van_ban():
    """D-04 A: gộp dòng giống liền nhau, dòng ngắn lặp >= 3 lần giữ lần đầu, bỏ dòng số trang; giữ dòng trống."""
    text = "\n".join(["Tiêu đề", "Tiêu đề", "Nội dung dài một chút để không bị coi là dòng ngắn nhé các bạn ơi", "",
                      "Xem thêm", "Đoạn hai", "Xem thêm", "- 12 -", "Trang 13", "Xem thêm", "Đoạn ba"])
    out, removed = clean_lines_in_doc(text)
    assert out.split("\n") == ["Tiêu đề", "Nội dung dài một chút để không bị coi là dòng ngắn nhé các bạn ơi", "",
                               "Xem thêm", "Đoạn hai", "Đoạn ba"]
    assert removed == ["Tiêu đề", "Xem thêm", "- 12 -", "Trang 13", "Xem thêm"]


def test_clean_lines_lien_van_ban_chi_nguon_web():
    """D-04 C: dòng có ở >= ngưỡng văn bản bị xoá (chỉ nguồn cross_line_clean); thống kê và cột lines_removed."""
    rows = [{"doc_id": f"d{i}", "source_key": "sea_pile_v2", "text": f"{SENT[i % 5]} số {i}\nĐăng ký nhận bản tin"}
            for i in range(25)]
    stats, removed = clean_lines_cross_doc(rows, min_docs=20, min_frac=0.0001)
    assert stats["threshold"] == 20 and removed["Đăng ký nhận bản tin"] == 25
    assert all("Đăng ký" not in r["text"] for r in rows) and rows[0]["lines_removed"] == 1
    books = [{"doc_id": f"b{i}", "source_key": "stbook", "text": f"{SENT[0]} {i}\nLời nói đầu"} for i in range(25)]
    _, st = clean_rows(books, make_cfg())
    assert "cross_doc" not in st["per_source"]["stbook"] and all("Lời nói đầu" in r["text"] for r in books)


def test_chunk_theo_token_uu_tien_tieu_de():
    """D-10: cắt trước dòng tiêu đề khi đoạn đã đủ min; không vượt max; đoạn cuối ngắn gộp vào đoạn trước."""
    para = " ".join(["từ"] * 120)
    text = "\n\n".join([para, para, "Chương 2. Mở đầu", para, para, para, "đoạn cuối"])
    chunks = chunk_text(text, WORDS, target=300, max_tokens=400, min_tokens=100)
    assert chunks[1].startswith("Chương 2")  # cắt ngay trước tiêu đề dù đoạn đầu chưa đạt mục tiêu
    assert all(WORDS.count(c) <= 400 for c in chunks) and chunks[-1].endswith("đoạn cuối")
    assert is_heading("CHƯƠNG III\nNội dung") and is_heading("Mục 2.1 Khái niệm") and not is_heading("Bài 5 nói về x.")


def test_chunk_doan_van_qua_dai_cat_theo_cau_roi_cat_cung():
    """Đoạn văn dài hơn max: tách theo câu; câu dài hơn max: cắt cứng."""
    long_para = " ".join(SENT * 6)  # ~ 600 từ, có dấu chấm
    chunks = chunk_text(long_para, WORDS, target=100, max_tokens=150, min_tokens=30)
    assert len(chunks) > 3 and all(WORDS.count(c) <= 150 for c in chunks)
    assert all(c.rstrip().endswith(".") for c in chunks[:-1])
    hard = chunk_text(" ".join(["x"] * 500), WORDS, target=100, max_tokens=150, min_tokens=30)
    assert [WORDS.count(c) for c in hard] == [150, 150, 150, 50]


def test_chunk_rows_bai_web_dai_moi_cat_va_gan_van_tay_van_ban():
    """Bài web ngắn giữ nguyên, bài dài hơn max bị cắt; mọi đoạn mang doc_sha256 / doc_minhash của cả văn bản gốc."""
    cfg = make_cfg(mix={"sea_pile_v2": 2}, chunk_target_tokens=100, chunk_max_tokens=150, chunk_min_tokens=30)
    rows = [{"doc_id": "ngan", "source_key": "sea_pile_v2", "text": vi_text(3, 1)},
            {"doc_id": "dai", "source_key": "sea_pile_v2", "text": "\n\n".join(vi_text(5, i) for i in range(6))}]
    out, stats = chunk_rows(rows, cfg, WORDS)
    by_parent = {}
    for r in out:
        by_parent.setdefault(r.get("parent_doc_id") or r["doc_id"], []).append(r)
    assert len(by_parent["ngan"]) == 1 and by_parent["ngan"][0].get("parent_doc_id") is None
    assert len(by_parent["dai"]) > 1 and len({r["doc_sha256"] for r in by_parent["dai"]}) == 1
    assert all(r["doc_minhash"] is not None and r["token_count"] <= 150 for r in by_parent["dai"])
    assert stats["per_source"]["sea_pile_v2"]["docs_split"] == 1


class FakeLid:
    """Bộ nhận diện giả: đoạn có chữ 'English' là en (0,95), còn lại vi (0,98)."""

    def predict(self, segments):
        """(nhãn, độ tin cậy) cho từng đoạn."""
        return [("en", 0.95) if "English" in s else ("vi", 0.98) for s in segments]


def test_detect_mix_theo_doan_vi_du_d02():
    """Ví dụ D-02: 10 đoạn dài bằng nhau, 8 vi (0,98) + 2 en (0,95) -> vi, lang_mix 0,8 / 0,2, lang_score ~0,78."""
    vi_seg, en_seg = "x" * 49 + "a", "English " + "y" * 42
    text = "\n\n".join([vi_seg] * 8 + [en_seg] * 2)
    lang, score, mix = detect_mix(text, FakeLid())
    assert lang == "vi" and mix == {"vi": 0.8, "en": 0.2} and score == pytest.approx(0.784, abs=1e-3)
    assert detect_mix("123 456", FakeLid()) == ("und", 0.0, {})  # F-07: không còn chữ cái
    assert len(segments_of("\n\n".join(f"đoạn {i}" for i in range(200)), 50)) == 50


def test_lid_text_bo_luot_system_cua_sea_instruct():
    """Hội thoại: chỉ nội dung các lượt không phải system đi vào nhận diện (không có tiền tố vai)."""
    turns = [{"role": "system", "content": "You are an AI assistant."}, {"role": "human", "content": "Xin chào"},
             {"role": "gpt", "content": "Chào bạn"}]
    row = {"text": "system: You are an AI assistant.\nhuman: Xin chào\ngpt: Chào bạn", "meta": json.dumps({"turns": turns})}
    assert lid_text(row) == "Xin chào\n\nChào bạn"
    assert lid_text({"text": "abc", "meta": None}) == "abc"


@pytest.mark.skipif(not (Path(os.environ.get("VI_CORPUS_MODEL_DIR", "~/.cache/vi_corpus")).expanduser()
                         / "lid.176.bin").exists(), reason="chưa có lid.176.bin (tải ở lần chạy pipeline đầu tiên)")
def test_fasttext_lid176_that():
    """fastText lid.176 thật: tiếng Việt / tiếng Anh, văn bản song ngữ ra lang_mix hai ngôn ngữ."""
    from vi_corpus.pipeline.language import get_lid

    lid = get_lid("fasttext:lid.176")
    lang, score, mix = detect_mix(" ".join(SENT), lid)
    assert lang == "vi" and score > 0.9
    en = "The history of the country is one of the most important parts of the world and it is studied widely."
    lang, _, mix = detect_mix("\n\n".join([" ".join(SENT[:3]), en]), lid)
    assert lang == "vi" and set(mix) == {"vi", "en"}


def test_ma_chat_luong_ngon_ngu():
    """low_lang_score trừ điểm nhẹ, mixed_language chỉ gắn nhãn, low_diacritic bắt tiếng Việt không dấu."""
    prof = make_cfg().profile("sea_pile_v2")
    m = compute_metrics(" ".join(SENT))
    score, codes = score_metrics(m, "vi", prof, 0.5, json.dumps({"vi": 0.7, "en": 0.3}))
    assert "low_lang_score" in codes and "mixed_language" in codes and score == 90.0
    khong_dau = "Ha Noi la thu do cua nuoc Viet Nam va la trung tam chinh tri van hoa cua ca nuoc tu rat lau roi " * 3
    _, codes = score_metrics(compute_metrics(khong_dau), "vi", prof, 0.9, "{}")
    assert "low_diacritic" in codes


def test_mien_xoa_dong_cho_van_ban_co_code_hoac_bang():
    """C-07: văn bản có code / bảng không bị xoá dòng ở mục A; mục C không xoá dòng cú pháp / ô số trong văn bản đó."""
    from vi_corpus.pipeline.lines import has_code_or_table

    code = "Ví dụ hàm đệ quy trong C:\nint f(int n) {\n  if (n < 2) {\n    return 1;\n  }\n  return n * f(n - 1);\n}\n}"
    table = "Bảng 1. Sản lượng\nNăm\n2020\n2021\nLúa\n12\n13\nNgô\n5\n5"
    pipe = "| Tỉnh | Dân số |\n|---|---|\n| Hà Nội | 8 |"
    phap_luat = "Điều 5. Điều kiện\na) Có quốc tịch Việt Nam;\nb) Đủ 18 tuổi;\nc) Có sức khoẻ tốt;"
    assert has_code_or_table(code) and has_code_or_table(table) and has_code_or_table(pipe)
    assert not has_code_or_table(phap_luat) and not has_code_or_table(" ".join(SENT))
    trafilatura = "Tỉnh | Dân số | Diện tích\nHà Nội | 8 | 3.359\nHuế | 1 | 4.947"  # bảng không có dòng kẻ
    menu = "Trang chủ | Tin tức | Thể thao\nGiới thiệu | Liên hệ\nTrang chủ | Tin tức | Thể thao"  # menu, không phải bảng
    python = "Chạy đoạn sau:\nimport numpy as np\nfor i in range(3):\n    print(i)"
    assert has_code_or_table(trafilatura) and has_code_or_table(python) and not has_code_or_table(menu)
    rows = [{"doc_id": "c", "source_key": "stbook", "text": code},
            {"doc_id": "t", "source_key": "stbook", "text": table + "\nTrang 12"}]
    _, st = clean_rows(rows, make_cfg())
    assert rows[0]["text"] == code and rows[1]["text"] == table  # chỉ nhãn số trang chắc chắn bị bỏ
    assert st["per_source"]["stbook"]["in_doc_exempt_docs"] == 2
    web = [{"doc_id": f"w{i}", "source_key": "sea_pile_v2", "text": f"{SENT[i % 5]} {i}\n}}\nĐăng ký nhận bản tin"}
           for i in range(25)]
    web[0]["text"] = code + "\nĐăng ký nhận bản tin"
    _, st = clean_rows(web, make_cfg(cross_line_min_docs=20, cross_line_min_frac=0.0001))
    assert web[0]["text"] == code  # "}" giữ, câu mời đăng ký bị xoá
    assert all("}" not in r["text"] and "Đăng ký" not in r["text"] for r in web[1:])


def test_mien_xoa_dong_code_python_mat_thut_le():
    """Python đã mất thụt lề (trích PDF / OCR) vẫn được nhận là code nhờ dòng "yếu" (gán, return) trong khối có dòng
    chắc; mục A không xoá thân hàm lặp. Chỉ có công thức toán (toàn dòng yếu) thì không coi là code."""
    from vi_corpus.pipeline.lines import has_code_or_table
    from vi_corpus.pipeline.normalize import normalize_text

    fn = "def xep_loai{i}(d):\nif d >= 8:\nreturn 'gioi'\nelse:\nreturn 'kha'"
    text = normalize_text("Các hàm xếp loại:\n" + "\n\n".join(fn.format(i=i) for i in range(4)))
    rows = [{"doc_id": "p", "source_key": "stbook", "text": text}]
    clean_rows(rows, make_cfg())
    assert rows[0]["text"] == text and rows[0]["text"].count("return 'gioi'") == 4
    assert not has_code_or_table("Ta có:\ny = ax + b\nx = 2\nz = x + y\nVậy z bằng 2a + b + 2.")


def test_normalize_giu_thut_le_cho_code():
    """C-09: văn bản có code giữ khoảng trắng đầu dòng qua normalize, xoá dòng và cắt đoạn (chỉ gộp khoảng trắng giữa
    dòng, bỏ khoảng trắng cuối dòng); văn bản không có code vẫn bỏ khoảng trắng đầu dòng."""
    from vi_corpus.pipeline.chunk import chunk_text
    from vi_corpus.pipeline.normalize import normalize_rows, normalize_text
    from vi_corpus.pipeline.tokens import TokenCounter

    raw = ("Ví dụ:\u00a0 hàm  đệ quy\n\n    def f(n):   \n        if n  <  2:\n            return 1\n\n"
           "        return n * f(n - 1)\n\tprint(f(5))")
    want = ("Ví dụ: hàm đệ quy\n\n    def f(n):\n        if n < 2:\n            return 1\n\n"
            "        return n * f(n - 1)\n\tprint(f(5))")
    assert normalize_text(raw) == want
    assert normalize_text("  Đoạn văn   thụt đầu dòng.\n   Dòng hai.  ") == "Đoạn văn thụt đầu dòng.\nDòng hai."
    rows = [{"doc_id": "c", "source_key": "stbook", "text": raw},
            {"doc_id": "v", "source_key": "stbook", "text": "  Văn xuôi."}]
    _, st = normalize_rows(rows)
    assert st["indent_kept_docs"] == 1
    clean_rows(rows, make_cfg())
    assert rows[0]["text"] == want
    counter = TokenCounter("words")
    assert "\n\n".join(chunk_text(want, counter, 5, 12, 2)) == want  # đoạn văn bắt đầu bằng thụt lề giữ nguyên


def test_mien_xoa_dong_code_pascal():
    """Pascal chỉ có phép gán ":=" và begin / end vẫn là code (mục A không xoá dòng "begin" lặp)."""
    from vi_corpus.pipeline.lines import has_code_or_table

    pascal = "Chương trình:\nbegin\na := 1;\nb := a + 2;\nend;\nbegin\nc := b * 2;\nend."
    assert has_code_or_table(pascal)
