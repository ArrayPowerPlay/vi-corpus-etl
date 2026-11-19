"""
Test cho bộ kiểm tra nhanh nguồn ứng viên (vi_corpus.source_probe), không cần mạng: lấy mẫu parquet trên hệ thống file local,
số đo LaTeX hỏng, ghép trường, tổng hợp.
"""

import random

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from vi_corpus.source_probe.measure import latex_stats, measure_records, summarize, summarize_extra
from vi_corpus.source_probe.sample import make_record, pick_groups, sample_hf_parquet
from vi_corpus.source_probe.specs import SPECS, ProbeSpec


def test_latex_hong_do_dich_may():
    """Khoảng trắng chen sau dấu "\\" (Google Translate) bị đếm; LaTeX lành thì không."""
    ok = latex_stats(r"Ta có $\frac{1}{2} \times 3$.")
    bad = latex_stats(r"$$ \ frac {dp} {dt} = k \ cdot p $$ và $x")
    assert ok == {"has_latex": True, "broken": 0, "odd_dollar": False}
    assert bad["broken"] == 2 and bad["odd_dollar"]
    assert not latex_stats("Văn bản thường, giá 5 đô.")["has_latex"]


def test_pick_groups_on_dinh_va_khong_trung():
    """Cùng seed cho cùng lựa chọn; không chọn trùng cặp; xin nhiều hơn tổng thì lấy hết."""
    a = pick_groups([3, 5, 2], 4, random.Random(1))
    assert a == pick_groups([3, 5, 2], 4, random.Random(1))
    assert len(set(a)) == 4 and all(gi < [3, 5, 2][fi] for fi, gi in a)
    assert len(pick_groups([1, 1], 10, random.Random(0))) == 2


def test_make_record_ghep_truong_hoi_dap():
    """Trường "a+b" được nối bằng dòng trống, trường rỗng bị bỏ; extra chỉ giữ cột có trong dòng."""
    spec = ProbeSpec("x", "hf_rows", "x", text_field="q_vi+a_vi", en_field="q_en+a_en", extra=("score", "missing"))
    rec = make_record(spec, {"q_vi": "Hỏi", "a_vi": "Đáp", "q_en": "Q", "a_en": "", "score": 3}, "r1")
    assert rec == {"id": "r1", "text": "Hỏi\n\nĐáp", "en": "Q", "extra": {"score": 3}}


def test_sample_parquet_local(tmp_path):
    """Đọc đúng số dòng, chỉ cột cần, trên nhiều file và row group (hệ thống file local thay cho HfFileSystem)."""
    for i in range(3):
        table = pa.table({"text": [f"văn bản {i}-{j}" for j in range(40)], "id": [f"{i}-{j}" for j in range(40)],
                          "score": list(range(40)), "junk": ["x"] * 40})
        pq.write_table(table, tmp_path / f"f{i}.parquet", row_group_size=10)
    spec = ProbeSpec("t", "hf_parquet", str(tmp_path), extra=("score",))
    rows = sample_hf_parquet(spec, 25, seed=7, fs=fsspec.filesystem("file"))
    assert len(rows) == 25 and len({r["id"] for r in rows}) == 25
    assert all(set(r["extra"]) == {"score"} for r in rows)
    assert rows == sample_hf_parquet(spec, 25, seed=7, fs=fsspec.filesystem("file"))


def test_measure_va_summarize_offline():
    """Đo bằng bộ nhận diện heuristic + đếm từ; tổng hợp có tỷ lệ chưa dịch và tỷ lệ LaTeX hỏng."""
    vi = "Đây là một đoạn văn bản tiếng Việt khá dài để kiểm tra chất lượng của các nguồn dữ liệu ứng viên mới. " * 8
    records = [{"id": "1", "text": vi, "en": "This is a test.", "extra": {"score": [1.0, 3.0]}},
               {"id": "2", "text": "This text was never translated into Vietnamese at all. " * 10, "en": "x", "extra": {}},
               {"id": "3", "text": vi + r" Công thức $\ frac{1}{2}$.", "en": "y", "extra": {"score": 5}}]
    rows = measure_records(records, lang_model="heuristic", tokenizer="words")
    s = summarize(rows)
    assert s["n"] == 3 and s["mt_untranslated_share"] == round(1 / 3, 3)
    assert s["latex_broken_share"] == 1.0
    assert s["extra"]["score"]["median"] == 3.5


def test_summarize_extra_url_va_chuoi():
    """Cột url thành danh sách tên miền; cột chuỗi đếm giá trị."""
    out = summarize_extra([{"url": "https://www.a.vn/x", "extractor": "docling"}, {"url": "http://a.vn/y", "extractor": "rolmOCR"}])
    assert out["url_hosts_top"] == [("a.vn", 2)]
    assert dict(out["extractor"]) == {"docling": 1, "rolmOCR": 1}


def test_specs_hop_le():
    """Mọi spec có kind được hỗ trợ; dữ liệu dịch máy có trường bản gốc."""
    assert all(s.kind in ("hf_parquet", "hf_rows", "hplt") for s in SPECS.values())
    assert SPECS["vi_pubmed"].en_field == "en"
