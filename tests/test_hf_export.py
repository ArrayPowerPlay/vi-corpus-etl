"""Test xuất dataset Hugging Face: làm phẳng meta, split_bucket khớp knowledge, ép kiểu không bịa số."""

import json

from vi_corpus import hf_export as hx
from vi_corpus.pipeline import knowledge


def _row(meta: dict) -> dict:
    """Một dòng clean.parquet tối thiểu."""
    base = {name: None for name, _ in hx.COMMON_CLEAN}
    base.update(doc_id="vjol:x:0#c0", parent_doc_id="vjol:x:0", chunk_index=0, source_key="vjol", text="xin chào",
               language="vi", token_count=3, dedup_family_id="fam_abc", reason_codes=None, meta=json.dumps(meta))
    return base


def test_split_bucket_matches_knowledge():
    """split_bucket của bản xuất phải đúng công thức của knowledge.py (cắt train/val/test nhất quán)."""
    assert hx.split_bucket("fam_abc") == knowledge.split_bucket("fam_abc")


def test_flatten_clean_vjol_and_bad_types():
    """meta được làm phẳng; id số thành chuỗi; giá trị không ép được thành null thay vì số bịa."""
    t = hx.flatten_clean([_row({"article_id": 12, "year": "2024", "pages": "n/a", "keywords": ["a", None]})], "vjol")
    r = t.to_pylist()[0]
    assert r["article_id"] == "12" and r["year"] == 2024 and r["pdf_pages"] is None and r["keywords"] == ["a"]
    assert r["reason_codes"] == [] and r["split_bucket"] == hx.split_bucket("fam_abc")
    assert t.schema == hx.clean_schema("vjol")


def test_flatten_clean_without_meta():
    """Dòng không có meta vẫn xuất được, cột metadata là null."""
    row = _row({})
    row["meta"] = None
    r = hx.flatten_clean([row], "vista").to_pylist()[0]
    assert r["title"] is None and r["authors"] is None


def test_flatten_cpt_reads_year_from_meta_json():
    """cpt_segments lấy year và journal_title từ meta_json."""
    row = {name: None for name, _ in hx.CPT_COLS}
    row.update(segment_id="s", meta_json=json.dumps({"year": 2020, "journal_title": "T"}), doc_ids=["a"], quality_bands=["A"],
               chunk_token_counts=[3], chunk_char_offsets=[0], token_count=3)
    r = hx.flatten_cpt([row]).to_pylist()[0]
    assert r["year"] == 2020 and r["journal_title"] == "T"
