"""
Test cho schema chung và adapter nguồn text (vi_corpus.common.schema, vi_corpus.sea.reader), không cần mạng.
"""

import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq

from vi_corpus.common.schema import CORPUS_SCHEMA, lineage_rate
from vi_corpus.common.registry import SOURCES, load_sources
from vi_corpus.sea.reader import conversations_to_text, iter_records


def test_conversations_dung_literal_eval():
    """Chuỗi Python-repr (nháy đơn, None) phải được parse, không dùng json."""
    raw = "[{'from': 'human', 'value': 'Xin chào', 'x': None}, {'from': 'gpt', 'value': 'Chào bạn'}]"
    assert conversations_to_text(raw) == "human: Xin chào\ngpt: Chào bạn"
    assert conversations_to_text("không hợp lệ (") == ""


def test_iter_records_parquet_va_jsonl(tmp_path):
    """Đọc được cả parquet lẫn jsonl.gz, ra đủ trường schema và lineage 100%."""
    d1 = tmp_path / "raw/sea_vi/sea_pile_v2/vi"
    d1.mkdir(parents=True)
    pq.write_table(pa.table({"text": ["một", "hai"]}), d1 / "a.parquet")
    d2 = tmp_path / "raw/sea_vi/sea_lion_pile_v1/sea-pile-mc4/vi"
    d2.mkdir(parents=True)
    with gzip.open(d2 / "b.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"text": "ba"}) + "\n")

    rows = list(iter_records(SOURCES["sea_pile_v2"], tmp_path)) + list(
        iter_records(SOURCES["sea_lion_pile_v1"], tmp_path)
    )
    assert [r["text"] for r in rows] == ["một", "hai", "ba"]
    assert set(rows[0]) == set(CORPUS_SCHEMA.names)
    assert len({r["doc_id"] for r in rows}) == 3
    assert lineage_rate(rows) == 1.0


def test_load_sources_ghi_de_va_them_nguon(tmp_path):
    """File cấu hình ghi đè đường dẫn nguồn có sẵn và thêm được nguồn mới (Drive)."""
    cfg = tmp_path / "s.json"
    cfg.write_text(json.dumps({
        "vista": {"path": "/mnt/vista"},
        "drive_books": {"fmt": "parquet", "path": "/d", "owner": "o", "license": "l", "domain": "sách"},
    }), encoding="utf-8")
    sources = load_sources(cfg)
    assert sources["vista"].path == "/mnt/vista" and sources["vista"].fmt == "pdf"
    assert sources["drive_books"].domain == "sách"
    assert "drive_books" not in SOURCES
