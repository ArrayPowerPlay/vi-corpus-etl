"""
Test bước chuẩn bị CPT vi_corpus.pipeline.cpt_pack (T-07, T-08, T-09): cắt segment đúng ranh giới đoạn, EOS chỉ ở cuối thật
của văn bản, best-fit packing không vượt độ dài chuỗi, tiền tố metadata tắt mặc định, đọc / ghi Parquet theo lô.
"""

import json

import pyarrow.parquet as pq
import pytest

from vi_corpus.pipeline.cpt_pack import best_fit_pack, pack, parse_buckets, render_prefix, split_segment
from vi_corpus.pipeline.knowledge import build_cpt_segments, write_segments


def _rows(doc: str, sizes: list[int], key: str = "vjol", meta: str | None = None) -> list[dict]:
    """Các đoạn còn giữ của một văn bản, đoạn i có chữ "<doc>-<i>" và sizes[i] token."""
    return [{"doc_id": f"{doc}#c{i}", "parent_doc_id": doc, "chunk_index": i, "source_key": key, "status": "kept",
             "text": f"{doc}-{i}", "token_count": n, "dedup_family_id": f"fam_{doc}", "meta": meta}
            for i, n in enumerate(sizes)]


def test_cat_segment_dung_ranh_gioi_doan_va_eos_chi_o_cuoi_van_ban():
    """Segment dài bị cắt ở ranh giới đoạn, mỗi mảnh vừa chuỗi; chỉ mảnh cuối của văn bản có EOS; văn bản bị ngắt giữa chừng không có EOS."""
    seg = build_cpt_segments(_rows("A", [3000, 3000, 3000, 3000]))[0]
    pieces = split_segment(seg, 8192)
    assert [p["text"] for p in pieces] == ["A-0\n\nA-1", "A-2\n\nA-3"]
    assert [p["token_count"] for p in pieces] == [6001, 6001]
    assert [p["eos"] for p in pieces] == [False, True] and all(p["cost"] <= 8192 for p in pieces)
    cut = build_cpt_segments(_rows("B", [100, 100]) + [{**_rows("B", [1, 1, 1])[2], "status": "rejected:quality"}])[0]
    assert not cut["doc_end"] and [p["eos"] for p in split_segment(cut, 8192)] == [False]


def test_doan_qua_dai_thanh_manh_rieng_oversize():
    """Một đoạn dài hơn chuỗi vẫn thành một mảnh riêng (không lặp vô hạn), được đánh dấu oversize."""
    pieces = split_segment(build_cpt_segments(_rows("C", [100, 9000, 100]))[0], 8192)
    assert [p["token_count"] for p in pieces] == [100, 9000, 100] and [p["oversize"] for p in pieces] == [False, True, False]


def test_best_fit_khong_vuot_do_dai_va_lap_day():
    """Best-fit decreasing: mọi chuỗi <= seq_len, mảnh lớn ghép với mảnh vừa chỗ trống nhỏ nhất."""
    items = [{"cost": c, "id": i} for i, c in enumerate([6000, 5000, 3000, 2000, 1000, 100])]
    bins = best_fit_pack(items, 8192)
    assert all(sum(p["cost"] for p in b) <= 8192 for b in bins)
    assert [[p["cost"] for p in b] for b in bins] == [[6000, 2000, 100], [5000, 3000], [1000]]
    assert sum(len(b) for b in bins) == len(items)


def test_tien_to_metadata():
    """render_prefix in các khoá có giá trị theo thứ tự cố định; meta rỗng / hỏng thì không có tiền tố."""
    assert render_prefix('{"source": "vjol", "title": "Bài A", "year": 2020}') == "Tiêu đề: Bài A\nNăm: 2020\nNguồn: vjol\n\n"
    assert render_prefix(None) == "" and render_prefix("không phải json") == ""


def test_pack_ghi_parquet_tat_tien_to_mac_dinh_va_loc_bucket(tmp_path):
    """pack: tiền tố TẮT mặc định; bật thì có cột prefix + prefix_tokens; thống kê đúng; lọc theo split_bucket."""
    segs = build_cpt_segments(_rows("A", [3000, 3000, 3000], meta='{"title": "Bài A"}') + _rows("B", [500])
                              + _rows("C", [7000], key="stbook"))
    path = tmp_path / "cpt_segments.parquet"
    write_segments(path, segs)
    out = tmp_path / "packed.parquet"
    stats = pack([path], out, seq_len=8192, buffer_tokens=1)  # vùng đệm 1 token: ghi ra sau mỗi segment
    rows = pq.read_table(out).to_pylist()
    assert stats["segments"] == 3 and stats["pieces"] == 4 and stats["split_segments"] == 1 and stats["eos_pieces"] == 3
    assert all(r["prefix"] is None and r["prefix_tokens"] == 0 for r in rows)
    by_seq: dict[int, int] = {}
    for r in rows:
        by_seq[r["seq_id"]] = by_seq.get(r["seq_id"], 0) + r["token_count"] + r["eos"]
    assert all(v <= 8192 for v in by_seq.values()) and stats["tokens"] == sum(r["token_count"] for r in rows)

    pack([path], out, seq_len=8192, use_metadata_prefix=True, count_tokens=lambda xs: [len(x.split()) for x in xs])
    rows = pq.read_table(out).to_pylist()
    a = [r for r in rows if r["segment_id"] == "A:s0"]
    assert all(r["prefix"] == "Tiêu đề: Bài A\nNguồn: vjol\n\n" and r["prefix_tokens"] == 6 for r in a)
    with pytest.raises(ValueError):
        pack([path], out, use_metadata_prefix=True)

    bucket = segs[0]["split_bucket"]
    stats = pack([path], out, buckets=(bucket, bucket))
    assert stats["segments"] == sum(s["split_bucket"] == bucket for s in segs)
    assert json.loads(json.dumps(stats))["sequences"] >= 1


def test_parse_buckets():
    """"A-B" hợp lệ, một số, rỗng = mọi bucket; ngoài 0..99 hoặc ngược thì báo lỗi."""
    assert parse_buckets("0-89") == (0, 89) and parse_buckets("95") == (95, 95) and parse_buckets("") is None
    with pytest.raises(ValueError):
        parse_buckets("90-10")
