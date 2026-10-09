"""
Test dedup vòng 1 (theo văn bản, nhóm cpt / sft, ưu tiên nguồn R-15) và vòng 2 (kho dấu vân tay, verdict, apply_verdict,
D-09). Không cần mạng.
"""

import numpy as np

from tests.test_pipeline import _row, make_cfg, vi_text
from vi_corpus.pipeline.dedup import (
    UnionFind,
    dedup_rows,
    index_rows,
    lsh_pairs,
    minhash,
    minhash_bytes,
    write_index,
)
from vi_corpus.pipeline.dedup_global import (
    apply_verdict,
    global_verdict,
    read_index,
    write_verdict,
)


def test_union_find_goc_luon_la_chi_so_nho_nhat():
    """Gộp hai họ có gốc lệch thứ tự vẫn ra gốc nhỏ nhất (bản tốt nhất)."""
    uf = UnionFind(8)
    uf.union(3, 4)
    uf.union(1, 7)
    uf.union(4, 7)
    assert {uf.find(i) for i in (1, 3, 4, 7)} == {1}


def test_lsh_pairs_vector_hoa():
    """Bản gần trùng ra cặp, văn bản khác không ra cặp."""
    base = vi_text(30, 5)
    sigs = np.stack([minhash(base), minhash(base.replace("Hà Nội", "Thủ đô Hà Nội", 1)), minhash(vi_text(30, 77))])
    assert lsh_pairs(sigs, 0.8).tolist() == [[0, 1]]


def test_uu_tien_nguon_khi_trung_giua_nguon():
    """R-15: SEA được giữ trước stbook dù điểm thấp hơn; cùng nguồn thì điểm cao hơn được giữ."""
    text = vi_text(30, 3)
    rows = [_row("stbook:x", text, 99, key="stbook"), _row("sea:x", text, 60), _row("sea:y", text, 80)]
    rows, _ = dedup_rows(rows, make_cfg())
    st = {r["doc_id"]: r for r in rows}
    assert st["sea:y"]["status"] == "kept"
    assert st["stbook:x"]["dup_of"] == st["sea:x"]["dup_of"] == "sea:y"
    assert len({r["dedup_family_id"] for r in rows}) == 1


def test_sft_chi_dedup_trong_nhom():
    """Q4: hội thoại SEA-Instruct trùng text với bài web không bị loại; hai hội thoại trùng nhau thì bị."""
    text = vi_text(30, 4)
    rows = [_row("web", text), _row("chat1", text, 90, key="sea_instruct_2602"),
            _row("chat2", text, 80, key="sea_instruct_2602")]
    rows, stats = dedup_rows(rows, make_cfg())
    st = {r["doc_id"]: r["status"] for r in rows}
    assert st == {"web": "kept", "chat1": "kept", "chat2": "rejected:duplicate"}
    assert set(stats["groups"]) == {"cpt", "sft"}


def _chunk(doc_id, parent, idx, text, doc_text, score=90.0, key="stbook"):
    """Bản ghi đoạn có vân tay văn bản gốc (như sau stage prepare)."""
    from vi_corpus.common.schema import text_sha256

    row = _row(doc_id, text, score, key=key)
    row.update(parent_doc_id=parent, chunk_index=idx, doc_sha256=text_sha256(doc_text), doc_minhash=minhash_bytes(doc_text))
    return row


def test_dedup_theo_van_ban_va_trung_doan_chinh_xac():
    """R-19: sách gần trùng thì mọi đoạn của bản kém bị loại; đoạn giống hệt ở sách khác bị loại exact theo đoạn."""
    a = [vi_text(30, i) for i in range(3)]
    book_a = "\n\n".join(a)
    book_b = book_a.replace("Hà Nội", "Thủ đô Hà Nội", 1)  # bản gần trùng của sách A
    other = [vi_text(30, 100), a[1]]  # sách C có một chương chép nguyên của A
    rows = ([_chunk(f"A#c{i}", "A", i, t, book_a, 95) for i, t in enumerate(a)]
            + [_chunk("B#c0", "B", 0, a[0], book_b, 80)]
            + [_chunk(f"C#c{i}", "C", i, t, "\n\n".join(other), 70) for i, t in enumerate(other)])
    rows, _ = dedup_rows(rows, make_cfg())
    st = {r["doc_id"]: r for r in rows}
    assert st["B#c0"]["status"] == "rejected:duplicate" and st["B#c0"]["dup_of"] == "A"
    assert st["C#c0"]["status"] == "kept"
    assert st["C#c1"]["dup_kind"] == "exact" and st["C#c1"]["dup_of"] == "A#c1"
    assert st["A#c0"]["dedup_family_id"] == st["B#c0"]["dedup_family_id"] != st["C#c0"]["dedup_family_id"]


def test_dedup_vong_2_giua_cac_lan_chay(tmp_path):
    """D-09: hai lần chạy (hai nguồn) ghi kho; vòng 2 tìm bản trùng giữa nguồn, áp verdict đổi họ + loại bản kém."""
    text = vi_text(30, 9)
    cfg = make_cfg()
    run1, _ = dedup_rows([_row("sea:1", text, 70), _row("sea:2", vi_text(30, 10))], cfg)
    run2, _ = dedup_rows([_row("stbook:1", text.replace("Hà Nội", "Thủ đô Hà Nội", 1), 99, key="stbook")], cfg)
    write_index(tmp_path, index_rows(run1, cfg, "r1"), "r1")
    write_index(tmp_path, index_rows(run2, cfg, "r2"), "r2")
    write_index(tmp_path, index_rows(run2, cfg, "r2"), "r2")  # chạy lại ghi đè, không nhân đôi
    items = read_index(tmp_path / "state" / "dedup_index")
    assert len(items) == 3
    verdict, stats = global_verdict(items, 0.8)
    v = {x["doc_id"]: x for x in verdict}
    assert v["sea:1"]["status"] == "kept" and v["stbook:1"]["status"] == "duplicate"
    assert v["stbook:1"]["dup_of"] == "sea:1" and v["sea:1"]["family_size"] == 2
    assert stats["cpt"]["fuzzy"] == 1 and stats["cpt"]["largest_family"] == 2
    path = tmp_path / "verdict.parquet"
    write_verdict(path, verdict)
    out = apply_verdict(run2, path)
    assert run2[0]["status"] == "rejected:duplicate" and run2[0]["dup_kind"] == "global_fuzzy"
    assert run2[0]["dedup_family_id"] == v["sea:1"]["dedup_family_id"] and out["rejected"] == 1


def test_l1_van_ban_nam_trong_van_ban_lon_hon():
    """G-05 bước 3: chương chép nguyên vào sách lớn hơn bị loại "contained", chung họ với sách; văn bản chỉ chung một
    đoạn văn thì giữ."""
    chapters = [vi_text(4, 200 + i) for i in range(8)]
    book = "\n\n".join(chapters)
    chapter = "\n\n".join(chapters[2:6])
    partial = "\n\n".join([chapters[0]] + [vi_text(4, 300 + i) for i in range(4)])
    rows = [_row("sach", book, 70, key="stbook"), _row("chuong", chapter, 99), _row("mot_phan", partial, 80)]
    rows, stats = dedup_rows(rows, make_cfg(para_min_words=10))
    st = {r["doc_id"]: r for r in rows}
    assert st["chuong"]["status"] == "rejected:duplicate" and st["chuong"]["dup_kind"] == "contained"
    assert st["chuong"]["dup_of"] == "sach" and st["chuong"]["dedup_family_id"] == st["sach"]["dedup_family_id"]
    assert st["sach"]["status"] == st["mot_phan"]["status"] == "kept"
    assert stats["contained_dups"] == 1


def _write_run(tmp_path, rows, cfg, run, sources):
    """Chạy dedup vòng 1 cho một lần chạy giả và ghi kho dấu vân tay của nó."""
    rows, _ = dedup_rows(rows, cfg)
    write_index(tmp_path, index_rows(rows, cfg, run), run, sources)
    return rows


def test_vong_2_lay_dong_moi_nhat_va_bia_mo(tmp_path):
    """Văn bản giữ ở lần chạy cũ nhưng bị loại ở lần chạy mới (bia mộ) không còn trong vòng 2, nên bản trùng ở nguồn khác
    không bị loại oan; "mới nhất" theo thời điểm ghi, không theo tên file."""
    text = vi_text(30, 21)
    cfg = make_cfg()
    _write_run(tmp_path, [_row("sea:X", text, 90)], cfg, "run_b", ["sea_pile_v2"])
    _write_run(tmp_path, [_row("sea:X", text, 10, band="D")], cfg, "run_a", ["sea_pile_v2"])  # ghi sau, tên xếp trước
    _write_run(tmp_path, [_row("stbook:Z", text, 80, key="stbook")], cfg, "run_c", ["stbook"])
    items = read_index(tmp_path / "state" / "dedup_index")
    assert [it["doc_id"] for it in items] == ["stbook:Z"]
    verdict, _ = global_verdict(items, 0.8)
    assert verdict[0]["status"] == "kept"


def test_ghi_lai_cung_lan_chay_khong_con_dong_cu(tmp_path):
    """Chạy lại cùng tên run mà nguồn không còn văn bản sống: file của nguồn đó ghi lại (chỉ còn bia mộ); nguồn bị bỏ
    khỏi mix thì file cùng tên run bị xoá."""
    cfg = make_cfg()
    _write_run(tmp_path, [_row("sea:1", vi_text(30, 1)), _row("stbook:1", vi_text(30, 2), key="stbook")], cfg, "r",
               ["sea_pile_v2", "stbook"])
    _write_run(tmp_path, [_row("sea:1", vi_text(30, 1), 10, band="D")], cfg, "r", ["sea_pile_v2"])
    index_dir = tmp_path / "state" / "dedup_index"
    assert not (index_dir / "stbook" / "r.parquet").exists()
    assert read_index(index_dir) == []


def test_vong_2_tinh_lai_uu_tien_va_rights_gate(tmp_path):
    """Ưu tiên tính lại ở vòng 2 từ source_key (không dùng số đã ghi); enforce chặn văn bản quyền chưa rõ."""
    text = vi_text(30, 33)
    cfg = make_cfg()
    _write_run(tmp_path, [_row("sea:1", text, 50)], cfg, "r1", ["sea_pile_v2"])
    _write_run(tmp_path, [_row("stbook:1", text, 99, key="stbook")], cfg, "r2", ["stbook"])
    items = read_index(tmp_path / "state" / "dedup_index")
    v = {x["doc_id"]: x for x in global_verdict(items, 0.8)[0]}
    assert v["sea:1"]["status"] == "kept"  # mặc định R-15: SEA trước stbook
    v = {x["doc_id"]: x for x in global_verdict(items, 0.8, {"stbook": 0, "sea_pile_v2": 1}.get)[0]}
    assert v["stbook:1"]["status"] == "kept" and v["sea:1"]["status"] == "duplicate"
    for it in items:
        it["rights_status"] = "open" if it["source_key"] == "stbook" else "unknown"
    verdict, stats = global_verdict(items, 0.8, rights_gate="enforce")
    v = {x["doc_id"]: x for x in verdict}
    assert v["sea:1"]["status"] == "rights" and v["stbook:1"]["status"] == "kept" and stats["cpt"]["rights_blocked"] == 1
    path = tmp_path / "v.parquet"
    write_verdict(path, verdict)
    rows = [_row("sea:1", text, 50)]
    rows, _ = dedup_rows(rows, cfg)
    apply_verdict(rows, path)
    assert rows[0]["status"] == "rejected:rights"


def test_minhash_ca_van_ban_sach_chung_phan_dau(tmp_path):
    """Hai sách dài chung 20.000 từ đầu nhưng khác 5.000 từ sau (Jaccard ~0,67) không bị coi là gần trùng (trước đây
    MinHash chỉ băm 20.000 từ đầu nên hai chữ ký giống hệt)."""
    import random

    rng = random.Random(7)
    syl = "ba bé bi bo ca cá cò cô da dê đi đo ga gà hè hi la lê mi mơ na nê pha phi ra rê sa sê ta tê va vê xa xê".split()
    head = [rng.choice(syl) for _ in range(20_000)]
    a = " ".join(head + [rng.choice(syl) for _ in range(5_000)])
    b = " ".join(head + [rng.choice(syl) for _ in range(5_000)])
    sa, sb = minhash(a), minhash(b)
    assert not np.array_equal(sa, sb)
    assert lsh_pairs(np.stack([sa, sb]), 0.8).tolist() == []
    assert np.array_equal(minhash(a), sa)  # tái lập được


def test_dedup_thong_ke_ho_lon_nhat():
    """R-19 mục 4: thống kê dedup vòng 1 có cỡ họ trùng lớn nhất theo nhóm."""
    text = vi_text(30, 44)
    _, stats = dedup_rows([_row(f"d{i}", text) for i in range(4)] + [_row("x", vi_text(30, 45))], make_cfg())
    assert stats["groups"]["cpt"]["largest_family"] == 4 and stats["largest_family"] == 4
