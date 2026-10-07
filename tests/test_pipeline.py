"""
Test cho pipeline xử lý chung (vi_corpus.pipeline): từng stage trên dữ liệu nhỏ và chạy end-to-end trên cây data giả.
"""

import gzip
import json
import random

import pyarrow as pa
import pytest
import pyarrow.parquet as pq

from vi_corpus.common.registry import SOURCES
from vi_corpus.pipeline import embed as embed_mod
from vi_corpus.pipeline import io
from vi_corpus.pipeline import reduce as reduce_mod
from vi_corpus.pipeline.chunk import chunk_text
from vi_corpus.pipeline.config import RunConfig, scale_mix
from vi_corpus.pipeline.dedup import dedup_rows
from vi_corpus.pipeline.knowledge import build_units, sft_pairs, split_bucket
from vi_corpus.pipeline.language import detect_language
from vi_corpus.pipeline.normalize import normalize_text
from vi_corpus.pipeline.quality import band_of, compute_metrics, score_metrics
from vi_corpus.pipeline.runner import run_pipeline

SENT = ["Hà Nội là thủ đô của nước Cộng hòa xã hội chủ nghĩa Việt Nam và là trung tâm chính trị của cả nước.",
        "Người dân ở đồng bằng sông Cửu Long trồng lúa, nuôi thủy sản và có nhiều loại trái cây nhiệt đới.",
        "Giáo dục đại học đã được cải cách trong nhiều năm qua để đáp ứng yêu cầu của thị trường lao động.",
        "Các nhà khoa học cho rằng biến đổi khí hậu sẽ ảnh hưởng đến sản xuất nông nghiệp trong những thập kỷ tới.",
        "Văn hóa truyền thống của dân tộc được gìn giữ qua các lễ hội, phong tục và những làn điệu dân ca."]


def vi_text(n_sent: int, seed: int) -> str:
    """Văn bản tiếng Việt giả gồm n_sent câu chọn ngẫu nhiên, mỗi câu thêm số để các văn bản khác nhau."""
    rng = random.Random(seed)
    return " ".join(f"{rng.choice(SENT)} Năm {rng.randint(1000, 9999)}." for _ in range(n_sent))


def test_normalize_nfc_va_ky_tu_vo_hinh():
    """Chữ + dấu rời thành NFC, bỏ zero-width, đổi khoảng trắng lạ, gộp dòng trống thừa."""
    assert normalize_text("Việt​  Nam\r\n\n\n\nx") == "Việt Nam\n\nx"


def test_language_vi_en_other():
    """Nhận ra tiếng Việt, tiếng Anh; bảng số không có chữ thì other."""
    assert detect_language(vi_text(3, 1))[0] == "vi"
    assert detect_language("The history of the country is one of the most important parts of the world and it is")[0] == "en"
    assert detect_language("1 2 3 4 5 6 7 8")[0] == "other"


def test_chunk_text_giu_ranh_gioi_doan_va_gop_doan_cuoi():
    """Đoạn cắt ở ranh giới đoạn văn, không vượt max, đoạn cuối quá ngắn gộp vào đoạn trước."""
    paras = [" ".join(["từ"] * 100) for _ in range(5)] + ["đoạn cuối ngắn"]
    chunks = chunk_text("\n\n".join(paras), target_words=200, max_words=300, min_words=80)
    assert all(len(c.split()) <= 303 for c in chunks)
    assert chunks[-1].endswith("đoạn cuối ngắn") and len(chunks) == 3
    assert chunk_text("", 200, 300, 80) == []


def test_quality_loi_cung_va_band():
    """Văn bản ngắn / lặp bị hard reason và band D; văn bản bình thường band A."""
    cfg = RunConfig(mix={})
    prof = cfg.profile("sea_pile_v2")
    m = compute_metrics(" ".join(SENT))  # 5 câu khác nhau, không lặp
    score, reasons = score_metrics(m, "vi", prof)
    assert band_of(score) == "A" and not reasons
    _, reasons = score_metrics(compute_metrics("quá ngắn"), "vi", prof)
    assert "too_short" in reasons
    spam = "mua ngay giá rẻ nhất thị trường " * 60
    _, reasons = score_metrics(compute_metrics(spam), "vi", prof)
    assert "repeated_ngrams" in reasons


def _row(doc_id, text, score=90.0, band="A", key="sea_pile_v2"):
    """Bản ghi tối thiểu cho test dedup / knowledge."""
    return io.fill({"doc_id": doc_id, "source_key": key, "source_path": "p", "source_sha256": "h", "text": text,
                    "quality_score": score, "quality_band": band, "rights_status": "unknown", "word_count": len(text.split()),
                    "token_count": len(text.split()), "meta": None})


def test_dedup_exact_fuzzy_va_ban_tot_nhat_duoc_giu():
    """Bản trùng chính xác và gần trùng bị loại, bản điểm cao nhất làm đại diện, bản khác giữ nguyên, band D bị loại."""
    base = vi_text(30, 5)
    near = base.replace("Hà Nội", "Thủ đô Hà Nội", 1)
    rows = [_row("a", base, 80), _row("b", base, 95), _row("c", near, 70), _row("d", vi_text(30, 99), 85),
            _row("e", "x", 10, "D")]
    rows, stats = dedup_rows(rows, RunConfig(mix={}))
    st = {r["doc_id"]: r for r in rows}
    assert st["b"]["status"] == "kept" and st["a"]["dup_kind"] == "exact" and st["a"]["dup_of"] == "b"
    assert st["c"]["status"] == "rejected:duplicate" and st["c"]["dup_kind"] == "fuzzy"
    assert st["d"]["status"] == "kept" and st["d"]["dedup_family_id"] != st["b"]["dedup_family_id"]
    assert st["e"]["status"] == "rejected:quality" and st["e"]["dedup_family_id"] is None
    assert all(r["rights_gate"] == "quarantine" for r in rows) and stats["exact_dups"] == 1


def test_rights_gate_enforce_loai_ban_ghi():
    """Chế độ enforce loại bản ghi có quyền chưa rõ; chế độ tag thì chỉ gắn nhãn."""
    rows, _ = dedup_rows([_row("a", vi_text(30, 1))], RunConfig(mix={}, rights_gate="enforce"))
    assert rows[0]["status"] == "rejected:rights"


def test_knowledge_units_sft_va_split_theo_ho():
    """Hội thoại thành các cặp SFT, văn bản thành CPT; split_bucket chỉ phụ thuộc dedup_family_id."""
    turns = [{"role": "human", "content": "Q1"}, {"role": "gpt", "content": "A1"},
             {"role": "human", "content": "Q2"}, {"role": "gpt", "content": "A2"}]
    assert sft_pairs(turns) == [("Q1", "A1"), ("Q2", "A2")]
    chat = _row("chat", "human: Q1\ngpt: A1", key="sea_instruct_2602")
    chat.update(status="kept", dedup_family_id="fam_x", meta=json.dumps({"turns": turns}))
    doc = _row("doc", vi_text(5, 2))
    doc.update(status="kept", dedup_family_id="fam_y")
    drop = _row("drop", "bỏ")
    drop.update(status="rejected:quality")
    units = build_units([chat, doc, drop])
    assert [u["unit_type"] for u in units] == ["sft", "sft", "cpt"]
    assert units[0]["split_bucket"] == units[1]["split_bucket"] == split_bucket("fam_x")


def test_pipeline_end_to_end_va_chay_lai(tmp_path):
    """Chạy cả pipeline trên data giả (3 nguồn text + stbook): có đủ file kết quả, lineage 100%, chạy lại thì bỏ qua stage."""
    root = tmp_path
    d = root / "raw/sea_vi/sea_pile_v2/vi"
    d.mkdir(parents=True)
    texts = [vi_text(25, i) for i in range(40)] + ["ngắn quá"] * 5 + [vi_text(25, 0)] * 3  # có rác và bản trùng
    pq.write_table(pa.table({"text": texts}), d / "a.parquet")
    d = root / "raw/sea_vi/sea_lion_pile_v1/sea-pile-mc4/vi"
    d.mkdir(parents=True)
    with gzip.open(d / "b.jsonl.gz", "wt", encoding="utf-8") as f:
        for i in range(30):
            f.write(json.dumps({"text": vi_text(25, 1000 + i)}) + "\n")
    d = root / "raw/sea_vi/sea_instruct_2602/Vietnamese"
    d.mkdir(parents=True)
    conv = "[{'from': 'human', 'value': 'Thủ đô của Việt Nam là gì?'}, {'from': 'gpt', 'value': 'Thủ đô của Việt Nam là Hà Nội.'}]"
    pq.write_table(pa.table({"conversations": [conv] * 10}), d / "c.parquet")
    ocr = root / "interim/stbook_ocr/kinh-dien"
    ocr.mkdir(parents=True)
    pages = [vi_text(8, 5000 + i) for i in range(40)]
    (ocr / "1.json").write_text(json.dumps({"pdf_path": "raw/stbook/kinh-dien/content/1.pdf", "pages": pages,
                                            "book": {"title": "Sách thử"}, "category_slug": "kinh-dien"}), encoding="utf-8")

    cfg = RunConfig(mix={"sea_pile_v2": 40, "sea_lion_pile_v1": 20, "sea_instruct_2602": 8, "stbook": 10},
                    rows_per_file=1000, chunk_target_words=200, chunk_max_words=300, embed_spec=None)
    run = root / "run"
    manifest = run_pipeline(SOURCES, root, run, cfg)
    for name in ("01_ingest", "02_prepare", "03_language", "04_quality", "05_dedup", "clean"):
        assert (run / f"{name}.parquet").exists()
    assert (run / "report.html").exists() and (run / "knowledge_units.parquet").exists()
    rows = io.read_rows(run / "05_dedup.parquet")
    assert len(rows) == 40 + 20 + 8 + 10
    assert {r["source_key"] for r in rows} == {"sea_pile_v2", "sea_lion_pile_v1", "sea_instruct_2602", "stbook"}
    assert json.loads((run / "audit.json").read_text(encoding="utf-8"))["lineage_rate_all"] == 1.0
    assert manifest["finalize"]["clean_rows"] > 0
    assert sum(r["status"] == "rejected:duplicate" for r in rows) >= 3  # 3 bản trùng chính xác của vi_text(25, 0)
    first = (run / "05_dedup.parquet").stat().st_mtime_ns
    run_pipeline(SOURCES, root, run, cfg)
    assert (run / "05_dedup.parquet").stat().st_mtime_ns == first  # chạy lại không tính lại stage


def test_chay_tung_buoc_bang_until(tmp_path):
    """until dừng sau stage đã chọn (chưa có file của stage sau); chạy tiếp thì dùng lại các stage đã xong."""
    d = tmp_path / "raw/sea_vi/sea_pile_v2/vi"
    d.mkdir(parents=True)
    pq.write_table(pa.table({"text": [vi_text(25, i) for i in range(20)]}), d / "a.parquet")
    cfg = RunConfig(mix={"sea_pile_v2": 15}, embed_spec=None)
    run = tmp_path / "run"
    run_pipeline(SOURCES, tmp_path, run, cfg, until="prepare")
    assert (run / "02_prepare.parquet").exists() and not (run / "03_language.parquet").exists()
    first = (run / "01_ingest.parquet").stat().st_mtime_ns
    manifest = run_pipeline(SOURCES, tmp_path, run, cfg)
    assert (run / "01_ingest.parquet").stat().st_mtime_ns == first and "finalize" in manifest


def test_embed_reduce_viz_tfidf(tmp_path):
    """Stage embed (tfidf, CPU) + reduce + bản đồ plotly: ra file embedding, toạ độ 2D, cụm và bản đồ nhúng trong report.html."""
    pytest.importorskip("sklearn")
    pytest.importorskip("plotly")
    d = tmp_path / "raw/sea_vi/sea_pile_v2/vi"
    d.mkdir(parents=True)
    pq.write_table(pa.table({"text": [vi_text(25, i) for i in range(60)]}), d / "a.parquet")
    cfg = RunConfig(mix={"sea_pile_v2": 60}, embed_spec="tfidf", prefer_gpu=False)
    run = tmp_path / "run"
    manifest = run_pipeline(SOURCES, tmp_path, run, cfg)
    ids, matrix = embed_mod.read_embeddings(run / "06_embeddings.parquet")
    assert len(ids) == 60 and matrix.shape[0] == 60
    reduced = reduce_mod.read_reduced(run / "07_reduced.parquet")
    assert len(reduced) == 60 and all(r["pca_x"] is not None for r in reduced)
    report = (run / "report.html").read_text(encoding="utf-8")
    assert "reduce" in manifest and "Bản đồ embedding" in report
    assert 'id="emb-data"' in report and 'data-k="pca-status"' in report and not (run / "viz").exists()
