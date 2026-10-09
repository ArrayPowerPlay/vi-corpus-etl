"""
Test các phần mới của finalize và các bước phụ: vân tay stage (G-02), gom cụm HDBSCAN trên không gian embedding (G-01),
khối CPT (R-09), quét nhiễm benchmark 13-gram (R-16), bộ đọc định dạng theo magic bytes (D-05) và lấy mẫu chấm điểm
phân tầng (G-06). Không cần mạng.
"""

import json
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.test_pipeline import SOURCES, make_cfg, vi_text
from vi_corpus.common.parsers import detect_format, parse_file
from vi_corpus.pipeline.contamination import load_benchmarks, scan
from vi_corpus.pipeline.judge_sample import allocate, draw_sample, length_bucket
from vi_corpus.pipeline.knowledge import pack_cpt_blocks
from vi_corpus.pipeline.runner import fingerprints, run_pipeline

PROSE_WORDS = ("người dân thành phố nông thôn kinh tế phát triển giáo dục học sinh trường lớp giáo viên chính sách nhà "
               "nước sản xuất công nghiệp nông nghiệp lúa gạo xuất khẩu thị trường doanh nghiệp đầu tư vốn ngân hàng "
               "lãi suất văn hoá lịch sử truyền thống lễ hội miền Bắc miền Nam miền Trung sông núi biển đảo khí hậu "
               "mùa mưa nắng sức khoẻ bệnh viện bác sĩ thuốc men khoa học nghiên cứu kỹ thuật máy móc mạng lưới").split()


def prose(n_sent: int, seed: int) -> str:
    """Văn xuôi giả ít lặp (câu ghép từ ngẫu nhiên) để qua được bộ lọc lặp n-gram của stage quality."""
    import random

    rng = random.Random(seed)
    return " ".join(" ".join(rng.choice(PROSE_WORDS) for _ in range(rng.randint(10, 18))).capitalize() + "."
                    for _ in range(n_sent))


def _web_root(tmp_path, n=20, gen=vi_text):
    """Data root giả chỉ có sea_pile_v2 (n bài, sinh bằng gen)."""
    d = tmp_path / "raw/sea_vi/sea_pile_v2/vi"
    d.mkdir(parents=True)
    pq.write_table(pa.table({"text": [gen(25, i) for i in range(n)]}), d / "a.parquet")
    return tmp_path


def test_van_tay_doi_tham_so_thi_chay_lai_tu_stage_do(tmp_path):
    """G-02: đổi ngưỡng fuzzy chỉ đổi vân tay từ dedup; chạy lại thì prepare giữ nguyên, dedup bị tính lại."""
    root = _web_root(tmp_path)
    cfg = make_cfg(mix={"sea_pile_v2": 15}, embed_spec=None)
    other = make_cfg(mix={"sea_pile_v2": 15}, embed_spec=None, fuzzy_threshold=0.9)
    a, b = fingerprints(SOURCES, root, cfg), fingerprints(SOURCES, root, other)
    assert [k for k in a if a[k] != b[k]] == ["dedup", "embed", "reduce"]
    run = root / "run"
    run_pipeline(SOURCES, root, run, cfg)
    prep, ded = ((run / f).stat().st_mtime_ns for f in ("02_prepare.parquet", "05_dedup.parquet"))
    manifest = run_pipeline(SOURCES, root, run, other)
    assert (run / "02_prepare.parquet").stat().st_mtime_ns == prep
    assert (run / "05_dedup.parquet").stat().st_mtime_ns != ded
    assert manifest["fingerprints"]["dedup"] == b["dedup"]


def test_van_tay_keep_stale_va_doi_dau_vao(tmp_path):
    """keep_stale giữ kết quả cũ dù cấu hình đổi; thêm file đầu vào thì vân tay ingest đổi."""
    root = _web_root(tmp_path)
    cfg = make_cfg(mix={"sea_pile_v2": 15}, embed_spec=None)
    run = root / "run"
    run_pipeline(SOURCES, root, run, cfg, until="quality")
    first = (run / "04_quality.parquet").stat().st_mtime_ns
    run_pipeline(SOURCES, root, run, make_cfg(mix={"sea_pile_v2": 15}, embed_spec=None, lang_max_segments=7),
                 until="quality", keep_stale=True)
    assert (run / "04_quality.parquet").stat().st_mtime_ns == first
    before = fingerprints(SOURCES, root, cfg)["ingest"]
    pq.write_table(pa.table({"text": [vi_text(25, 99)]}), root / "raw/sea_vi/sea_pile_v2/vi/b.parquet")
    assert fingerprints(SOURCES, root, cfg)["ingest"] != before


def test_gom_cum_tren_embedding_goc():
    """G-01: HDBSCAN trên embedding gốc tách được 3 cụm rõ; info ghi không gian, số chiều, tham số."""
    pytest.importorskip("sklearn")
    from vi_corpus.pipeline.reduce import cluster_matrix, reduce_embeddings

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 64)) * 10
    matrix = np.vstack([c + rng.normal(scale=0.1, size=(20, 64)) for c in centers])
    rows, info = reduce_embeddings([f"d{i}" for i in range(60)], matrix, prefer_gpu=False)
    assert info["space"] == "raw" and info["dim"] == 64 and info["min_cluster_size"] == 6 and info["clusters"] == 3
    assert len({r["cluster_id"] for r in rows[:20]}) == 1
    assert cluster_matrix(matrix, "pca50", False).shape == (60, 50)
    with pytest.raises(ValueError):
        cluster_matrix(matrix, "tsne", False)


def _kept(doc_id, parent, idx, tokens, status="kept", key="stbook"):
    """Đoạn đã qua dedup cho pack_cpt_blocks."""
    return {"doc_id": doc_id, "parent_doc_id": parent, "chunk_index": idx, "source_key": key, "status": status,
            "text": doc_id, "token_count": tokens, "dedup_family_id": f"fam_{doc_id}"}


def test_khoi_cpt_noi_doan_lien_nhau():
    """R-09: nối đoạn liền nhau còn giữ tới max; đoạn bị loại cắt khối; hội thoại SFT không vào khối."""
    rows = [_kept("A#0", "A", 0, 400), _kept("A#1", "A", 1, 400), _kept("A#2", "A", 2, 400),
            _kept("A#3", "A", 3, 100, status="rejected:quality"), _kept("A#4", "A", 4, 100),
            _kept("web", None, None, 50, key="sea_pile_v2"), _kept("chat", None, None, 50, key="sea_instruct_2602")]
    blocks = pack_cpt_blocks(rows, max_tokens=1000)
    assert [b["doc_ids"] for b in blocks] == [["A#0", "A#1"], ["A#2"], ["A#4"], ["web"]]
    assert blocks[0]["token_count"] == 800 and blocks[0]["dedup_family_id"] == "fam_A#0"


def test_quet_nhiem_benchmark_13gram(tmp_path):
    """R-16: danh sách rỗng thì có ghi chú; benchmark có câu trùng 13 từ liên tiếp thì bản ghi bị đánh dấu."""
    assert "note" in scan([], load_benchmarks(tmp_path / "khong_co"))
    bench = " ".join(f"từ{i}" for i in range(20))
    (tmp_path / "vmlu.jsonl").write_text(json.dumps({"question": bench}, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = [{"doc_id": "x", "status": "kept", "text": "mở đầu " + bench}, {"doc_id": "y", "status": "kept",
                                                                            "text": vi_text(5, 1)},
            {"doc_id": "z", "status": "rejected:duplicate", "text": bench}]
    out = scan(rows, load_benchmarks(tmp_path))
    assert out["flagged"] == {"vmlu": 1} and out["flagged_doc_ids"]["vmlu"] == ["x"]


def test_nhan_dinh_dinh_dang_theo_magic_bytes(tmp_path):
    """D-05: đuôi file sai vẫn nhận đúng định dạng; docx đọc được; html đọc qua trafilatura; định dạng lạ bị bỏ qua."""
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Chương 1. Giới thiệu")
    doc.add_paragraph(vi_text(3, 1))
    path = tmp_path / "giao_trinh.pdf"  # đuôi sai
    doc.save(str(path))
    assert detect_format(path) == "docx"
    out = parse_file(path, lambda: None)
    assert out["error"] is None and "Chương 1" in out["pages"][0]["text"]
    html = tmp_path / "bai.txt"
    html.write_text(f"<!DOCTYPE html><html><body><article><p>{vi_text(6, 2)}</p></article></body></html>",
                    encoding="utf-8")
    assert detect_format(html) == "html"
    assert "Hà Nội" in " ".join(p["text"] for p in parse_file(html, lambda: None)["pages"])
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("a.txt", "x")
    assert parse_file(z, lambda: None)["skipped"] == "unsupported_format"


def test_lay_mau_cham_diem_phan_tang():
    """G-06: tỉ lệ sft ~10%, qua rule ~70%, chia đều nguồn; thiếu mẫu ở tầng nhỏ thì bù sang tầng khác."""
    assert allocate(10, {"a": 2, "b": 100, "c": 100}) == {"a": 2, "b": 4, "c": 4}
    assert length_bucket(100) == "<256" and length_bucket(5000) == ">2k"
    rows = []
    for src, n in (("sea_pile_v2", 500), ("stbook", 300), ("sea_instruct_2602", 200)):
        for i in range(n):
            rows.append({"doc_id": f"{src}:{i}", "source_key": src, "text": f"{src} {i}", "token_count": 100 + 10 * i,
                         "quality_band": "A" if i % 3 else "D"})
    sample, stats = draw_sample(rows, n=200, seed=1)
    assert stats["total"] == 200 and stats["groups"]["sft"]["pass"] + stats["groups"]["sft"]["fail"] == 20
    assert stats["groups"]["cpt"] == {"pool": 800, "pass": 126, "fail": 54}
    srcs = [r["source_key"] for r in sample if r["judge_group"] == "cpt"]
    assert srcs.count("sea_pile_v2") == srcs.count("stbook") == 90
    again, _ = draw_sample(rows, n=200, seed=1)
    assert [r["doc_id"] for r in again] == [r["doc_id"] for r in sample]


def _first_changed(a: dict, b: dict) -> str | None:
    """Stage đầu tiên có vân tay khác nhau giữa hai bộ vân tay."""
    return next((k for k in a if a[k] != b[k]), None)


def test_van_tay_chi_bam_tham_so_stage_doc(tmp_path):
    """G-02: chỉnh ngưỡng chất lượng của hồ sơ nguồn chỉ chạy lại từ quality; đổi chunked chạy lại từ ingest; đổi nhóm
    dedup chạy lại từ dedup; đổi tokenizer chạy lại từ prepare."""
    from dataclasses import replace

    root = _web_root(tmp_path)
    base = make_cfg(mix={"sea_pile_v2": 15}, embed_spec=None)

    def with_profile(**kw):
        """Cấu hình base với hồ sơ sea_pile_v2 đổi vài trường."""
        profiles = dict(base.profiles)
        profiles["sea_pile_v2"] = replace(profiles["sea_pile_v2"], **kw)
        return make_cfg(mix={"sea_pile_v2": 15}, embed_spec=None, profiles=profiles)

    ref = fingerprints(SOURCES, root, base)
    assert _first_changed(ref, fingerprints(SOURCES, root, with_profile(min_words=10))) == "quality"
    assert _first_changed(ref, fingerprints(SOURCES, root, with_profile(min_lang_score=0.3))) == "quality"
    assert _first_changed(ref, fingerprints(SOURCES, root, with_profile(chunked=True))) == "ingest"
    assert _first_changed(ref, fingerprints(SOURCES, root, with_profile(dedup_group="sft"))) == "dedup"
    assert _first_changed(ref, fingerprints(SOURCES, root, make_cfg(mix={"sea_pile_v2": 15}, embed_spec=None,
                                                                   tokenizer="tiktoken:cl100k_base"))) == "prepare"


def test_retry_skipped_lam_lai_file_bi_bo_qua():
    """--retry-skipped: checkpoint skipped chỉ làm lại khi bật cờ; checkpoint lỗi luôn làm lại."""
    from vi_corpus.common.batch_extract import _needs_redo

    skipped = {"skipped": "needs_libreoffice", "error": None, "pages": []}
    assert not _needs_redo(skipped, ocr_enabled=True) and _needs_redo(skipped, True, retry_skipped=True)
    assert _needs_redo({"error": "x", "pages": []}, False) and _needs_redo(None, False)
    assert not _needs_redo({"error": None, "skipped": None, "pages": [{"method": "text"}]}, True, retry_skipped=True)


def test_parser_pptx_epub_va_ole_thieu_libreoffice(tmp_path, monkeypatch):
    """pptx (đuôi sai) và epub đọc được qua parse_file; .doc OLE khi máy không có LibreOffice thì skipped, không lỗi."""
    pptx = pytest.importorskip("pptx")
    pres = pptx.Presentation()
    slide = pres.slides.add_slide(pres.slide_layouts[1])
    slide.shapes.title.text = "Bài 1. Giới thiệu"
    slide.placeholders[1].text = "Nội dung bài giảng về Hà Nội"
    path = tmp_path / "slide.bin"
    pres.save(str(path))
    out = parse_file(path, lambda: None)
    assert out["format"] == "pptx" and out["error"] is None and "Bài 1" in out["pages"][0]["text"]

    epub = pytest.importorskip("ebooklib.epub")
    book = epub.EpubBook()
    book.set_identifier("x")
    book.set_title("Sách")
    book.set_language("vi")
    chapters = []
    for i, body in enumerate(["<h1>Chương 1</h1><p>Mở đầu</p>", "<h1>Chương 2</h1><p>Kết thúc</p>"]):
        ch = epub.EpubHtml(title=f"c{i}", file_name=f"c{i}.xhtml", lang="vi")
        ch.content = f"<html><body>{body}</body></html>".encode()
        book.add_item(ch)
        chapters.append(ch)
    book.spine = chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path = tmp_path / "sach.epub"
    epub.write_epub(str(path), book)
    out = parse_file(path, lambda: None)
    assert out["format"] == "epub" and [p["text"].split("\n")[0] for p in out["pages"]] == ["Chương 1", "Chương 2"]

    from vi_corpus.common.parsers import office

    monkeypatch.setattr(office, "office_binary", lambda: None)
    doc = tmp_path / "cu.doc"
    doc.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 600)
    out = parse_file(doc, lambda: None)
    assert out["format"] == "doc" and out["skipped"] == "needs_libreoffice" and out["error"] is None


def test_finalize_ap_verdict_vong_2_va_so_tokenizer(tmp_path):
    """Runner: --global-verdict loại thêm bản trùng toàn cục ở finalize (các stage trước không chạy lại); kho dấu vân tay
    được ghi; compare_tokenizers ghi tổng token theo tokenizer khác vào audit."""
    from tests.test_prepare_language import _wordlevel_tokenizer
    from vi_corpus.pipeline import io
    from vi_corpus.pipeline.dedup_global import (
        global_verdict,
        read_index,
        write_verdict,
    )

    root = _web_root(tmp_path, n=12, gen=prose)
    other = _wordlevel_tokenizer(tmp_path)
    run = root / "run"
    kw = {"mix": {"sea_pile_v2": 12}, "embed_spec": None, "compare_tokenizers": [other]}
    cfg = make_cfg(**kw)
    run_pipeline(SOURCES, root, run, cfg)
    audit = json.loads((run / "audit.json").read_text(encoding="utf-8"))
    assert audit["tokens_kept_other_tokenizers"][other] > 0
    items = read_index(root / "state" / "dedup_index")
    kept = [r for r in io.read_rows(run / "05_dedup.parquet") if r["status"] == "kept"]
    assert {it["doc_id"] for it in items} == {r["parent_doc_id"] or r["doc_id"] for r in kept}
    victim = kept[0]["parent_doc_id"] or kept[0]["doc_id"]
    verdict, _ = global_verdict(items, 0.8)
    for v in verdict:  # giả lập: một nguồn khác ưu tiên hơn chứa bản trùng của victim
        if v["doc_id"] == victim:
            v.update(status="duplicate", dup_kind="exact", dup_of="vjol:goc")
    path = root / "verdict.parquet"
    write_verdict(path, verdict)
    dedup_mtime = (run / "05_dedup.parquet").stat().st_mtime_ns
    manifest = run_pipeline(SOURCES, root, run, make_cfg(**kw, global_verdict=str(path)))
    assert (run / "05_dedup.parquet").stat().st_mtime_ns == dedup_mtime
    assert manifest["finalize"]["global_verdict"]["rejected"] >= 1
    clean_ids = {r["parent_doc_id"] or r["doc_id"] for r in io.read_rows(run / "clean.parquet")}
    assert victim not in clean_ids
