"""
Test VISTA (VS-01..VS-03): bộ xử lý bài báo dùng chung với VJOL, chỉ khác bố cục (tên file là mã bài, nhiều bài cùng một PDF, thư mục unknown/, metadata vista_detail.jsonl).

Dữ liệu giả dựng bằng PDF có lớp chữ (không cần GPU / mạng). Kiểm: đường dẫn -> mã bài, một PDF dùng chung cho nhiều bài thành MỘT bản ghi mang đủ mã bài,
nối metadata theo luồng (chọn bài mã nhỏ nhất có dòng), unknown/ vẫn thành bản ghi thiếu metadata, dòng lệnh scripts/vista/extract.py (checkpoint riêng, không đụng
interim/vjol_text), registry / profile / run_all / ingest, và pipeline chạy hết với nguồn vista.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from scripts.run_all import PARTS
from tests.test_pipeline import make_cfg, vi_text
from tests.test_vjol import text_pdf
from vi_corpus.common.registry import SOURCES
from vi_corpus.pipeline import io as pio
from vi_corpus.pipeline.config import PROFILES
from vi_corpus.pipeline.ingest import input_files, iter_book_source
from vi_corpus.pipeline.runner import run_pipeline
from vi_corpus.vista.layout import TEXT_DIR, VISTA_LAYOUT, path_meta, publication_id_of
from vi_corpus.vista.records import iter_records, load_metadata, publication_ids
from vi_corpus.vjol import extract as vx

REPO = Path(__file__).resolve().parents[1]


def detail_row(pid: str, year: int, title: str, source: str = "Tạp chí Khoa học Thử Nghiệm") -> dict:
    """Một dòng vista_detail.jsonl tối thiểu (kèm tóm tắt dài để kiểm tra nó không bị giữ lại)."""
    return {"publication_id": pid, "title_vi": title, "title_en": "", "authors": ["Nguyễn Văn A"], "keywords": ["thử"], "source_name": source,
            "issn": "1234-5678", "issue_year": year, "issue_number": "2", "pages": "10-15", "research_field_code": "5020101",
            "research_field_name": "Kinh tế học", "document_type": "", "abstract_vi": "Tóm tắt " * 50, "has_fulltext": True}


def make_raw(root: Path) -> Path:
    """
    <root>/raw/vista gồm: 2019/101.pdf; 2020/201.pdf và 2020/202.pdf CÙNG nội dung (một PDF của hai bài); 2020/203.pdf (bài 203 không có dòng metadata);
    unknown/<tên băm>.pdf (không có trong metadata); metadata/vista_detail.jsonl (bài 202 và 201 đều có dòng, 101 có dòng, thêm một dòng của bài không có PDF).
    """
    raw = root / "raw/vista"
    text_pdf(raw / "2019/101.pdf", [vi_text(6, 1)])
    text_pdf(raw / "2020/201.pdf", [vi_text(6, 2)])
    shutil.copyfile(raw / "2020/201.pdf", raw / "2020/202.pdf")  # cùng từng byte (PDF dựng hai lần có dấu thời gian khác nhau nên phải chép)
    text_pdf(raw / "2020/203.pdf", [vi_text(6, 3)])
    text_pdf(raw / "unknown/0a1b2c3d4e5f.pdf", [vi_text(6, 4)])
    (raw / "metadata").mkdir(parents=True)
    rows = [detail_row("999", 2020, "Bài không có PDF"), detail_row("202", 2020, "Tiêu đề bài 202"), detail_row("101", 2019, "Tiêu đề bài 101"),
            detail_row("201", 2020, "Tiêu đề bài 201")]
    (raw / "metadata/vista_detail.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\nkhông phải json\n", encoding="utf-8")
    return raw


def run_cli(*args) -> subprocess.CompletedProcess:
    """Chạy scripts/vista/extract.py như người dùng gõ lệnh (tiến trình riêng)."""
    return subprocess.run([sys.executable, str(REPO / "scripts/vista/extract.py"), *args], capture_output=True, text=True, timeout=300, check=False)


def test_ten_file_thanh_ma_bai_va_nam():
    """<năm>/<publication_id>.pdf cho mã bài và năm; unknown/<băm>.pdf không có cả hai."""
    assert publication_id_of("raw/vista/2020/201.pdf") == "201" and publication_id_of("raw/vista/unknown/0a1b2c.pdf") is None
    assert path_meta(Path("raw/vista/2020/201.pdf")) == {"year_dir": 2020, "publication_id": "201"}
    assert path_meta(Path("raw/vista/unknown/0a1b2c.pdf")) == {"year_dir": None, "publication_id": None}
    assert publication_ids({"paths": ["raw/vista/2020/202.pdf", "raw/vista/2020/9.pdf", "raw/vista/unknown/ab.pdf"]}) == ["9", "202"]  # xếp theo SỐ, bỏ tên không phải mã


def test_dong_lenh_trich_text_checkpoint_rieng_va_xuat_parquet(tmp_path):
    """E2E: scripts/vista/extract.py trích mọi PDF (kể cả unknown/), hai file cùng nội dung chỉ một checkpoint, ghi vào interim/vista_text chứ không vào vjol_text."""
    make_raw(tmp_path)
    r = run_cli("--data-root", str(tmp_path), "--no-ocr")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[vista] 5 PDF trong raw; 4 đã trích text" in r.stdout  # 5 file, 4 nội dung duy nhất
    assert len(list((tmp_path / TEXT_DIR).glob("*.json"))) == 4 and not (tmp_path / vx.TEXT_DIR).exists()
    assert pq.read_table(tmp_path / "processed/vista/vista.parquet").num_rows == 4
    shared = [json.loads(p.read_text()) for p in (tmp_path / TEXT_DIR).glob("*.json") if len(json.loads(p.read_text())["paths"]) == 2]
    assert len(shared) == 1 and sorted(Path(x).stem for x in shared[0]["paths"]) == ["201", "202"]
    assert "4 đã trích text" in run_cli("--data-root", str(tmp_path), "--status").stdout
    assert run_cli("--data-root", str(tmp_path / "khong_co"), "--status").returncode == 2  # thiếu raw/vista


def test_ban_ghi_mot_pdf_dung_chung_mang_du_ma_bai_va_metadata_bai_nho_nhat(tmp_path):
    """PDF của bài 201 và 202 thành MỘT bản ghi: publication_ids đủ, shared_pdf, metadata lấy từ bài 201 (mã nhỏ nhất); unknown/ và bài 203 thiếu metadata; tóm tắt không vào meta."""
    make_raw(tmp_path)
    assert run_cli("--data-root", str(tmp_path), "--no-ocr", "--no-export").returncode == 0
    recs = list(iter_records(SOURCES["vista"], tmp_path, with_meta=True))
    assert len(recs) == 4 and {r["source_key"] for r in recs} == {"vista"}
    by_ids = {tuple(json.loads(r["meta"])["publication_ids"]): json.loads(r["meta"]) for r in recs}
    shared = by_ids[("201", "202")]
    assert shared["shared_pdf"] is True and shared["publication_id"] == "201" and shared["title"] == "Tiêu đề bài 201" and shared["year"] == 2020
    assert shared["journal_title"] == "Tạp chí Khoa học Thử Nghiệm" and shared["metadata_found"] is True and shared["pages"] == 1 and shared["page_range"] == "10-15"
    one = by_ids[("101",)]
    assert one["shared_pdf"] is False and one["title"] == "Tiêu đề bài 101" and one["year"] == 2019
    no203 = by_ids[("203",)]
    assert no203["metadata_found"] is False and no203["title"] is None and no203["year"] == 2020  # năm lấy từ thư mục
    unknown = by_ids[()]
    assert unknown["metadata_found"] is False and unknown["publication_id"] is None and unknown["year"] is None and unknown["year_dir"] is None
    assert all("abstract" not in json.loads(r["meta"]) and "Tóm tắt" not in r["meta"] for r in recs)
    assert all(r["rights_status"] == "unknown" for r in recs)


def test_doc_metadata_khong_co_file_va_dong_hong(tmp_path):
    """Thiếu vista_detail.jsonl thì trả {} (có cảnh báo); dòng JSON hỏng bị bỏ, dòng của bài không có PDF không vào kết quả."""
    assert load_metadata(tmp_path / "khong_co.jsonl", [{"sha256": "a", "paths": ["raw/vista/2020/1.pdf"]}]) == {}
    raw = make_raw(tmp_path)
    out = load_metadata(raw / "metadata/vista_detail.jsonl", [{"sha256": "s1", "paths": ["raw/vista/2020/202.pdf"]}, {"sha256": "s2", "paths": ["raw/vista/2020/5.pdf"]}])
    assert set(out) == {"s1"} and out["s1"]["publication_id"] == "202" and "abstract_vi" not in out["s1"]


def test_dang_ky_profile_run_all_va_dau_vao_van_tay(tmp_path):
    """Registry trỏ raw/vista, profile như VJOL, run_all có phần vista sau vjol, ingest có nhánh vista và vân tay đầu vào gồm checkpoint + metadata."""
    assert SOURCES["vista"].path == "raw/vista" and SOURCES["vista"].fmt == "vista" and SOURCES["vista"].rights_status == "unknown"
    assert PROFILES["vista"] == PROFILES["vjol"]
    assert PARTS["vista"] == ("raw/vista", "scripts/vista/extract.py")
    make_raw(tmp_path)
    assert run_cli("--data-root", str(tmp_path), "--no-ocr", "--no-export").returncode == 0
    files = input_files(SOURCES["vista"], tmp_path, chunked=True)
    assert len([f for f in files if f.parent.name == "vista_text"]) == 4 and tmp_path / "raw/vista/metadata/vista_detail.jsonl" in files
    assert len(list(iter_book_source(SOURCES["vista"], tmp_path))) == 4


def test_pipeline_chay_het_voi_nguon_vista(tmp_path):
    """Pipeline ingest -> finalize chạy với mix vista, bản ghi giữ source_key vista và meta (tiêu đề, tạp chí) tới tận cuối."""
    make_raw(tmp_path)
    assert run_cli("--data-root", str(tmp_path), "--no-ocr", "--no-export").returncode == 0
    cfg = make_cfg(mix={"vista": 4}, chunk_target_tokens=400, chunk_max_tokens=600, chunk_min_tokens=80, embed_spec=None)
    run = tmp_path / "processed/pipeline_runs/t"
    run_pipeline(SOURCES, tmp_path, run, cfg)
    rows = pio.read_rows(run / "05_dedup.parquet")
    assert rows and {r["source_key"] for r in rows} == {"vista"}
    assert any("Tiêu đề bài" in (r.get("meta") or "") for r in rows)


def test_layout_vista_khac_vjol_o_cho_phai_khac():
    """Thư mục checkpoint / khoá / Parquet / nhóm font của VISTA không trùng VJOL (không đụng checkpoint của lượt full VJOL); producer OCR cũ của VISTA là VJOL cộng thêm OCR Manager / ORPALIS / GdPicture (VS-05)."""
    v = vx.VJOL_LAYOUT
    assert VISTA_LAYOUT.text_dir != v.text_dir and VISTA_LAYOUT.claim_dir != v.claim_dir and VISTA_LAYOUT.out_path != v.out_path
    assert VISTA_LAYOUT.font_groups_rel != v.font_groups_rel and set(vx.OCR_ALL_PRODUCERS) < set(VISTA_LAYOUT.ocr_all_producers)
    assert v.text_dir == "interim/vjol_text" and v.claim_dir == "state/vjol_extract/claims" and v.ocr_all_producers == vx.OCR_ALL_PRODUCERS  # VJOL không đổi


def test_kiem_tra_ban_chep_so_voi_nhat_ky(tmp_path):
    """check_raw báo đúng file thiếu / sai kích thước / sai sha256 / thừa, không tính unknown/; script thoát 1 khi bản chép chưa khớp."""
    import hashlib

    raw = tmp_path / "raw/vista"
    (raw / "metadata").mkdir(parents=True)
    files = {"2019/1.pdf": b"aaa", "2020/2.pdf": b"bbbb", "2020/3.pdf": b"cc", "2020/9.pdf": b"thua"}
    for rel, data in files.items():
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        (raw / rel).write_bytes(data)
    (raw / "unknown").mkdir()
    (raw / "unknown/ab.pdf").write_bytes(b"x")
    sha = lambda d: hashlib.sha256(d).hexdigest()  # noqa: E731
    log = [{"publication_id": "1", "status": "downloaded", "sha256": sha(b"aaa"), "bytes": 3},
           {"publication_id": "2", "status": "downloaded", "sha256": sha(b"bbbb"), "bytes": 5},  # sai kích thước
           {"publication_id": "3", "status": "downloaded", "sha256": sha(b"XX"), "bytes": 2},  # đúng byte, sai sha256
           {"publication_id": "4", "status": "downloaded", "sha256": sha(b"d"), "bytes": 1},  # thiếu file
           {"publication_id": "5", "status": "server_error_502"}]  # lỗi tải: không đòi có file
    (raw / "metadata/pdf_download_log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n{hỏng\n", encoding="utf-8")
    from vi_corpus.vista.raw_check import check_raw

    fast = check_raw(raw)
    assert (fast["logged_ok"], fast["on_disk"], fast["unknown_dir"], fast["missing"], fast["size_mismatch"], fast["extra"]) == (4, 4, 1, 1, 1, 1)
    assert fast["sha_mismatch"] is None and fast["examples"]["missing"] == ["4"] and fast["examples"]["extra"] == ["9"]
    assert fast["per_year"] == {"2019": 1, "2020": 3} and fast["log_statuses"] == {"downloaded": 4, "server_error_502": 1}
    assert check_raw(raw, verify_sha=True)["examples"]["sha_mismatch"] == ["3"]
    r = subprocess.run([sys.executable, str(REPO / "scripts/vista/check_raw.py"), "--data-root", str(tmp_path)], capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 1 and '"missing": 1' in r.stdout


def test_producer_ocr_cu_cua_vista_ca_bai_di_ocr_va_doi_danh_sach_thi_lam_lai(tmp_path):
    """VS-05: PDF có producer OCR Manager / ORPALIS / GdPicture (chữ trông bình thường) thì mọi trang đi OCR cả bài; producer hiện đại thì không; danh sách đổi thì extract_version đổi (VJOL không đổi)."""
    import pymupdf

    from tests.test_garbled import mixed_pdf, ocr_engine
    from vi_corpus.common import garbled as g
    from vi_corpus.common.pdf_text import extract_pdf
    from vi_corpus.vista.layout import OCR_ALL_PRODUCERS

    def build(name: str, producer: str, creator: str = "") -> Path:
        """PDF 3 trang chữ tốt với producer / creator cho trước."""
        pdf = tmp_path / name
        mixed_pdf(pdf, [vi_text(4, k) for k in range(3)])
        with pymupdf.open(pdf) as d:
            d.set_metadata({"producer": producer, "creator": creator})
            d.saveIncr()
        return pdf

    for producer in ("OCR Manager", "ORPALIS PDF Reducer 2.0.4 Professional", "GdPicture Managed PDF Plugin V14"):
        seen: list = []
        pages = extract_pdf(build(f"{producer[:5]}.pdf", producer), ocr_engine(seen), detect_garbled=True, ocr_all_producers=OCR_ALL_PRODUCERS)
        assert [p["reason"] for p in pages] == [g.REASON_PRODUCER] * 3 and len(seen) == 3, producer
    seen2: list = []
    modern = extract_pdf(build("moi.pdf", "Microsoft® Word 2019", "Microsoft® Word 2019"), ocr_engine(seen2), detect_garbled=True, ocr_all_producers=OCR_ALL_PRODUCERS)
    assert all("reason" not in p for p in modern) and not seen2
    # extract_version: VJOL giữ nguyên, VISTA có hậu tố +op, đổi danh sách thì đổi, chấp nhận đúng phiên bản của mình
    vocab = frozenset({"a"})
    assert "+op" not in vx.version_for(vocab) and "+op" in vx.version_for(vocab, None, (), OCR_ALL_PRODUCERS)
    assert vx.version_for(vocab, None, (), OCR_ALL_PRODUCERS) != vx.version_for(vocab, None, (), OCR_ALL_PRODUCERS + (("X", ""),))
    assert vx.version_is_current(vx.version_for(vocab, None, (), OCR_ALL_PRODUCERS))
    assert vx.version_for(vocab, None, (), OCR_ALL_PRODUCERS) in vx.accepted_versions(vocab, None, (), OCR_ALL_PRODUCERS)
    assert vx.version_for(vocab) not in vx.accepted_versions(vocab, None, (), OCR_ALL_PRODUCERS)  # checkpoint thử không luật producer bị làm lại


def test_khao_sat_producer_tim_producer_mat_dau_chua_phu(tmp_path):
    """producer_survey: producer cũ đã nằm trong luật thì covered, producer lạ có lớp chữ mất dấu thì lên đầu bảng với uncovered_suspect > 0; PDF scan thuần không bị nghi."""
    import pymupdf

    from vi_corpus.common.garbled import page_signals
    from vi_corpus.vista.producer_survey import run_survey

    raw = tmp_path / "raw/vista"
    nodiac = "Nang cao chat luong cua kiem toan noi bo trong cac tong cong ty duoc nhung cho nhung cua cac duoc " * 12
    assert "nodiac" in page_signals(nodiac)  # điều kiện của test: văn bản mất dấu này phải trúng tín hiệu mất dấu

    def build(path: Path, producer: str, text: str | None) -> None:
        """PDF 1 trang có chữ `text` (hoặc trang trắng nếu None) và producer cho trước."""
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = pymupdf.open()
        page = doc.new_page()
        if text:
            page.insert_htmlbox(pymupdf.Rect(30, 30, 565, 810), f"<p style='font-size:8pt'>{text}</p>")
        doc.set_metadata({"producer": producer})
        doc.save(path)
        doc.close()

    build(raw / "2010/1.pdf", "OCR Manager", nodiac)
    build(raw / "2011/2.pdf", "LạLùngOCR 9.0", nodiac)
    build(raw / "2012/3.pdf", "LạLùngOCR 9.0", nodiac + " thêm")
    build(raw / "2013/4.pdf", "Adobe PDF Library 5.0", None)
    res = run_survey(raw, tmp_path, sample=10, workers=1)
    by = {t["producer"]: t for t in res["table"]}
    assert res["pdfs"] == 4 and res["errors"] == 0 and res["no_text"] == 1 and res["suspect"] == 3
    assert by["OCR Manager"]["covered"] is True and by["OCR Manager"]["uncovered_suspect"] == 0
    assert by["LạLùngOCR 9.0"]["covered"] is False and by["LạLùngOCR 9.0"]["uncovered_suspect"] == 2 and res["table"][0]["producer"] == "LạLùngOCR 9.0"
    assert by["Adobe PDF Library 5.0"]["text_share"] == 0.0 and by["Adobe PDF Library 5.0"]["suspect_share"] is None
    assert (tmp_path / "processed/vista_producer_survey/producers.csv").is_file()
