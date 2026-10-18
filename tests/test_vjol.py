"""
Test xử lý VJOL (J-01 ... J-05): chọn mẫu chia đều theo năm, trích text có checkpoint + chạy tiếp + chia việc bằng khoá, OCR trang
scan bằng chuỗi engine giả (song song, trạng thái từng trang, dừng cả lượt khi chuỗi hỏng), làm sạch bài báo (bìa, mục lục, tài liệu
tham khảo), nối metadata theo luồng, nối vào pipeline (ingest -> finalize) và dòng lệnh scripts/vjol/extract.py.
Không cần mạng, GPU hay mô hình (OCR, server vLLM được thay bằng giả).
"""

import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pyarrow.parquet as pq
import pymupdf
import pytest
from PIL import Image

from scripts.run_all import PARTS
from tests.test_ocr_vlm import GOOD, LOOP, make_cascade, ok
from tests.test_pipeline import make_cfg, vi_text
from vi_corpus.common.batch_extract import claim_file, file_sha256
from vi_corpus.common.ocr_cascade import CascadeOcrGetter, CascadePageOcr
from vi_corpus.common.parquet_export import export_parquet
from vi_corpus.common.pdf_text import extract_pdf
from vi_corpus.common.registry import SOURCES
from vi_corpus.common.schema import CORPUS_SCHEMA, lineage_rate
from vi_corpus.common.state import write_json_atomic
from vi_corpus.ocr_bakeoff.cascade import EngineUnavailable, ServerDown
from vi_corpus.pipeline import io as pio
from vi_corpus.pipeline.config import PROFILES
from vi_corpus.pipeline.ingest import input_files, iter_book_source
from vi_corpus.pipeline.runner import run_pipeline
from vi_corpus.vjol import extract as vx
from vi_corpus.vjol.clean import (
    clean_paper_pages,
    drop_front_matter,
    heading_key,
    is_toc_page,
    strip_references,
)
from vi_corpus.vjol.records import iter_records, load_metadata

REPO = Path(__file__).resolve().parents[1]
NAMES = ["alpha", "beta", "gamma", "delta", "epsilon"]  # nội dung mỗi trang khác nhau (clean_pages coi dòng chỉ khác chữ số là tiêu đề chạy)


# --------------------------------------------------------------------------------------------------------------------
# Dựng dữ liệu giả
# --------------------------------------------------------------------------------------------------------------------


def text_pdf(path: Path, pages: list[str]) -> None:
    """PDF có lớp chữ Unicode (tiếng Việt có dấu), mỗi phần tử của `pages` là một trang."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_htmlbox(pymupdf.Rect(30, 30, 565, 810), f"<p style='font-size:9pt'>{text}</p>")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


def scan_pdf(path: Path, n: int) -> None:
    """PDF n trang (n <= 6) chỉ có ảnh (trang quét); trang i là ảnh đơn sắc mức xám 40*i để engine giả nhận ra số trang."""
    doc = pymupdf.open()
    for i in range(n):
        buf = io.BytesIO()
        Image.new("RGB", (60, 90), (40 * i, 40 * i, 40 * i)).save(buf, format="JPEG", quality=95)
        doc.new_page(width=60, height=90).insert_image(pymupdf.Rect(0, 0, 60, 90), stream=buf.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


def page_no(image: bytes) -> int:
    """Số trang suy từ mức xám của ảnh đơn sắc (xem scan_pdf)."""
    return round(Image.open(io.BytesIO(image)).convert("L").getpixel((5, 5)) / 40)


def make_raw(root: Path, years=(2019, 2020), per_year=3) -> Path:
    """<root>/raw/vjol với per_year PDF text mỗi năm (<article_id>_<galley_id>.pdf), thư mục metadata/ và thư mục ẩn."""
    raw = root / "raw/vjol"
    for y in years:
        for k in range(per_year):
            text_pdf(raw / str(y) / f"{y}{k}_{k + 1}.pdf", [vi_text(6, y * 10 + k)])
    (raw / "metadata").mkdir(parents=True)
    (raw / "metadata/vjol_merged.jsonl").write_text("", encoding="utf-8")
    (raw / ".cache").mkdir()
    (raw / ".cache/an.pdf").write_bytes(b"x")
    return raw


class CountingOcr:
    """Bộ OCR giả kiểu cũ (.page_text từng ảnh): đếm số lần gọi."""

    def __init__(self):
        """Khởi tạo bộ đếm."""
        self.calls = 0

    def page_text(self, image):
        """Trả text giả cho mọi ảnh."""
        self.calls += 1
        return "Nội dung OCR giả"


# --------------------------------------------------------------------------------------------------------------------
# Kiểm kê, chọn mẫu
# --------------------------------------------------------------------------------------------------------------------


def test_kiem_ke_theo_nam_bo_metadata_va_file_an(tmp_path):
    """Chỉ lấy PDF trong thư mục năm; bỏ metadata/, thư mục ẩn; PDF nằm thẳng trong raw có khoá rỗng; năm không có PDF vắng mặt."""
    raw = make_raw(tmp_path, per_year=2)
    (raw / "2021").mkdir()
    (raw / "metadata/ghi_nham.pdf").write_bytes(b"x")
    text_pdf(raw / "ngoai_le.pdf", [vi_text(4, 1)])
    by_year = vx.list_pdfs_by_year(raw)
    assert sorted(by_year) == ["", "2019", "2020"]
    assert [p.name for p in by_year["2019"]] == ["20190_1.pdf", "20191_2.pdf"]
    assert [p.name for p in by_year[""]] == ["ngoai_le.pdf"]
    (raw / "2019/HOA.PDF").write_bytes((raw / "2019/20190_1.pdf").read_bytes())
    assert "HOA.PDF" in [p.name for p in vx.list_pdfs_by_year(raw)["2019"]]  # đuôi viết hoa vẫn là PDF


def test_chon_mau_chia_deu_theo_nam_va_long_nhau():
    """Mẫu chia đều theo năm (năm ít bài nhường chỗ), tái lập theo seed, mẫu nhỏ nằm trong mẫu lớn, vượt tổng thì lấy hết."""
    sizes = {"2001": 10, "2002": 10, "2003": 3, "2004": 1}
    by_year = {y: [Path(f"/raw/{y}/{y}{k}_{k}.pdf") for k in range(n)] for y, n in sizes.items()}

    def per_year(sample):
        """Số bài mỗi năm trong mẫu."""
        return {y: sum(p.parent.name == y for p in sample) for y in sizes}

    s8 = vx.sample_pdfs(by_year, 8, seed=0)
    assert len(s8) == len(set(s8)) == 8
    assert per_year(s8)["2004"] == 1 and per_year(s8)["2003"] == 2 and min(per_year(s8).values()) >= 1
    assert per_year(vx.sample_pdfs(by_year, 12, 0)) == {"2001": 4, "2002": 4, "2003": 3, "2004": 1}  # 2003 chỉ có 3 bài
    assert vx.sample_pdfs(by_year, 8, seed=0) == s8
    assert vx.sample_pdfs(by_year, 8, seed=1) != s8
    assert set(vx.sample_pdfs(by_year, 5, 0)) <= set(s8) <= set(vx.sample_pdfs(by_year, 20, 0))
    assert len(vx.sample_pdfs(by_year, 999, 0)) == 24


def test_mau_khong_phu_thuoc_thu_tu_liet_ke():
    """Cùng tập PDF, liệt kê theo thứ tự khác vẫn ra cùng mẫu (xếp theo băm tên file, không theo thứ tự hệ điều hành)."""
    paths = [Path(f"/raw/2001/{k}_{k}.pdf") for k in range(30)]
    a = vx.sample_pdfs({"2001": paths}, 10, 3)
    b = vx.sample_pdfs({"2001": list(reversed(paths))}, 10, 3)
    assert a == b


def test_path_meta():
    """Năm lấy từ thư mục, article_id / galley_id từ tên file; tên lạ thì thiếu id, thư mục lạ thì không có năm."""
    assert vx.path_meta(Path("raw/vjol/2025/100240_84547.pdf")) == {"year_dir": 2025, "article_id": "100240", "galley_id": "84547"}
    assert vx.path_meta(Path("raw/vjol/abc.pdf")) == {"year_dir": None, "article_id": None, "galley_id": None}


# --------------------------------------------------------------------------------------------------------------------
# Làm sạch bài báo
# --------------------------------------------------------------------------------------------------------------------


def test_heading_key_bo_so_thu_tu_va_dau_cau():
    """Khoá tiêu đề bỏ số thứ tự (số, số La Mã), dấu hai chấm / chấm cuối, hoa thường; câu văn không thành tiêu đề."""
    for line in ("TÀI LIỆU THAM KHẢO", "5. Tài liệu tham khảo", "VI) Tài liệu tham khảo:", "  References.  ", "2.1 References"):
        assert heading_key(line) in ("tài liệu tham khảo", "references"), line
    assert heading_key("Xem tài liệu tham khảo ở cuối bài") != "tài liệu tham khảo"


def test_bo_tai_lieu_tham_khao_giu_tom_tat_tieng_anh_sau_do():
    """Cắt từ tiêu đề tham khảo (tiêu đề cuối cùng) đến hết, trừ phần Abstract / Phụ lục đứng sau; nhắc trong câu văn không bị cắt."""
    body = [f"Đây là câu thân bài số {i} của bài báo khoa học về kinh tế." for i in range(20)]
    pages = ["\n".join(body[:10] + ["Xem tài liệu tham khảo để biết thêm."]),
             "\n".join(body[10:] + ["TÀI LIỆU THAM KHẢO", "[1] Nguyễn Văn A (2019), Kinh tế học, NXB X.", "[2] Smith J. (2020), Economics."]),
             "[3] Trần B (2018), Bài toán Y.\nAbstract\nThis paper studies the economy of Vietnam."]
    out, removed = strip_references(pages)
    joined = "\n".join(out)
    assert removed == 4 and "[1] Nguyễn" not in joined and "[3] Trần" not in joined and "TÀI LIỆU THAM KHẢO" not in joined
    assert "Xem tài liệu tham khảo để biết thêm." in joined and "Abstract\nThis paper studies" in joined
    assert "câu thân bài số 19" in joined
    # không có tiêu đề thì giữ nguyên; tiêu đề ở đầu bài (trước 30% số dòng) không phải phần tham khảo
    assert strip_references(["a\nb\nc"]) == (["a\nb\nc"], 0)
    early = ["References"] + [f"Dòng nội dung {i} của bài." for i in range(20)]
    assert strip_references(["\n".join(early)])[1] == 0


def test_bo_nhieu_muc_tai_lieu_tham_khao_va_giu_phu_luc_abstract_cung_dong():
    """Bài có hai danh sách (Việt, rồi References sau Abstract) bị bỏ cả hai; "Phụ lục 1" và "Abstract: ..." cùng dòng kết thúc phần tham khảo."""
    body = [f"Câu thân bài số {i} bàn về chính sách tiền tệ của ngân hàng trung ương." for i in range(20)]
    pages = ["\n".join(body), "\n".join(["Tài liệu tham khảo", "[1] Nguyễn Văn A (2019), Tiền tệ.",
                                         "Abstract: This paper studies monetary policy.", "Keywords: money",
                                         "References", "[2] Smith J. (2020), Money.",
                                         "PHỤ LỤC 1", "Bảng số liệu phụ lục.", "Tài liệu tham khảo", "[3] Trần B (2018), Lãi suất."])]
    joined = "\n".join(strip_references(pages)[0])
    for gone in ("[1] Nguyễn", "[2] Smith", "[3] Trần"):
        assert gone not in joined, gone
    for kept in ("Abstract: This paper studies", "Keywords: money", "PHỤ LỤC 1", "Bảng số liệu phụ lục", "Câu thân bài số 19"):
        assert kept in joined, kept
    assert heading_key("civil engineering") == "civil engineering" and heading_key("vi. Tham khảo") == "tham khảo"  # La Mã viết thường cần dấu chấm


def test_trang_needs_ocr_khong_lam_lech_trang_bia(tmp_path):
    """Trang scan chưa OCR (needs_ocr) thành trang rỗng chứ không bị xoá khỏi danh sách: trang nội dung đầu tiên (ngắn) không bị coi là bìa."""
    short_but_real = "Tóm tắt nghiên cứu về lạm phát tại Việt Nam trong giai đoạn gần đây."
    body = [f"Câu thân bài số {i} bàn về lạm phát và tăng trưởng của nền kinh tế." for i in range(10)]
    write_ckpt(tmp_path, "d" * 64, "raw/vjol/2019/100_10.pdf", ["", short_but_real, "\n".join(body)])
    ck = vx.checkpoint_path(tmp_path, "d" * 64)
    data = json.loads(ck.read_text(encoding="utf-8"))
    data["pages"][0]["method"] = "needs_ocr"
    ck.write_text(json.dumps(data), encoding="utf-8")
    text = next(iter_records(SOURCES["vjol"], tmp_path))["text"]
    assert short_but_real in text


def test_bo_trang_bia_va_muc_luc():
    """Trang đầu rất ngắn (bài >= 3 trang) là bìa; trang có tiêu đề Mục lục hoặc nhiều dòng chấm dẫn là mục lục; bài ngắn giữ nguyên."""
    toc_leaders = "\n".join(f"Chương {i} ..................... {i * 10}" for i in range(1, 8))
    body = " ".join(["từ"] * 120)
    pages = ["TẠP CHÍ KHOA HỌC\nSố 3 - 2019", "MỤC LỤC\n1. Giới thiệu 3", body, toc_leaders]
    out, st = drop_front_matter(pages)
    assert out[0] == "" and out[1] == "" and out[2] == body and st == {"cover_pages": 1, "toc_pages": 1}
    assert is_toc_page(toc_leaders) and not is_toc_page(body)
    assert drop_front_matter(["Tiêu đề ngắn", body])[1] == {"cover_pages": 0, "toc_pages": 0}  # bài 2 trang: không coi là bìa
    out, st = clean_paper_pages(pages)
    assert st == {"cover_pages": 1, "toc_pages": 1, "reference_lines": 0}


# --------------------------------------------------------------------------------------------------------------------
# Trích text: mẫu, checkpoint, chạy tiếp
# --------------------------------------------------------------------------------------------------------------------


def test_trich_mau_checkpoint_chay_tiep_va_tang_mau(tmp_path):
    """Mẫu 3 bài chỉ tạo 3 checkpoint; tăng lên 5 chỉ làm 2 bài mới; PDF hỏng tính một lỗi và được thử lại; báo cáo đúng số."""
    raw = make_raw(tmp_path)
    calls = []
    assert vx.run_extract(raw, tmp_path, lambda: calls.append(1), sample_docs=3, seed=0) == 0 and not calls
    ckpts = sorted((tmp_path / vx.TEXT_DIR).glob("*.json"))
    assert len(ckpts) == 3
    first = {c: c.stat().st_mtime_ns for c in ckpts}
    done = json.loads(ckpts[0].read_text(encoding="utf-8"))
    assert done["format"] == "pdf" and done["pages"][0]["method"] == "text" and done["article_id"] and done["year_dir"] in (2019, 2020)
    assert done["paths"][0].startswith("raw/vjol/20") and done["seconds"] >= 0

    assert vx.run_extract(raw, tmp_path, lambda: None, sample_docs=5, seed=0) == 0
    after = sorted((tmp_path / vx.TEXT_DIR).glob("*.json"))
    assert len(after) == 5 and all(c.stat().st_mtime_ns == t for c, t in first.items())  # 3 bài cũ không làm lại

    (raw / "2020/hong_1.pdf").write_bytes(b"khong phai pdf")
    assert vx.run_extract(raw, tmp_path, lambda: None, sample_docs=None) == 1  # mọi PDF: 6 bài tốt + 1 hỏng
    rep = vx.report(raw, tmp_path)
    assert (rep["pdfs_in_raw"], rep["checkpoints"], rep["errors"], rep["pages"]) == (7, 7, 1, {"text": 6})
    assert vx.run_extract(raw, tmp_path, lambda: None, sample_docs=None) == 1  # chỉ file lỗi được thử lại


def test_limit_files_va_pdf_khong_phai_pdf(tmp_path):
    """--limit-files chỉ lấy N file đầu; file đuôi .pdf nhưng nội dung là HTML (trang lỗi tải) thì bị bỏ qua chứ không lỗi."""
    raw = make_raw(tmp_path)
    (raw / "2019/99_9.pdf").write_text("<!DOCTYPE html><html><body>404</body></html>", encoding="utf-8")
    assert vx.run_extract(raw, tmp_path, lambda: None, limit_files=2) == 0
    assert len(list((tmp_path / vx.TEXT_DIR).glob("*.json"))) == 2
    assert vx.run_extract(raw, tmp_path, lambda: None) == 0
    assert vx.report(raw, tmp_path)["skipped"] == 1


def test_chia_viec_bang_khoa_giua_cac_tien_trinh(tmp_path):
    """PDF đang do tiến trình khác giữ khoá thì bị bỏ qua (không ghi checkpoint); nhả khoá rồi chạy lại thì làm nốt."""
    raw = make_raw(tmp_path, years=(2019,), per_year=3)
    target = vx.select_files(raw)[1]
    with claim_file(tmp_path, vx.CLAIM_DIR, file_sha256(target)) as got:
        assert got
        assert vx.run_extract(raw, tmp_path, lambda: None) == 0
        assert len(list((tmp_path / vx.TEXT_DIR).glob("*.json"))) == 2  # file bị giữ khoá chưa làm
    assert vx.run_extract(raw, tmp_path, lambda: None) == 0
    assert len(list((tmp_path / vx.TEXT_DIR).glob("*.json"))) == 3


# --------------------------------------------------------------------------------------------------------------------
# OCR trang scan bằng chuỗi engine
# --------------------------------------------------------------------------------------------------------------------


def test_trang_scan_ocr_vlm_ghi_tier_status_va_trang_bi_loai(tmp_path):
    """
    Bài 5 trang scan: trang 3 hỏng ở mọi engine (flagged, text rỗng, giữ rejected_text), trang 4 trắng. Kết quả đúng thứ tự trang,
    ghi method ocr_vlm + tier + status; báo cáo đếm đúng; bản ghi không chứa trang bị loại.
    """
    raw = tmp_path / "raw/vjol"
    scan_pdf(raw / "2019/7_1.pdf", 5)

    def engine(im):
        """Engine giả: trang 3 lặp (hỏng), trang 4 trắng, còn lại là câu có số trang."""
        n = page_no(im)
        return ok(LOOP if n == 3 else ("" if n == 4 else f"Nội dung {NAMES[n]} của bài báo khoa học. " * 6))

    cascade = make_cascade(engine, engine)
    assert vx.run_extract(raw, tmp_path, lambda: CascadePageOcr(cascade, concurrency=3)) == 0
    pages = json.loads(next((tmp_path / vx.TEXT_DIR).glob("*.json")).read_text(encoding="utf-8"))["pages"]
    assert [p["method"] for p in pages] == ["ocr_vlm"] * 5
    assert [p["status"] for p in pages] == ["ok", "ok", "ok", "flagged", "blank"]
    assert pages[0]["text"].startswith("Nội dung alpha") and pages[2]["text"].startswith("Nội dung gamma")
    assert pages[3]["text"] == "" and pages[3]["rejected_text"] == LOOP.strip() and pages[0]["tier"] == "t0"
    rep = vx.report(raw, tmp_path)
    assert rep["ocr_status"] == {"ok": 3, "flagged": 1, "blank": 1} and rep["flagged_rate"] == pytest.approx(0.2)
    assert rep["ocr_tier"] == {"t0": 3}
    rec = next(iter_records(SOURCES["vjol"], tmp_path))
    assert "Nội dung alpha" in rec["text"] and LOOP.strip() not in rec["text"]


def test_ocr_gui_song_song_va_giu_thu_tu(tmp_path):
    """Các trang scan của một PDF được gửi đồng thời (nhiều hơn 1 yêu cầu cùng lúc) mà kết quả vẫn đúng thứ tự trang."""
    pdf = tmp_path / "scan.pdf"
    scan_pdf(pdf, 6)
    lock, state = threading.Lock(), {"now": 0, "max": 0}

    def engine(im):
        """Engine giả chậm, đo số yêu cầu đồng thời."""
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return ok(f"Trang số {page_no(im)}. " + GOOD)

    pages = extract_pdf(pdf, lambda: CascadePageOcr(make_cascade(engine), concurrency=4))
    assert [p["text"].split(".")[0] for p in pages] == [f"Trang số {i}" for i in range(6)]
    assert state["max"] >= 2


def test_ocr_cu_kieu_page_text_van_dung_duoc(tmp_path):
    """Bộ OCR chỉ có .page_text (kiểu cũ) vẫn dùng được: trang scan OCR tuần tự, method ocr."""
    pdf = tmp_path / "scan.pdf"
    scan_pdf(pdf, 2)
    ocr = CountingOcr()
    assert [p["method"] for p in extract_pdf(pdf, lambda: ocr)] == ["ocr", "ocr"] and ocr.calls == 2


def test_tat_ocr_cho_needs_ocr_roi_bat_ocr_thi_lam_lai(tmp_path):
    """--no-ocr: trang scan thành needs_ocr (không vào bản ghi); chạy lại có OCR thì làm lại đúng file đó."""
    raw = tmp_path / "raw/vjol"
    scan_pdf(raw / "2019/7_1.pdf", 2)
    assert vx.run_extract(raw, tmp_path, lambda: None, ocr_enabled=False) == 0
    assert vx.report(raw, tmp_path)["pages"] == {"needs_ocr": 2}
    assert list(iter_records(SOURCES["vjol"], tmp_path)) == []
    cascade = make_cascade(lambda im: ok(f"Trang {page_no(im)} " + GOOD))
    assert vx.run_extract(raw, tmp_path, lambda: CascadePageOcr(cascade, 2)) == 0
    assert vx.report(raw, tmp_path)["pages"] == {"ocr_vlm": 2}


def test_chuoi_ocr_hong_dung_ca_luot_khong_ghi_file_do(tmp_path):
    """ServerDown / EngineUnavailable không phải lỗi của file: dừng cả lượt, file đang làm không có checkpoint, file xong trước vẫn còn."""
    raw = make_raw(tmp_path, years=(2019,), per_year=1)
    scan_pdf(raw / "2020/5_5.pdf", 2)  # xếp sau bài 2019 (theo năm)
    for exc in (EngineUnavailable("thiếu libGL"), ServerDown("server chết")):
        def get_ocr(exc=exc):
            """Bộ OCR giả luôn hỏng."""
            raise exc

        with pytest.raises(type(exc)):
            vx.run_extract(raw, tmp_path, get_ocr)
        rep = vx.report(raw, tmp_path)
        assert rep["checkpoints"] == 1 and rep["errors"] == 0  # bài text đã xong, bài scan không bị ghi thành lỗi


def test_getter_nap_luoi_nho_loi_va_tat_server(tmp_path, monkeypatch):
    """CascadeOcrGetter: tắt OCR thì trả None; dựng một lần ở lần gọi đầu; lỗi dựng thành EngineUnavailable và không dựng lại; close tắt server."""
    from vi_corpus.ocr_bakeoff import engines

    events = []
    monkeypatch.setattr(engines, "start_server", lambda cfg, run_dir, gpu, vllm_bin: events.append(("start", gpu, vllm_bin)) or "PROC")
    monkeypatch.setattr(engines, "stop_server", lambda proc: events.append(("stop", proc)))
    monkeypatch.setattr(engines, "wait_ready", lambda url, health, timeout, proc: events.append(("ready", url, proc)))
    tiers = make_cascade(lambda im: ok()).tiers
    tiers[0].cfg.update({"base_url": "http://127.0.0.1:1/v1", "serve": ["vllm", "serve", "M"]})

    assert CascadeOcrGetter(tiers, enabled=False)() is None and not events
    with pytest.raises(ValueError, match="state_dir"):
        CascadeOcrGetter(tiers, serve=True)
    getter = CascadeOcrGetter(tiers, 7, serve=True, gpu="2", vllm_bin="/x/vllm", state_dir=tmp_path / "w")
    assert events == []  # chưa gặp trang scan thì chưa dựng gì
    ocr = getter()
    assert getter() is ocr and ocr.concurrency == 7
    assert events == [("start", "2", "/x/vllm"), ("ready", "http://127.0.0.1:1/v1", "PROC")]
    getter.close()
    getter.close()
    assert events[-1] == ("stop", "PROC") and sum(e[0] == "stop" for e in events) == 1

    def boom(*a):
        """wait_ready giả: server không bao giờ sẵn sàng."""
        events.append("wait")
        raise RuntimeError("quá thời gian")

    monkeypatch.setattr(engines, "wait_ready", boom)
    bad = CascadeOcrGetter(tiers)
    for _ in range(2):
        with pytest.raises(EngineUnavailable, match="quá thời gian"):
            bad()
    assert events.count("wait") == 1  # lỗi được nhớ, không chờ lại cho từng file


# --------------------------------------------------------------------------------------------------------------------
# Bản ghi, metadata, pipeline
# --------------------------------------------------------------------------------------------------------------------


def write_ckpt(root: Path, sha: str, rel: str, pages: list[str], **kw) -> None:
    """Ghi một checkpoint trích text giả cho PDF `rel` (mọi trang method text)."""
    stem = Path(rel).stem
    aid, gid = stem.split("_")
    write_json_atomic(vx.checkpoint_path(root, sha), {
        "sha256": sha, "size": 1, "paths": [rel], "year_dir": int(Path(rel).parent.name), "article_id": aid, "galley_id": gid,
        "format": "pdf", "pages": [{"text": t, "method": "text"} for t in pages], "error": None, "skipped": None, **kw})


def test_nap_metadata_theo_luong_chi_giu_bai_trong_mau(tmp_path):
    """Metadata đọc một lượt: giữ bài có trong mẫu, bỏ dòng JSON hỏng, đánh dấu galley khớp / không khớp, thiếu file thì {}."""
    meta = tmp_path / "vjol_merged.jsonl"
    rows = [{"article_id": 100, "title": "Bài một", "journal_slug": "tcvn", "journal_title": "Tạp chí A", "year": 2019, "language": "vi",
             "doi": "10.1/x", "keywords": ["kinh tế"], "abstract": "rất dài", "galleys": [{"galley_id": 10}, {"galley_id": 11}]},
            {"article_id": "101", "title": "Bài hai", "year": 2020, "galleys": [{"galley_id": 5}]},
            {"article_id": 999, "title": "Không có trong mẫu"}]
    meta.write_text("\n".join([json.dumps(rows[0], ensure_ascii=False), "{hỏng", "", json.dumps(rows[1]), json.dumps(rows[2])]), encoding="utf-8")
    got = load_metadata(meta, {"100_10", "101_99", "102_5", "tenla"})
    assert sorted(got) == ["100_10", "101_99"]
    assert got["100_10"]["title"] == "Bài một" and got["100_10"]["galley_match"] is True and "abstract" not in got["100_10"]
    assert got["101_99"]["title"] == "Bài hai" and got["101_99"]["galley_match"] is False and got["101_99"]["doi"] is None
    assert load_metadata(tmp_path / "khong_co.jsonl", {"1_1"}) == {}


def test_iter_records_lam_sach_noi_metadata_va_xuat_parquet(tmp_path):
    """Bản ghi bỏ bìa / mục lục / tài liệu tham khảo, giữ tóm tắt tiếng Anh; meta có tạp chí, năm, số trang bị bỏ; lineage 100%; xuất Parquet được."""
    body = [f"Phân tích kinh tế vĩ mô cho thấy lạm phát năm {2000 + i} chịu tác động của nhiều yếu tố khác nhau trong nước." for i in range(24)]
    pages = ["TẠP CHÍ KINH TẾ\nSố 3 năm 2019", "\n".join(body[:12]), "\n".join(body[12:] + ["Tài liệu tham khảo", "[1] Nguyễn Văn A (2019), Kinh tế vĩ mô."]),
             "Abstract\nThis paper studies inflation in Vietnam over two decades."]
    write_ckpt(tmp_path, "a" * 64, "raw/vjol/2019/100_10.pdf", pages)
    write_ckpt(tmp_path, "b" * 64, "raw/vjol/2020/200_20.pdf", ["Chỉ có một trang nên không bị coi là bìa. " * 3])
    write_json_atomic(vx.checkpoint_path(tmp_path, "c" * 64), {"sha256": "c" * 64, "paths": ["raw/vjol/2020/300_30.pdf"], "pages": [], "error": "hỏng", "skipped": None})
    meta = tmp_path / "raw/vjol/metadata/vjol_merged.jsonl"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"article_id": 100, "title": "Lạm phát", "journal_slug": "kt", "journal_title": "Tạp chí Kinh tế", "year": 2019,
                                "language": "vi", "doi": "10.1/ab", "keywords": ["lạm phát"], "galleys": [{"galley_id": 10}]}, ensure_ascii=False) + "\n",
                    encoding="utf-8")

    recs = list(iter_records(SOURCES["vjol"], tmp_path, with_meta=True))
    assert len(recs) == 2 and all(set(CORPUS_SCHEMA.names) <= set(r) for r in recs)  # checkpoint lỗi không thành bản ghi
    rec = next(r for r in recs if r["source_path"].endswith("100_10.pdf"))
    assert "TẠP CHÍ KINH TẾ" not in rec["text"] and "Tài liệu tham khảo" not in rec["text"] and "[1] Nguyễn" not in rec["text"]
    assert "lạm phát năm 2023" in rec["text"] and "This paper studies inflation" in rec["text"]
    md = json.loads(rec["meta"])
    assert (md["journal_title"], md["year"], md["doi"], md["title"], md["article_id"], md["galley_id"]) == ("Tạp chí Kinh tế", 2019, "10.1/ab", "Lạm phát", "100", "10")
    assert md["metadata_found"] and md["galley_match"] and md["cover_pages"] == 1 and md["reference_lines"] == 2 and md["pages"] == 4
    other = json.loads(next(r for r in recs if r["source_path"].endswith("200_20.pdf"))["meta"])
    assert other["metadata_found"] is False and other["journal_title"] is None and other["year"] == 2020  # thiếu metadata: lấy năm từ thư mục
    assert rec["rights_status"] == "unknown" and rec["source_key"] == "vjol" and lineage_rate(recs) == 1.0
    assert "meta" not in next(iter_records(SOURCES["vjol"], tmp_path))  # không with_meta thì không đọc metadata
    out = tmp_path / "processed/vjol/vjol.parquet"
    assert export_parquet(iter_records(SOURCES["vjol"], tmp_path), out) == 2 and pq.read_table(out).num_rows == 2


def test_dang_ky_nguon_profile_va_dong_run_all():
    """Registry trỏ raw/vjol (chữ thường), profile VJOL giữ cả tiếng Việt và tiếng Anh, run_all có phần vjol sau giáo trình."""
    assert SOURCES["vjol"].path == "raw/vjol" and SOURCES["vjol"].rights_status == "unknown"
    prof = PROFILES["vjol"]
    assert prof.chunked and prof.min_words == 80 and prof.allowed_langs == ("vi", "en") and prof.min_lang_score == 0.5
    assert list(PARTS)[-1] == "vjol" and PARTS["vjol"] == ("raw/vjol", "scripts/vjol/extract.py")


def test_ingest_nhanh_vjol_va_dau_vao_van_tay(tmp_path):
    """iter_book_source đọc checkpoint VJOL (có meta), input_files liệt kê checkpoint + file metadata (đổi metadata thì vân tay đầu vào đổi)."""
    write_ckpt(tmp_path, "a" * 64, "raw/vjol/2019/100_10.pdf", [vi_text(6, 1)])
    assert input_files(SOURCES["vjol"], tmp_path, chunked=True) == [vx.checkpoint_path(tmp_path, "a" * 64)]
    meta = tmp_path / "raw/vjol/metadata/vjol_merged.jsonl"
    meta.parent.mkdir(parents=True)
    meta.write_text("", encoding="utf-8")
    assert input_files(SOURCES["vjol"], tmp_path, chunked=True)[-1] == meta
    rows = list(iter_book_source(SOURCES["vjol"], tmp_path))
    assert len(rows) == 1 and "meta" in rows[0]


def test_pipeline_end_to_end_voi_vjol(tmp_path):
    """
    PDF thật -> trích text -> pipeline (ingest ... finalize) chỉ với nguồn vjol: lấy mẫu xuống đúng số đoạn, nối metadata vào bản ghi,
    phần tài liệu tham khảo không lọt vào kho, lineage 100%, chạy lại bỏ qua stage.
    """
    raw = tmp_path / "raw/vjol"
    meta_lines = []
    for k in range(6):
        body = [vi_text(10, 100 * k + j) for j in range(3)]
        text_pdf(raw / f"2019/{k}1_1.pdf", [body[0], body[1], body[2] + "</p><p>Tài liệu tham khảo</p><p>[1] Nguyễn Văn A (2019), Sách tham khảo số 1."])
        meta_lines.append(json.dumps({"article_id": f"{k}1", "title": f"Bài {k}", "journal_title": "Tạp chí thử", "year": 2019, "language": "vi",
                                      "galleys": [{"galley_id": 1}]}, ensure_ascii=False))
    (raw / "metadata").mkdir()
    (raw / "metadata/vjol_merged.jsonl").write_text("\n".join(meta_lines), encoding="utf-8")
    assert vx.run_extract(raw, tmp_path, lambda: None, sample_docs=6) == 0

    cfg = make_cfg(mix={"vjol": 8}, chunk_target_tokens=200, chunk_max_tokens=300, chunk_min_tokens=80, embed_spec=None)
    run = tmp_path / "run"
    run_pipeline(SOURCES, tmp_path, run, cfg)
    ingest_rows = pio.read_rows(run / "01_ingest.parquet")
    assert len(ingest_rows) == 6 and all("Sách tham khảo số 1" not in r["text"] for r in ingest_rows)
    assert json.loads(ingest_rows[0]["meta"])["journal_title"] == "Tạp chí thử"
    rows = pio.read_rows(run / "05_dedup.parquet")
    assert len(rows) == 8 and {r["source_key"] for r in rows} == {"vjol"}  # 6 bài cắt đoạn rồi lấy mẫu xuống 8 đoạn
    assert {r["language"] for r in rows} == {"vi"} and json.loads(rows[0]["meta"])["journal_title"] == "Tạp chí thử"
    assert json.loads((run / "audit.json").read_text(encoding="utf-8"))["lineage_rate_all"] == 1.0
    first = (run / "05_dedup.parquet").stat().st_mtime_ns
    run_pipeline(SOURCES, tmp_path, run, cfg)
    assert (run / "05_dedup.parquet").stat().st_mtime_ns == first


# --------------------------------------------------------------------------------------------------------------------
# Dòng lệnh
# --------------------------------------------------------------------------------------------------------------------


def run_cli(*args) -> subprocess.CompletedProcess:
    """Chạy scripts/vjol/extract.py như người dùng gõ lệnh (tiến trình riêng)."""
    return subprocess.run([sys.executable, str(REPO / "scripts/vjol/extract.py"), *args], capture_output=True, text=True, timeout=300, check=False)


def test_dong_lenh_trich_mau_xuat_parquet_va_status(tmp_path):
    """E2E dòng lệnh: --sample-docs --no-ocr trích đúng số bài và xuất Parquet; chạy lại không làm lại; --status và --export-only chạy được."""
    make_raw(tmp_path)
    r = run_cli("--data-root", str(tmp_path), "--sample-docs", "4", "--no-ocr")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "4 đã trích text" in r.stdout and "text=4" in r.stdout
    assert pq.read_table(tmp_path / "processed/vjol/vjol.parquet").num_rows == 4
    before = {c: c.stat().st_mtime_ns for c in (tmp_path / vx.TEXT_DIR).glob("*.json")}
    r = run_cli("--data-root", str(tmp_path), "--sample-docs", "6", "--no-ocr", "--no-export")
    assert r.returncode == 0 and "6 đã trích text" in r.stdout
    assert all(c.stat().st_mtime_ns == t for c, t in before.items())
    assert "6 đã trích text" in run_cli("--data-root", str(tmp_path), "--status").stdout
    assert run_cli("--data-root", str(tmp_path), "--export-only").returncode == 0
    assert run_cli("--data-root", str(tmp_path / "khong_co"), "--status").returncode == 2  # thiếu raw/vjol


def test_muc_tham_khao_bat_dau_bang_tu_khoa_ket_thuc_khong_cat_som():
    """(reviewer lần 2) Một mục tài liệu tham khảo bắt đầu bằng "Summary" / "Abstract algebra" / "Phụ lục Nghị định" không được coi là tiêu đề kết thúc."""
    body = [f"Câu thân bài số {i} bàn về chính sách tiền tệ của ngân hàng trung ương." for i in range(20)]
    entries = ["[1] Nguyễn A (2019), Tiền tệ.", "Summary of the economic report, World Bank (2020).",
               "Abstract algebra, Dummit D. (2004).", "Phụ lục Nghị định 12/2019/NĐ-CP, Chính phủ.", "Smith J (2000), Money."]
    out, dropped = strip_references(["\n".join(body + ["Tài liệu tham khảo"] + entries)])
    assert dropped == len(entries) + 1
    for e in entries:
        assert e not in out[0], e
