"""
Test script đếm số file và số trang cần OCR (vi_corpus/vjol/ocr_count.py, scripts/count_ocr_pages.py).

Dữ liệu giả dựng bằng PDF nhỏ (có lớp chữ và trang scan), không cần GPU / mạng. Kiểm: số liệu khớp với `report()` của extract.py (nguồn sự thật cũ, chậm), PDF chưa quét được
báo riêng rồi `--scan-missing` quét nốt ra số thật, bộ nhớ đệm chỉ đọc lại checkpoint đổi, hai file cùng nội dung là 2 file nhưng 1 lần trang, và phần ước theo năm.
"""

import json
import subprocess
import sys
from pathlib import Path

from tests.test_vista import make_raw
from tests.test_vjol import scan_pdf
from vi_corpus.vista.layout import VISTA_LAYOUT
from vi_corpus.vjol import extract as vx
from vi_corpus.vjol.ocr_count import aggregate, count_source, list_pdfs

REPO = Path(__file__).resolve().parents[1]


def make_vista(root: Path) -> Path:
    """Raw VISTA giả: 5 PDF của make_raw (2 file cùng nội dung) + 2020/204.pdf là PDF scan 3 trang; trả về thư mục raw/vista."""
    raw = make_raw(root)
    scan_pdf(raw / "2020/204.pdf", 3)
    return raw


def run_count(root: Path, *args) -> subprocess.CompletedProcess:
    """Chạy scripts/count_ocr_pages.py như người dùng gõ lệnh (tiến trình riêng)."""
    return subprocess.run([sys.executable, str(REPO / "scripts/count_ocr_pages.py"), "--data-root", str(root), "--workers", "2", *args],
                          capture_output=True, text=True, timeout=300, check=False)


def test_kiem_ke_pdf_theo_nam_bo_metadata_va_thu_muc_an(tmp_path):
    """list_pdfs gom theo thư mục năm, thư mục không phải năm (unknown/) vào nhãn khác, bỏ metadata/ và file ẩn."""
    raw = make_vista(tmp_path)
    (raw / ".cache").mkdir()
    scan_pdf(raw / ".cache/x.pdf", 1)
    listed = list_pdfs(raw, tmp_path, workers=2)
    assert {k: len(v) for k, v in listed.items()} == {"2019": 1, "2020": 4, "(khác)": 1}
    assert "raw/vista/2020/204.pdf" in listed["2020"]


def test_chua_quet_roi_scan_missing_ra_so_that_khop_report(tmp_path):
    """E2E: lúc đầu mọi PDF chưa quét (checkpoint chưa có); --scan-missing 2 quét nốt rồi số trang cần OCR bằng đúng số của `report()` cũ (3 trang scan)."""
    make_vista(tmp_path)
    r = run_count(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    before = json.loads((tmp_path / "processed/ocr_count/vista.json").read_text(encoding="utf-8"))["total"]
    assert before["files"] == 6 and before["missing_files"] == 6 and before["ocr_pages"] == 0
    r = run_count(tmp_path, "--sources", "vista", "--scan-missing", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    t = json.loads((tmp_path / "processed/ocr_count/vista.json").read_text(encoding="utf-8"))["total"]
    old = vx.report(tmp_path / "raw/vista", tmp_path, VISTA_LAYOUT)
    assert t["files"] == 6 and t["missing_files"] == 0 and t["scanned_files"] == 6
    assert t["contents"] == old["checkpoints"] == 5  # 2 file cùng nội dung chỉ một checkpoint
    assert t["pages"] == sum(old["pages"].values()) == 7  # 4 PDF chữ một trang + PDF scan 3 trang
    assert t["methods"] == old["pages"]
    assert t["ocr_pending"] == old["pages"]["needs_ocr"] == t["ocr_pages"] == 3 and t["ocr_done"] == 0
    assert "Trang CẦN OCR (đã quét)   : 3" in r.stdout
    assert (tmp_path / "processed/ocr_count/vista_by_year.csv").read_text(encoding="utf-8").splitlines()[0].startswith("year,files,scanned_files")


def test_bo_nho_dem_chi_doc_lai_checkpoint_doi(tmp_path):
    """Lần đếm thứ hai lấy hết từ bộ nhớ đệm; sửa một checkpoint (đổi kích thước) thì chỉ file đó được đọc lại và số liệu đổi theo."""
    make_vista(tmp_path)
    assert run_count(tmp_path, "--sources", "vista", "--scan-missing", "1").returncode == 0
    first = count_source("vista", tmp_path, workers=2)
    second = count_source("vista", tmp_path, workers=2)
    assert first["cache_hits"] == 5 and second["cache_hits"] == 5 and second["total"] == first["total"]
    ckpt = next(p for p in (tmp_path / VISTA_LAYOUT.text_dir).glob("*.json") if any(pg["method"] == "needs_ocr" for pg in json.loads(p.read_text(encoding="utf-8"))["pages"]))
    data = json.loads(ckpt.read_text(encoding="utf-8"))
    data["pages"][0]["method"] = "ocr_vlm"
    data["pages"][0]["status"] = "ok"
    ckpt.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    third = count_source("vista", tmp_path, workers=2)
    assert third["cache_hits"] == 4 and third["total"]["ocr_done"] == 1 and third["total"]["ocr_pending"] == 2 and third["total"]["vlm"] == {"ok": 1}
    assert count_source("vista", tmp_path, workers=2, use_cache=False)["cache_hits"] == 0


def test_checkpoint_hong_hoac_loi_khong_lam_hong_so_liem(tmp_path):
    """Checkpoint bị cắt dở được đếm vào 'unreadable', checkpoint lỗi không cộng trang và PDF của nó vẫn tính là chưa quét."""
    make_vista(tmp_path)
    assert run_count(tmp_path, "--sources", "vista", "--scan-missing", "1").returncode == 0
    ckpts = sorted((tmp_path / VISTA_LAYOUT.text_dir).glob("*.json"))
    ckpts[0].write_text('{"pages": [', encoding="utf-8")
    bad = json.loads(ckpts[1].read_text(encoding="utf-8"))
    bad["error"] = "Lỗi giả"
    ckpts[1].write_text(json.dumps(bad), encoding="utf-8")
    r = count_source("vista", tmp_path, workers=2, use_cache=False)
    assert r["unreadable"] == 1 and r["total"]["errors"] == 1
    assert r["total"]["missing_files"] >= 2  # PDF của checkpoint hỏng và của checkpoint lỗi đều chưa được coi là đã quét


def test_uoc_phan_chua_quet_theo_tung_nam():
    """Năm 2020: 2 file đã quét có 10 trang cần OCR -> 5 trang/file; 4 file chưa quét ước thêm 20 trang. Năm không có file đã quét thì không ước."""
    def entry(year, paths, ocr, text=0):
        return {"pages": ocr + text, "methods": {"needs_ocr": ocr, "text": text}, "vlm": {}, "error": False, "skipped": False, "current": True, "year": year, "paths": paths}
    entries = {"a.json": entry("2020", ["r/2020/1.pdf"], 4, 1), "b.json": entry("2020", ["r/2020/2.pdf"], 6)}
    listed = {"2020": [f"r/2020/{i}.pdf" for i in range(1, 7)], "1999": ["r/1999/1.pdf"]}
    agg = aggregate(entries, listed)
    y2020, y1999 = agg["by_year"]["2020"], agg["by_year"]["1999"]
    assert y2020["scanned_files"] == 2 and y2020["missing_files"] == 4 and y2020["ocr_pages"] == 10 and y2020["est_missing_ocr_pages"] == 20
    assert y1999["scanned_files"] == 0 and y1999["est_missing_ocr_pages"] == 0
    assert agg["total"]["files"] == 7 and agg["total"]["missing_files"] == 5 and agg["total"]["est_total_ocr_pages"] == 30 and agg["total"]["pages"] == 11
