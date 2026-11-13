"""
Test scripts/ocr_internal/trial_vjol_vista.py không cần mạng: PDF dựng bằng PyMuPDF (1 lớp, 2 lớp, chỉ chữ) và một máy chủ giả
nói đúng giao thức 3 bước của API OCR nội bộ (upload / tạo phiên / hỏi kết quả có link ký sẵn).
"""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pymupdf
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ocr_internal" / "trial_vjol_vista.py"
spec = importlib.util.spec_from_file_location("trial_vjol_vista", SCRIPT)
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)

TEXT = "Đây là một đoạn văn tiếng Việt đủ dài để được coi là có lớp chữ trong PDF thử nghiệm này."


def _page_png(doc_text: str) -> bytes:
    """Ảnh PNG của một trang có chữ (giả ảnh scan)."""
    d = pymupdf.open()
    d.new_page().insert_text((72, 100), doc_text, fontsize=14)
    return d[0].get_pixmap(dpi=60).tobytes("png")


def make_pdf(path: Path, kind: str, pages: int = 2) -> None:
    """Dựng PDF thử: kind = scan (chỉ ảnh), image_text (ảnh + chữ vô hình), text_only (chỉ chữ)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    png = _page_png(f"{path.stem} nội dung") if kind != "text_only" else b""
    for i in range(pages):
        page = doc.new_page()
        if kind != "text_only":
            page.insert_image(page.rect, stream=png)
        if kind != "scan":
            page.insert_textbox(pymupdf.Rect(50, 50, 500, 300), f"{TEXT} {path.stem} trang {i}", fontsize=11, render_mode=3 if kind == "image_text" else 0)
    doc.save(path)


class FakeApi(BaseHTTPRequestHandler):
    """Máy chủ giả: addFile -> hash, gen-ai -> session_id, result lần 1 'đang xử lý', lần 2 trả link tới /result.json, cancel -> success."""

    polls: dict = {}
    seen_headers: list = []

    def log_message(self, *a):  # im lặng
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # link kết quả ký sẵn: nhận được header Authorization là lỗi (lộ khoá)
        if self.headers.get("Authorization") or self.headers.get("Token-key"):
            return self._send({"leak": True}, 400)
        self._send({"object": {"paragraphs": [{"type": "List", "cells": [
            {"text": TEXT, "page_id": 1, "type": "Paragraph"}, {"text": "bảng", "page_id": 2, "type": "Table"}]}]}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        FakeApi.seen_headers.append((self.path, self.headers.get("Authorization"), self.headers.get("Token-id"), self.headers.get("Token-key")))
        ok = {"status": "OK", "statusCode": 200, "message": "IDG-00000000"}
        if self.path.endswith("/addFile"):
            assert b"%PDF" in raw
            return self._send({**ok, "object": {"hash": "idg/abc"}})
        body = json.loads(raw)
        if self.path.endswith("/gen-ai"):
            assert body["file_hash"] == "idg/abc" and body["file_type"] == "pdf" and body["exporter"] == "json"
            return self._send({**ok, "object": {"session_id": f"s{len(FakeApi.polls)}"}})
        if self.path.endswith("/result"):
            sid = body["session_id"]
            FakeApi.polls[sid] = FakeApi.polls.get(sid, 0) + 1
            if FakeApi.polls[sid] < 2:
                return self._send({**ok, "object": {"warning": [trial.PENDING_WARNING], "total_page_num": 2}})
            link = f"http://127.0.0.1:{self.server.server_port}/result.json"
            return self._send({**ok, "object": {"link": link, "num_of_pages": 2, "total_page_num": 2}})
        if self.path.endswith("/cancel"):
            return self._send({**ok, "object": {"status": "success"}})
        self._send({}, 404)


@pytest.fixture
def server():
    """Chạy FakeApi ở cổng ngẫu nhiên; trả địa chỉ gốc."""
    FakeApi.polls, FakeApi.seen_headers = {}, []
    srv = HTTPServer(("127.0.0.1", 0), FakeApi)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture
def data_root(tmp_path):
    """data/raw/{vjol,vista} mỗi nguồn 3 PDF scan, 3 PDF ảnh+chữ, 3 PDF chỉ chữ, thêm một file metadata không được lấy."""
    for src in ("vjol", "vista"):
        for i in range(3):
            make_pdf(tmp_path / "raw" / src / "2020" / f"s{i}.pdf", "scan")
            make_pdf(tmp_path / "raw" / src / "2021" / f"i{i}.pdf", "image_text")
            make_pdf(tmp_path / "raw" / src / "2022" / f"t{i}.pdf", "text_only")
        (tmp_path / "raw" / src / "metadata").mkdir()
        (tmp_path / "raw" / src / "metadata" / "x.pdf").write_bytes(b"not a pdf")
    return tmp_path


def test_classify(tmp_path):
    """Ba dạng PDF được nhận đúng; file hỏng cho None."""
    make_pdf(tmp_path / "a.pdf", "scan")
    make_pdf(tmp_path / "b.pdf", "image_text")
    make_pdf(tmp_path / "c.pdf", "text_only")
    (tmp_path / "d.pdf").write_bytes(b"junk")
    assert trial.classify_pdf(tmp_path / "a.pdf")["layer"] == 1
    assert trial.classify_pdf(tmp_path / "b.pdf")["layer2_kind"] == "image_text"
    assert trial.classify_pdf(tmp_path / "c.pdf")["layer2_kind"] == "text_only"
    assert trial.classify_pdf(tmp_path / "d.pdf") is None


def test_sample_balanced_and_prefers_image_text(data_root):
    """n=6: 3 file 1 lớp + 3 file 2 lớp, file 2 lớp ưu tiên dạng ảnh+chữ; cùng seed cùng mẫu; dry-run không gọi API."""
    sample, stats = trial.pick_source_sample("vjol", data_root / "raw" / "vjol", 6, 1, 60, 30, 100)
    assert [e["layer"] for e in sample].count(1) == 3 and [e["layer"] for e in sample].count(2) == 3
    assert all(e["layer2_kind"] == "image_text" for e in sample if e["layer"] == 2)
    assert all("metadata" not in e["rel"] for e in sample)
    again, _ = trial.pick_source_sample("vjol", data_root / "raw" / "vjol", 6, 1, 60, 30, 100)
    assert [e["sha256"] for e in again] == [e["sha256"] for e in sample]
    # xin 8 mà chỉ có 3 ảnh+chữ: bù bằng text_only, không bù bằng file scan
    sample8, st8 = trial.pick_source_sample("vjol", data_root / "raw" / "vjol", 8, 1, 60, 30, 100)
    assert st8["got_layer1"] == 3 and st8["want_layer1"] == 4 and st8["got_layer2"] == 4 and st8["layer2_text_only"] == 1


def test_full_run_with_fake_server(data_root, server, monkeypatch):
    """Chạy cả luồng: mọi file ok, khoá nằm trong header API nhưng không lộ sang link ký sẵn, chạy lại thì không gọi API nữa, có báo cáo."""
    monkeypatch.setenv("VNPT_OCR_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("VNPT_OCR_TOKEN_ID", "tid")
    monkeypatch.setenv("VNPT_OCR_TOKEN_KEY", "tkey")
    monkeypatch.setenv("VNPT_OCR_DOMAIN", server)
    argv = ["--data-root", str(data_root), "--per-source", "4", "--poll-interval", "0.01", "--workers", "2"]
    assert trial.main(argv) == 0
    out = data_root / "processed" / "ocr_internal_trial" / "trial"
    results = [json.loads(p.read_text(encoding="utf-8")) for p in (out / "results").glob("*.json")]
    assert len(results) == 8 and all(r["status"] == "ok" and r["pages"] and r["ocr_chars"] > 0 for r in results)
    assert set(results[0]["ocr_pages"]) == {"1", "2"}
    assert {h[1] for h in FakeApi.seen_headers} == {"Bearer tok"} and {h[3] for h in FakeApi.seen_headers} == {"tkey"}
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert set(report["groups"]) == {"vjol/layer1", "vjol/layer2", "vista/layer1", "vista/layer2"}
    assert report["groups"]["vjol/layer2"]["mean_overlap_vs_text_layer"] is not None
    assert (out / "report.csv").exists()
    n_calls = len(FakeApi.seen_headers)
    assert trial.main(argv) == 0 and len(FakeApi.seen_headers) == n_calls  # resume: không gọi lại


def test_missing_keys_exit(data_root, monkeypatch):
    """Thiếu khoá thì thoát với thông báo nêu tên biến, không gọi mạng."""
    for k in ("VNPT_OCR_ACCESS_TOKEN", "VNPT_OCR_TOKEN_ID", "VNPT_OCR_TOKEN_KEY", "ACCESS_TOKEN", "TOKEN_ID", "TOKEN_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(trial, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(SystemExit) as ei:
        trial.main(["--data-root", str(data_root), "--per-source", "4"])
    assert "TOKEN_KEY" in str(ei.value)
