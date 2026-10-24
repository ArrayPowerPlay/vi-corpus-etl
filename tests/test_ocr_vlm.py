"""
Test cho chuỗi OCR VLM (O-20): quy tắc chọn engine (pick_tier / merge_cascade / cascade_hours), CascadeOcr với engine giả,
OCR sách stbook có checkpoint + nhận cuốn bằng khoá, chấm điểm engine chuỗi, mẫu trang và bảng tổng hợp của bộ thử concurrency.
Không cần mạng hay GPU (engine và server được thay bằng giả).
"""

import io
import json
import shutil
import threading
from pathlib import Path

import pymupdf
import pytest
import requests
from PIL import Image

from vi_corpus.common.state import read_json
from vi_corpus.ocr_bakeoff import cascade as cas
from vi_corpus.ocr_bakeoff.scoring import CASCADE_NAME, score_bakeoff
from vi_corpus.stbook import ocr_vlm
from vi_corpus.stbook.ocr_books import iter_records

GOOD = ("Người dân miền núi trồng chè trên những sườn đồi thoai thoải quanh năm. " * 12).strip()
LOOP = "Đọc đúng phần đầu\n" + " . " * 400


class FakeEngine:
    """Engine giả: `script(image_bytes) -> dict` quyết định đầu ra; đếm số lượt gọi."""

    def __init__(self, script):
        """Lưu kịch bản."""
        self.script, self.calls = script, 0

    def run_bytes(self, image, ext):
        """Chạy kịch bản trên ảnh."""
        self.calls += 1
        res = self.script(image)
        if isinstance(res, Exception):
            raise res
        return res


def ok(text: str = GOOD) -> dict:
    """Đầu ra bình thường."""
    return {"raw": text, "finish_reason": "stop"}


def make_cascade(*scripts, serial_last=False, **kw) -> cas.CascadeOcr:
    """CascadeOcr gồm các engine giả t0, t1, ... (engine cuối tên baseline nếu serial_last)."""
    tiers = []
    for i, sc in enumerate(scripts):
        last = serial_last and i == len(scripts) - 1
        name = "baseline" if last else f"t{i}"
        cfg = {"type": "baseline" if last else "openai", "output": "plain", "max_tokens": 4096}
        eng = FakeEngine(sc)
        tiers.append(cas.Tier(name, cfg, factory=lambda e=eng: e, serial=last))
    return cas.CascadeOcr(tiers, backoff_s=0, **kw)


# --------------------------------------------------------------------------------------------------------------------
# Quy tắc chọn
# --------------------------------------------------------------------------------------------------------------------


def test_pick_tier_chon_engine_dau_khong_bi_bao_va_goi_luoi():
    """Engine sau chỉ được gọi khi engine trước bị báo; engine cuối nhận bất kể; thiếu kết quả thì bỏ qua engine đó."""
    recs = {"a": {"raw": LOOP, "text": LOOP, "finish_reason": "stop"}, "b": {"raw": GOOD, "text": GOOD, "finish_reason": "stop"},
            "c": {"raw": "", "text": "", "finish_reason": "stop"}}
    asked = []

    def get(name):
        """Ghi lại engine được hỏi."""
        asked.append(name)
        return recs.get(name)

    assert cas.pick_tier(["a", "b", "c"], get, {})[:1] == ("b",) and asked == ["a", "b"]
    name, rec, reasons = cas.pick_tier(["a", "c"], recs.get, {})
    assert name == "c" and reasons == ["empty"]  # engine cuối nhận bất kể, lý do còn lại được báo
    assert cas.pick_tier(["x", "b"], recs.get, {})[0] == "b"
    assert cas.pick_tier(["x", "y"], recs.get, {}) == ("y", None, [])


def test_merge_cascade_va_gio_chay():
    """Ghép đầu ra theo chuỗi; giờ cả kho = tổng (tỷ lệ trang tới engine) x (giờ engine đó nếu chạy mọi trang)."""
    runs = {"a": {f"p{i}": {"raw": GOOD, "text": GOOD, "finish_reason": "stop"} for i in range(10)}, "b": {}, "c": {}}
    for i in (0, 1):  # 2 trang bị cắt cụt ở engine a
        runs["a"][f"p{i}"]["finish_reason"] = "length"
        runs["b"][f"p{i}"] = {"raw": GOOD, "text": GOOD, "finish_reason": "stop"}
    runs["b"]["p0"]["finish_reason"] = "length"  # p0 vẫn hỏng ở b -> xuống c
    runs["c"]["p0"] = {"raw": GOOD, "text": GOOD, "finish_reason": None}
    merged, used = cas.merge_cascade(runs, ["a", "b", "c"], {}, [f"p{i}" for i in range(10)])
    assert dict(used) == {"a": 8, "b": 1, "c": 1} and merged["p0"]["tier"] == "c" and merged["p1"]["guard"] == []
    h = {t: {"corpus_hours": g, "corpus_pages": 100, "bakeoff_pages": 10, "bakeoff_run_hours": 1.0, "bakeoff_startup_hours": 0.1,
             "steady": True, "corpus_hours_by_source": {"stbook": g}} for t, g in (("a", 40.0), ("b", 40.0), ("c", 20.0))}
    out = cas.cascade_hours(h, ["a", "b", "c"], used, 4)
    assert out["reach"] == {"a": 1.0, "b": 0.2, "c": 0.1}
    assert out["corpus_hours"] == pytest.approx(40 + 0.2 * 40 + 0.1 * 20) and out["corpus_hours_by_source"]["stbook"] == out["corpus_hours"]
    h["b"]["corpus_hours"] = None
    assert cas.cascade_hours(h, ["a", "b", "c"], used, 4)["corpus_hours"] is None


# --------------------------------------------------------------------------------------------------------------------
# CascadeOcr
# --------------------------------------------------------------------------------------------------------------------


def test_cascade_ocr_cac_trang_thai():
    """ok một lượt; bị cắt cụt thì chạy lại; mọi engine hỏng thì flagged (text rỗng, giữ rejected_text); mọi engine rỗng thì blank."""
    c = make_cascade(lambda im: ok(), lambda im: ok("chạy lại"))
    r = c.page(b"x")
    assert r["status"] == "ok" and r["tier"] == "t0" and r["text"] == GOOD and "attempts" not in r
    assert c.tiers[1].engine().calls == 0  # engine sau không được gọi

    c = make_cascade(lambda im: {"raw": GOOD, "finish_reason": "length"}, lambda im: ok("đã sửa"))
    r = c.page(b"x")
    assert r["status"] == "ok" and r["tier"] == "t1" and r["text"] == "đã sửa"
    assert [a["reasons"] for a in r["attempts"]] == [["length"], []]

    c = make_cascade(lambda im: ok(LOOP), lambda im: ok(LOOP), lambda im: ok(LOOP), serial_last=True)
    r = c.page(b"x")
    assert r["status"] == "flagged" and r["text"] == "" and r["rejected_text"] == LOOP.strip() and r["reasons"] == ["loop"]

    c = make_cascade(lambda im: ok(""), lambda im: ok(""), lambda im: ok(""), serial_last=True)
    r = c.page(b"x")
    assert r["status"] == "blank" and r["text"] == "" and "rejected_text" not in r

    c = make_cascade(lambda im: ok(""), lambda im: ok(LOOP))  # rỗng rồi lặp: không phải trang trắng
    assert c.page(b"x")["status"] == "flagged"


def test_cascade_ocr_loi_server_va_thu_lai():
    """Lỗi 500 được thử lại; lỗi 400 thì rơi xuống engine sau; không kết nối được nhiều lần liên tiếp thì ServerDown."""
    resp500 = requests.Response()
    resp500.status_code = 500
    calls = {"n": 0}

    def flaky(im):
        """Lỗi 500 hai lần rồi thành công."""
        calls["n"] += 1
        return requests.HTTPError("lỗi", response=resp500) if calls["n"] < 3 else ok()

    c = make_cascade(flaky, lambda im: ok("không dùng"), retries=2)
    r = c.page(b"x")
    assert r["status"] == "ok" and r["tier"] == "t0" and calls["n"] == 3

    resp400 = requests.Response()
    resp400.status_code = 400
    c = make_cascade(lambda im: requests.HTTPError("ảnh hỏng", response=resp400), lambda im: ok("engine sau"), retries=2)
    r = c.page(b"x")
    assert r["tier"] == "t1" and r["attempts"][0]["reasons"] == ["error"] and "ảnh hỏng" in r["attempts"][0]["error"]

    c = make_cascade(lambda im: requests.ConnectionError("refused"), lambda im: ok(), retries=0, max_conn_errors=3)
    c.page(b"x")
    c.page(b"x")
    with pytest.raises(cas.ServerDown):
        c.page(b"x")


def test_engine_khong_nap_duoc_thi_dung_ca_luot_chu_khong_loai_trang():
    """Engine dự phòng thiếu thư viện (lúc dựng hoặc lúc chạy) ném EngineUnavailable; lỗi riêng của một trang vẫn chỉ rơi trang đó (O-24)."""
    def broken_factory():
        """Giả lập ImportError libGL khi nạp engine."""
        raise ImportError("libGL.so.1: cannot open shared object file")

    c = make_cascade(lambda im: ok(LOOP), lambda im: ok())
    c.tiers[1] = cas.Tier("baseline", {"type": "baseline", "output": "plain"}, factory=broken_factory, serial=True)
    with pytest.raises(cas.EngineUnavailable, match="libGL"):
        c.page(b"x")  # engine đầu bị báo lặp, rơi xuống baseline, baseline không nạp được

    c = make_cascade(lambda im: ok(LOOP), lambda im: ImportError("cv2 thiếu libGL"), serial_last=True)
    with pytest.raises(cas.EngineUnavailable, match="thiếu thư viện"):
        c.page(b"x")  # nạp được nhưng lỗi import khi chạy

    c = make_cascade(lambda im: ok(LOOP), lambda im: RuntimeError("ảnh lạ"), serial_last=True)
    r = c.page(b"x")  # lỗi riêng của một trang: chỉ trang đó bị loại
    assert r["status"] == "flagged" and r["attempts"][-1]["error"].startswith("RuntimeError")


def test_ocr_sach_engine_khong_nap_duoc_dung_luot_va_khong_ghi_ket_qua(tmp_path):
    """EngineUnavailable dừng cả lượt, không ghi cuốn đang làm (không để trang bị loại âm thầm vào checkpoint)."""
    stbook = _stbook_root(tmp_path)
    c = make_cascade(lambda im: ok(LOOP), lambda im: ImportError("libGL"), serial_last=True)
    with pytest.raises(cas.EngineUnavailable):
        ocr_vlm.run_ocr_vlm(stbook, tmp_path, c, concurrency=2)
    assert not list((tmp_path / "interim/stbook_ocr").glob("*/*.json"))


def test_build_tiers_chung_server_va_doi_cong():
    """Các engine VLM dùng chung server của engine VLM đầu (cổng đổi được); khác mô hình thì lỗi; baseline nạp lười."""
    engines = {"a": {"type": "openai", "model": "M", "base_url": "http://127.0.0.1:8012/v1", "max_tokens": 4096,
                     "serve": ["vllm", "serve", "M", "--port", "8012", "--gpu-memory-utilization", "0.9"]},
               "b": {"type": "openai", "model": "M", "base_url": "http://127.0.0.1:8014/v1", "extra_body": {"repetition_penalty": 1.1}},
               "other": {"type": "openai", "model": "N", "base_url": "http://x:1/v1"}, "baseline": {"type": "baseline"},
               "pvl": {"type": "paddleocr_vl"}}
    tiers = cas.build_tiers(["a", "b", "baseline"], engines, port=8103, gpu_mem=0.8)
    assert tiers[0].cfg["base_url"] == tiers[1].cfg["base_url"] == "http://127.0.0.1:8103/v1"
    assert tiers[0].cfg["serve"][tiers[0].cfg["serve"].index("--port") + 1] == "8103"
    assert tiers[0].cfg["serve"][-1] == "0.8" and engines["a"]["base_url"].endswith(":8012/v1")  # cấu hình gốc không bị sửa
    assert tiers[1].cfg["extra_body"] == {"repetition_penalty": 1.1} and tiers[2].serial and tiers[2]._engine is None
    with pytest.raises(ValueError, match="khác"):
        cas.build_tiers(["a", "other"], engines)
    with pytest.raises(ValueError, match="không dùng được"):
        cas.build_tiers(["a", "pvl"], engines)
    with pytest.raises(ValueError, match="không có trong cấu hình"):
        cas.build_tiers(["zzz"], engines)


# --------------------------------------------------------------------------------------------------------------------
# OCR sách stbook
# --------------------------------------------------------------------------------------------------------------------


def _stbook_root(root: Path, books=(("b1", 6), ("b2", 3))) -> Path:
    """<root>/raw/stbook với một danh mục và các sách ảnh; trang i là ảnh đơn sắc mức xám 40*i (để engine giả nhận ra số trang)."""
    cat = root / "raw/stbook/sach-hay"
    (cat / "content").mkdir(parents=True)
    for pid, n in books:
        doc = pymupdf.open()
        for i in range(n):
            buf = io.BytesIO()
            Image.new("RGB", (60, 90), (40 * i, 40 * i, 40 * i)).save(buf, format="JPEG", quality=95)
            doc.new_page(width=60, height=90).insert_image(pymupdf.Rect(0, 0, 60, 90), stream=buf.getvalue())
        doc.save(cat / "content" / f"{pid}.pdf")
    (cat / "books.json").write_text(json.dumps({"books": [{"product_id": pid, "title": f"Sách {pid}"} for pid, _ in books]}))
    return root / "raw/stbook"


def _page_no(image: bytes) -> int:
    """Số trang suy từ mức xám của ảnh đơn sắc."""
    return round(Image.open(io.BytesIO(image)).convert("L").getpixel((5, 5)) / 40)


def test_ocr_sach_theo_chuoi_ghi_ket_qua_va_chay_tiep(tmp_path):
    """
    Sách 6 trang (đánh số từ 0): trang 2 bị cắt cụt ở t0 (t1 sửa được), trang 3 hỏng ở cả ba engine (bị loại), trang 4 trắng thật.
    Kết quả đúng thứ tự trang; trang bị loại có text rỗng + rejected_text; iter_records không đưa trang bị loại vào kho;
    chạy lại không làm lại; cuốn đã OCR bằng bộ cũ được làm lại và file cũ được chép sang thư mục dự phòng.
    """
    from vi_corpus.common.registry import SOURCES

    stbook = _stbook_root(tmp_path)

    def t0(im):
        """Engine chính."""
        n = _page_no(im)
        if n == 2:
            return {"raw": GOOD, "finish_reason": "length"}
        return ok(LOOP if n == 3 else ("" if n == 4 else f"Nội dung trang {n} của sách. " * 5))

    def t1(im):
        """Engine chạy lại."""
        n = _page_no(im)
        return ok(LOOP if n == 3 else ("" if n == 4 else f"Đã sửa trang {n}. " * 5))

    def base(im):
        """Engine dự phòng."""
        n = _page_no(im)
        return ok(LOOP if n == 3 else "")

    cascade = make_cascade(t0, t1, base, serial_last=True)
    # b2 đã được OCR bằng bộ cũ
    old = tmp_path / "interim/stbook_ocr/sach-hay/b2.json"
    old.parent.mkdir(parents=True)
    pdf2 = stbook / "sach-hay/content/b2.pdf"
    old.write_text(json.dumps({"book": {"product_id": "b2"}, "pdf_path": "x", "pdf_size": pdf2.stat().st_size, "ocr": {"det": "PP"},
                               "pages": ["cũ"], "category_slug": "sach-hay"}))
    assert ocr_vlm.vlm_status(stbook, tmp_path) == {"vlm": 0, "other": 1, "total": 2}

    assert ocr_vlm.run_ocr_vlm(stbook, tmp_path, cascade, concurrency=3) == 0
    done = read_json(tmp_path / "interim/stbook_ocr/sach-hay/b1.json")
    assert done["ocr"]["engine"] == "vlm_cascade" and len(done["pages"]) == 6
    assert done["pages"][0].startswith("Nội dung trang 0") and done["pages"][5].startswith("Nội dung trang 5")
    assert done["pages"][2].startswith("Đã sửa trang 2")  # bị cắt cụt ở t0, t1 sửa được
    assert done["page_info"][2]["tier"] == "t1" and done["page_info"][2]["attempts"][0]["reasons"] == ["length"]
    assert done["pages"][3] == "" and done["page_info"][3]["status"] == "flagged" and done["page_info"][3]["rejected_text"] == LOOP.strip()
    assert done["pages"][4] == "" and done["page_info"][4]["status"] == "blank"
    assert done["stats"]["flagged"] == 1 and done["stats"]["blank"] == 1 and done["stats"]["rerun"] == 3
    assert done["ocr"]["tiers"][0]["name"] == "t0"
    backup = tmp_path / "interim/stbook_ocr_baseline/sach-hay/b2.json"
    assert read_json(backup)["pages"] == ["cũ"]  # bản của bộ OCR cũ được giữ
    assert ocr_vlm.vlm_status(stbook, tmp_path) == {"vlm": 2, "other": 0, "total": 2}
    rep = ocr_vlm.flagged_report(tmp_path)
    assert rep["books"] == 2 and rep["flagged"] == 1 and rep["pages"] == 9

    recs = list(iter_records(SOURCES["stbook"], tmp_path))
    text = next(r["text"] for r in recs if r["source_path"].endswith("b1.pdf"))
    assert "Nội dung trang 0" in text and LOOP not in text

    calls = cascade.tiers[0].engine().calls
    assert ocr_vlm.run_ocr_vlm(stbook, tmp_path, cascade, concurrency=3) == 0
    assert cascade.tiers[0].engine().calls == calls  # xong rồi thì không làm lại
    assert ocr_vlm.run_ocr_vlm(stbook, tmp_path, cascade, concurrency=3, redo=True) == 0
    assert cascade.tiers[0].engine().calls > calls


def test_ocr_sach_server_chet_khong_ghi_ket_qua_va_pdf_hong_khong_dung_luot(tmp_path):
    """ServerDown dừng cả lượt và không ghi cuốn đang làm; PDF hỏng chỉ tính một cuốn lỗi."""
    stbook = _stbook_root(tmp_path)
    c = make_cascade(lambda im: requests.ConnectionError("refused"), retries=0, max_conn_errors=2)
    with pytest.raises(cas.ServerDown):
        ocr_vlm.run_ocr_vlm(stbook, tmp_path, c, concurrency=2)
    assert not list((tmp_path / "interim/stbook_ocr").glob("*/*.json"))

    (stbook / "sach-hay/content/b1.pdf").write_bytes(b"%PDF-1.4 hong")
    good = make_cascade(lambda im: ok())
    assert ocr_vlm.run_ocr_vlm(stbook, tmp_path, good, concurrency=2) == 1  # b1 hỏng, b2 làm được
    assert (tmp_path / "interim/stbook_ocr/sach-hay/b2.json").exists() and not (tmp_path / "interim/stbook_ocr/sach-hay/b1.json").exists()


def test_nhan_cuon_bang_khoa(tmp_path):
    """Hai tiến trình không làm cùng một cuốn: khoá đang giữ thì claim_book trả False; nhả khoá thì nhận được."""
    with ocr_vlm.claim_book(tmp_path, "sach-hay", "b1") as first:
        assert first
        res = {}

        def other():
            """Thử nhận cùng cuốn từ luồng khác (flock theo mô tả file mở nên khoá xung đột giữa các luồng)."""
            with ocr_vlm.claim_book(tmp_path, "sach-hay", "b1") as got:
                res["got"] = got

        t = threading.Thread(target=other)
        t.start()
        t.join()
        assert res["got"] is False
    with ocr_vlm.claim_book(tmp_path, "sach-hay", "b1") as again:
        assert again


# --------------------------------------------------------------------------------------------------------------------
# Chấm điểm chuỗi
# --------------------------------------------------------------------------------------------------------------------


def _scoring_bakeoff(root: Path, n: int = 20):
    """Thư mục so sánh giả n trang bộ A (đáp án đọc được); trả về danh sách đáp án."""
    (root / "gt").mkdir(parents=True)
    (root / "pages").mkdir()
    refs, lines = [], []
    for i in range(n):
        ref = f"Trang {i} có diện tích 3.000 km² và “trích dẫn” – nội dung bình thường của trang sách số {i}."
        (root / "gt" / f"A{i}.txt").write_text(ref, encoding="utf-8")
        refs.append(ref)
        lines.append({"page_id": f"A{i}", "set": "A", "image": f"pages/A{i}.jpg", "gt": f"gt/A{i}.txt", "flags": {},
                      "size": [600, 900], "source": {"file": "x.pdf", "page": i}})
    (root / "pages.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in lines), encoding="utf-8")
    (root / "selection.json").write_text(json.dumps({"corpus_pages": {"stbook": 1_000_000, "giao_trinh_ocr": 0}}))
    return refs


def _run(root: Path, name: str, recs: dict[str, dict], pps: float, max_tokens: int | None = None) -> None:
    """Ghi runs/<name>/outputs.jsonl giả (mỗi trang một bản ghi cho sẵn) cách đều 1/pps giây, và run_info.json."""
    d = root / "runs" / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    with open(d / "outputs.jsonl", "w", encoding="utf-8") as f:
        for k, (pid, rec) in enumerate(recs.items(), 1):
            f.write(json.dumps({"page_id": pid, "error": None, "session": "s", "t_start": 1000 + (k - 1) / pps,
                                "t_end": 1000 + k / pps, "raw": rec["text"], **rec}, ensure_ascii=False) + "\n")
    (d / "run_info.json").write_text(json.dumps({"config": {"output": "plain", "max_tokens": max_tokens}, "sessions": []}))


def test_cham_engine_chuoi_cat_cut_sau_buoc_kiem_va_gio_chay(tmp_path):
    """
    Engine chính cắt cụt 2/20 trang: chạy một mình vượt cổng cắt cụt 0,5% và bị loại; engine chuỗi (chạy lại bằng engine hai)
    còn 0 trang cắt cụt, 0 trang bị báo, qua cổng; giờ cả kho = giờ engine chính x (1 + tỷ lệ trang chạy lại) khi hai engine cùng tốc độ.
    """
    refs = _scoring_bakeoff(tmp_path)
    first = {f"A{i}": {"text": r, "finish_reason": "stop"} for i, r in enumerate(refs)}
    for i in (3, 7):
        first[f"A{i}"] = {"text": refs[i][:20] + " . " * 300, "finish_reason": "length"}
    second = {f"A{i}": {"text": refs[i], "finish_reason": "stop"} for i in (3, 7)}
    base = {f"A{i}": {"text": refs[i], "finish_reason": None} for i in range(20)}
    _run(tmp_path, "baseline", base, 4.0)
    _run(tmp_path, "first", first, 2.0, 4096)
    _run(tmp_path, "second", second, 2.0, 4096)
    s = score_bakeoff(tmp_path, ["baseline", "first"], cascade=["first", "second", "baseline"])
    assert s["cascade"]["pages_by_tier"] == {"first": 18, "second": 2, "baseline": 0}
    assert s["gates"]["first"]["checks"]["trunc_rate"]["status"] == "fail"
    g = s["gates"][CASCADE_NAME]["checks"]
    assert g["trunc_rate"]["status"] == "pass" and g["trunc_rate"]["value"] == 0
    assert g["flagged_rate"]["value"] == 0 and g["flagged_rate"]["status"] == "pass"
    assert s["set_a"][CASCADE_NAME]["trunc_rate"] == 0 and s["set_a"]["first"]["flagged_rate"] is None
    h1, hc = s["hours"]["first"]["corpus_hours"], s["hours"][CASCADE_NAME]
    h_base = s["hours"]["baseline"]["corpus_hours"]
    assert hc["reach"] == {"first": 1.0, "second": 0.1, "baseline": 0.0}
    assert hc["corpus_hours"] == pytest.approx(h1 * 1.1) and h_base < h1
    assert s["decision"]["winner"] == CASCADE_NAME
    assert (tmp_path / "report/report.html").exists()
    assert "second" not in s["set_a"]  # engine chỉ dùng để ghép không bị chấm riêng
    # engine cuối cũng hỏng ở A3: trang đó còn bị báo sau cả chuỗi, flagged_rate > 0 nhưng chỉ cảnh báo (không loại)
    second["A3"] = {"text": " . " * 400, "finish_reason": "stop"}
    base["A3"] = {"text": " . " * 400, "finish_reason": None}
    _run(tmp_path, "second", second, 2.0, 4096)
    _run(tmp_path, "baseline", base, 4.0)
    c = score_bakeoff(tmp_path, ["baseline", "first"], cascade=["first", "second", "baseline"])["gates"][CASCADE_NAME]
    assert c["checks"]["flagged_rate"]["status"] == "warn" and c["checks"]["flagged_rate"]["value"] == pytest.approx(0.05)
    assert c["passed_quality"] is True


# --------------------------------------------------------------------------------------------------------------------
# Bộ thử concurrency
# --------------------------------------------------------------------------------------------------------------------


def test_mau_trang_va_bang_tong_hop(tmp_path):
    """sample_stbook_pages chọn đủ số trang từ nhiều sách (bỏ sách ngắn, trang không trùng); summarize đọc các mức và lượt chạy lại."""
    from vi_corpus.ocr_bakeoff import vlm_bench as vb
    from vi_corpus.stbook.ocr_books import find_books

    stbook = _stbook_root(tmp_path, books=(("b1", 60), ("b2", 50), ("short", 5)))
    out = tmp_path / "bench"
    recs = vb.sample_stbook_pages(list(find_books(stbook)), out, 40, 5, seed=1)
    assert len(recs) == 40 and {r["source"]["product_id"] for r in recs} == {"b1", "b2"}
    assert len({(r["source"]["product_id"], r["source"]["page"]) for r in recs}) == 40
    assert all((out / r["image"]).exists() and 0.05 * 50 <= r["source"]["page"] <= 60 for r in recs)
    assert (out / "pages.jsonl").read_text().count("\n") == 40

    ids = [r["page_id"] for r in recs]
    for conc, pps, flagged in ((16, 2.0, 2), (32, 3.0, 2)):
        d = out / "runs" / f"c{conc}"
        d.mkdir(parents=True)
        with open(d / "outputs.jsonl", "w", encoding="utf-8") as f:
            for k, pid in enumerate(ids, 1):
                bad = k <= flagged
                f.write(json.dumps({"page_id": pid, "error": None, "session": "s", "t_start": 1000 + (k - 1) / pps, "t_end": 1000 + k / pps + 1,
                                    "raw": LOOP if bad else GOOD, "text": LOOP if bad else GOOD, "finish_reason": "stop",
                                    "completion_tokens": 500}) + "\n")
        (d / "run_info.json").write_text(json.dumps({"config": {"max_tokens": 4096}, "sessions": [{"session": "s", "concurrency": conc}]}))
    flagged_ids = list(vb.flagged_pages(out / "runs" / "c16"))
    assert flagged_ids == ids[:2]
    retry = vb.make_retry_dir(out, flagged_ids)
    assert [json.loads(line)["image"] for line in (retry / "pages.jsonl").read_text().splitlines()][0].startswith("../pages/")
    assert (retry / json.loads((retry / "pages.jsonl").read_text().splitlines()[0])["image"]).exists()
    r = retry / "runs" / "rp110"
    r.mkdir(parents=True)
    (r / "outputs.jsonl").write_text("\n".join(json.dumps({"page_id": pid, "error": None, "raw": LOOP if k == 0 else GOOD,
                                                          "text": LOOP if k == 0 else GOOD, "finish_reason": "stop"})
                                              for k, pid in enumerate(flagged_ids)))
    (r / "run_info.json").write_text(json.dumps({"config": {"max_tokens": 4096}, "sessions": []}))
    s = vb.summarize(out, corpus_pages=100_000, num_gpus=4)
    assert list(s["levels"]) == ["c16", "c32"] and s["levels"]["c16"]["flagged"] == 2 and s["levels"]["c16"]["flag_reasons"] == {"loop": 2}
    assert s["levels"]["c32"]["concurrency"] == 32 and s["retry"]["still_flagged"] == 1 and s["retry"]["recovered_rate"] == 0.5
    assert s["best_level"] == "c32" and s["levels"]["c32"]["pages_per_s"] > s["levels"]["c16"]["pages_per_s"]
    text = vb.format_report(s)
    assert "c32" in text and "còn bị báo 1" in text and (out / "report/summary.json").exists()


def test_alias_headless_opencv_cho_paddlex_thay_goi_headless(monkeypatch):
    """paddlex hỏi `opencv-contrib-python` mà chỉ có bản headless thì phải thấy phiên bản; gói khác không bị ảnh hưởng; gọi hai lần vẫn một lớp vá."""
    import functools
    import sys
    import types

    from vi_corpus.common.ocr import alias_headless_opencv

    installed = {"opencv-contrib-python-headless": "4.14.0.94", "pypdfium2": "5.0"}
    deps = types.ModuleType("paddlex.utils.deps")
    deps.get_dep_version = installed.get
    deps.is_dep_available = functools.lru_cache()(lambda dep: deps.get_dep_version(dep) is not None)
    utils = types.ModuleType("paddlex.utils")
    utils.deps = deps
    monkeypatch.setitem(sys.modules, "paddlex", types.ModuleType("paddlex"))
    monkeypatch.setitem(sys.modules, "paddlex.utils", utils)
    monkeypatch.setitem(sys.modules, "paddlex.utils.deps", deps)

    assert not deps.is_dep_available("opencv-contrib-python")  # đã bị nhớ là "không có"
    assert alias_headless_opencv() and alias_headless_opencv()
    assert deps.is_dep_available("opencv-contrib-python") and deps.get_dep_version("opencv-contrib-python") == "4.14.0.94"
    assert deps.get_dep_version("pypdfium2") == "5.0" and deps.get_dep_version("không-có-gói-này") is None
    installed.pop("opencv-contrib-python-headless")
    deps.is_dep_available.cache_clear()
    assert not deps.is_dep_available("opencv-contrib-python")  # không có cả hai thì vẫn báo thiếu


def test_alias_headless_opencv_khong_co_paddlex(monkeypatch):
    """Không cài paddlex (máy dev, group ocr chưa sync) thì trả False, không ném lỗi."""
    import sys

    from vi_corpus.common.ocr import alias_headless_opencv

    for name in ("paddlex", "paddlex.utils", "paddlex.utils.deps"):
        monkeypatch.setitem(sys.modules, name, None)  # import trả ImportError
    assert alias_headless_opencv() is False
