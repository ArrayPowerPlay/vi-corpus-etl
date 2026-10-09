"""
Test cho bộ công cụ so sánh engine OCR (F-11, vi_corpus/ocr_bakeoff): khoảng cách sửa, adapter văn bản, đáp án theo thứ
tự đọc trên PDF dựng sẵn, chọn trang đầu cuối trên data root dựng sẵn, chấm điểm + quy tắc chọn, client API. Không cần
mạng hay GPU (engine thật được thay bằng engine giả).
"""

import io
import json
import random
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from vi_corpus.ocr_bakeoff import engines as eng
from vi_corpus.ocr_bakeoff.metrics import (
    levenshtein,
    page_scores,
    paired_bootstrap,
    strip_diacritics,
)
from vi_corpus.ocr_bakeoff.pages import (
    analyze_page,
    parse_forced,
    prepare_pages,
    read_pages,
)
from vi_corpus.ocr_bakeoff.scoring import (
    Gates,
    page_bad_reason,
    score_bakeoff,
    throughput,
)
from vi_corpus.ocr_bakeoff.textnorm import to_plain

FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
needs_font = pytest.mark.skipif(not FONT.exists(), reason="thiếu font DejaVuSans (cần glyph tiếng Việt)")

SENT_L = "Người dân miền núi trồng chè trên những sườn đồi thoai thoải quanh năm."
SENT_R = "Cây lúa nước gắn bó với làng quê đồng bằng sông Hồng từ bao đời nay."
SENT_ONE = ("Giáo trình này trình bày những kiến thức cơ bản về kinh tế học vĩ mô, giúp sinh viên hiểu rõ các khái niệm "
            "tổng cầu, tổng cung, lạm phát và thất nghiệp — kèm ví dụ “thực tế” từ nền kinh tế Việt Nam…")


def _naive(a, b) -> int:
    """Levenshtein quy hoạch động thường, để đối chiếu."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def test_levenshtein_khop_quy_hoach_dong():
    """Bản song song bit cho cùng kết quả với quy hoạch động trên chuỗi và list từ ngẫu nhiên."""
    rng = random.Random(1)
    for _ in range(300):
        a = "".join(rng.choice("abcđê ") for _ in range(rng.randint(0, 80)))
        b = "".join(rng.choice("abcđê ") for _ in range(rng.randint(0, 80)))
        assert levenshtein(a, b) == _naive(a, b)
    assert levenshtein(["một", "hai", "ba"], ["một", "ba", "bốn"]) == 2
    assert strip_diacritics("Đường phố") == "Duong pho"


def test_page_scores_tach_loi_dau_va_thu_tu():
    """Mất dấu: CER > 0 nhưng CER bỏ dấu = 0. Đảo thứ tự: F1 túi từ = 1 nhưng WER > 0. Ký tự đặc biệt đếm đúng."""
    s = page_scores("Nguoi dan", "Người dân")
    assert s["char_dist"] > 0 and s["char_dist_nodiac"] == 0
    s = page_scores("b a", "a b")
    assert s["bow_f1"] == 1.0 and s["word_dist"] == 2
    s = page_scores("km2 “x”", "km² “x”")
    assert s["special_ref"] == {"²": 1, "“": 1, "”": 1} and sum(s["special_hit"].values()) == 2


def test_paired_bootstrap_khoang_tin_cay():
    """A tệ hơn B rõ rệt trên mọi trang: khoảng tin cậy của hiệu nằm hẳn trên 0; hai engine như nhau: chứa 0."""
    den = [100] * 50
    d, lo, hi = paired_bootstrap([10] * 50, [2] * 50, den)
    assert d == pytest.approx(0.08) and lo > 0
    rng = random.Random(0)
    a = [rng.randint(0, 10) for _ in range(50)]
    d, lo, hi = paired_bootstrap(a, list(reversed(a)), den)
    assert lo <= 0 <= hi


def test_to_plain_cac_dinh_dang():
    """Markdown (bảng HTML và bảng "|", mũ <sup>, hàng rào) và JSON bố cục dots.ocr (cả khi bị cắt cụt) về văn bản thuần."""
    md = "```markdown\n# Chương 1\nDiện tích **3.000 km<sup>2</sup>**\n| a | b |\n|---|---|\n| 1 | 2 |\n```"
    assert to_plain(md, "markdown") == "Chương 1\nDiện tích 3.000 km²\na\nb\n1\n2"
    assert to_plain("<table><tr><td>Hà&nbsp;Nội</td><td>x</td></tr></table>", "markdown") == "Hà Nội\nx"
    full = json.dumps([{"bbox": [0, 0, 1, 1], "category": "Picture"},
                       {"bbox": [0, 0, 1, 1], "category": "Text", "text": "Câu **một**"}], ensure_ascii=False)
    assert to_plain(full, "dots_layout_json") == "Câu một"
    cut = '[{"bbox":[1,2,3,4],"category":"Text","text":"Đầu"},{"bbox":[1,2,3,4],"category":"Text","text":"dở'
    assert to_plain(cut, "dots_layout_json") == "Đầu"
    with pytest.raises(ValueError):
        to_plain("x", "lạ")


def _two_column_page(doc: "pymupdf.Document", header: bool = True) -> None:
    """Thêm một trang A4 hai cột chữ tiếng Việt (tuỳ chọn có tiêu đề vắt qua khe ở đầu trang)."""
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="dv", fontfile=str(FONT))
    if header:
        page.insert_textbox(pymupdf.Rect(50, 40, 545, 70), "TIÊU ĐỀ CHƯƠNG MỘT VẮT QUA HAI CỘT CỦA TRANG",
                            fontname="dv", fontsize=12)
    for k in range(6):
        y = 90 + k * 110
        page.insert_textbox(pymupdf.Rect(50, y, 285, y + 100), f"Trái {k}. " + SENT_L * 2, fontname="dv", fontsize=9)
        page.insert_textbox(pymupdf.Rect(310, y, 545, y + 100), f"Phải {k}. " + SENT_R * 2, fontname="dv", fontsize=9)


def _one_column_page(doc: "pymupdf.Document") -> None:
    """Thêm một trang A4 một cột chữ tiếng Việt có ký tự đặc biệt."""
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="dv", fontfile=str(FONT))
    for k in range(5):
        y = 60 + k * 150
        page.insert_textbox(pymupdf.Rect(50, y, 545, y + 140), f"Đoạn {k}. " + SENT_ONE, fontname="dv", fontsize=10)


@needs_font
def test_dap_an_trang_hai_cot_doc_het_cot_trai():
    """Trang hai cột có tiêu đề: cờ two_column, đáp án = tiêu đề, hết cột trái rồi cột phải."""
    doc = pymupdf.open()
    _two_column_page(doc)
    _one_column_page(doc)
    problem, gt, flags = analyze_page(doc[0])
    assert problem is None and flags["two_column"]
    assert gt.startswith("TIÊU ĐỀ")
    assert gt.index("Trái 5.") < gt.index("Phải 0.")
    problem, gt, flags = analyze_page(doc[1])
    assert problem is None and not flags["two_column"] and flags["special"]
    assert gt.index("Đoạn 0.") < gt.index("Đoạn 4.")


@needs_font
def test_trang_khong_hop_le_bi_loai():
    """Trang ít chữ hoặc tiếng Việt không dấu không làm đáp án."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_font(fontname="dv", fontfile=str(FONT))
    page.insert_text((50, 80), "Ít chữ quá.", fontname="dv")
    assert analyze_page(doc[0])[0] == "few_chars"
    page = doc.new_page()
    page.insert_font(fontname="dv", fontfile=str(FONT))
    page.insert_textbox(pymupdf.Rect(50, 50, 545, 800), strip_diacritics(SENT_ONE) * 5, fontname="dv", fontsize=10)
    assert analyze_page(doc[1])[0] == "diacritics"


def _make_data_root(root: Path) -> None:
    """Data root nhỏ: 3 PDF giáo trình có lớp chữ, 2 sách stbook dạng ảnh (một là 1248)."""
    gt_dir = root / "raw/giao_trinh/kinh_te/vi_mo"
    gt_dir.mkdir(parents=True)
    for i in range(3):
        doc = pymupdf.open()
        _two_column_page(doc, header=i == 0)
        _one_column_page(doc)
        doc.save(gt_dir / f"gt{i}.pdf")
    cat = root / "raw/stbook/sach-hay"
    (cat / "content").mkdir(parents=True)
    books = []
    for pid, n in (("1248", 160), ("999", 30)):
        doc = pymupdf.open()
        buf = io.BytesIO()
        Image.new("RGB", (60, 90), "white").save(buf, format="JPEG")
        for _ in range(n):
            doc.new_page(width=60, height=90).insert_image(pymupdf.Rect(0, 0, 60, 90), stream=buf.getvalue())
        doc.save(cat / "content" / f"{pid}.pdf")
        books.append({"product_id": pid, "title": f"Sách {pid}"})
    (cat / "books.json").write_text(json.dumps({"books": books}), encoding="utf-8")
    ck = root / "interim/giao_trinh_text"
    ck.mkdir(parents=True)
    (ck / "abc.json").write_text(json.dumps({"pages": [{"method": "ocr"}, {"method": "text"},
                                                       {"method": "needs_ocr"}]}), encoding="utf-8")


@needs_font
def test_prepare_pages_dau_cuoi(tmp_path):
    """Chọn trang trên data root dựng: bộ B có trang ép 1248:150 (ảnh gốc), bộ A render cao bằng ảnh stbook, có đáp án."""
    _make_data_root(tmp_path)
    out = tmp_path / "bakeoff"
    sel = prepare_pages(tmp_path, out, n_a=4, n_b=4, min_books=2, scan_per_doc=2, max_per_doc=2,
                        quotas={"two_column": 1, "table": 0, "special": 1})
    pages = read_pages(out)
    b = [p for p in pages if p["set"] == "B"]
    a = [p for p in pages if p["set"] == "A"]
    assert {"1248", "999"} == set(sel["books_b"]) and len(b) == 4
    assert any(p["source"]["product_id"] == "1248" and p["source"]["page"] == 150 for p in b)
    assert sel["target_height"] == 90 and all(p["size"][1] == 90 for p in a)
    assert len(a) == 4 and sel["strata"]["two_column"] >= 1 and not sel["shortfall"]
    assert all((out / p["gt"]).read_text(encoding="utf-8") for p in a)
    assert all((out / p["image"]).exists() for p in pages)
    assert sel["corpus_pages"] == {"stbook": 190, "giao_trinh_ocr": 2, "giao_trinh_checkpoints": 1,
                                   "note": sel["corpus_pages"]["note"]}
    assert parse_forced(["1248:150,151", "1271:20"]) == {"1248": [150, 151], "1271": [20]}


def _fake_bakeoff(root: Path, n: int = 30) -> list[str]:
    """Thư mục so sánh giả: n trang bộ A (đáp án có ký tự đặc biệt; 10 trang đầu hai cột) và 2 trang bộ B."""
    (root / "gt").mkdir(parents=True)
    (root / "pages").mkdir()
    refs, lines = [], []
    for i in range(n):
        ref = f"Trang {i} có diện tích 3.000 km² và “trích dẫn” – {SENT_L} {SENT_R}"
        (root / "gt" / f"A{i}.txt").write_text(ref, encoding="utf-8")
        refs.append(ref)
        lines.append({"page_id": f"A{i}", "set": "A", "image": f"pages/A{i}.jpg", "gt": f"gt/A{i}.txt",
                      "flags": {"two_column": i < 10, "special": True}, "source": {"file": "x.pdf", "page": i}})
    for i in range(2):
        lines.append({"page_id": f"B{i}", "set": "B", "image": f"pages/B{i}.jpg", "gt": None, "flags": {},
                      "source": {"title": "Sách", "page": i}})
    (root / "pages.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in lines), encoding="utf-8")
    (root / "selection.json").write_text(json.dumps({"corpus_pages": {"stbook": 1_000_000, "giao_trinh_ocr": 0}}))
    return refs


def _write_run(root: Path, name: str, texts: dict[str, str], pps: float = 1.0, finish: str | None = "stop") -> None:
    """Ghi outputs.jsonl giả cho một engine, t_end cách đều 1/pps giây."""
    d = root / "runs" / name
    d.mkdir(parents=True)
    with open(d / "outputs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"page_id": pid, "text": text, "raw": text, "finish_reason": finish, "error": None,
                                "session": "s1", "t_start": 1000 + (k - 1) / pps, "t_end": 1000 + k / pps}, ensure_ascii=False) + "\n" for k, (pid, text) in enumerate(texts.items()))


def test_score_va_quy_tac_chon(tmp_path):
    """
    baseline mất ký tự đặc biệt và dấu; good gần hoàn hảo; swap đảo hai câu ở trang hai cột (bị loại vì thứ tự đọc);
    loopy lặp ở 1 trang (bị loại vì tỷ lệ trang hỏng). Kết luận: good.
    """
    refs = _fake_bakeoff(tmp_path)
    ids = [f"A{i}" for i in range(len(refs))] + ["B0", "B1"]
    full = refs + ["Sách quét trang một", "Sách quét trang hai"]
    _write_run(tmp_path, "baseline", {p: strip_diacritics(t).replace("²", "?").replace("“", '"').replace("”", '"')
                                      .replace("–", "-") for p, t in zip(ids, full)})
    _write_run(tmp_path, "good", dict(zip(ids, full)), pps=2.0)
    swapped = [t.replace(f"{SENT_L} {SENT_R}", f"{SENT_R} {SENT_L}") if i < 10 else t for i, t in enumerate(full)]
    _write_run(tmp_path, "swap", dict(zip(ids, swapped)), pps=5.0)
    loopy = list(full)
    loopy[20] = full[20] + " lặp lại mãi thôi" * 10
    _write_run(tmp_path, "loopy", dict(zip(ids, loopy)), pps=5.0)
    s = score_bakeoff(tmp_path, ["baseline", "good", "swap", "loopy"], vocab=frozenset({"sách", "quét", "trang"}),
                      gates=Gates(hour_budget=None))
    a = s["set_a"]
    assert a["good"]["cer"] == 0 and a["good"]["special_recall"] == 1.0
    assert a["baseline"]["special_recall"] == 0 and a["baseline"]["cer_nodiac"] < a["baseline"]["cer"]
    assert a["swap"]["order_gap_two_column"] > 0.02 and s["gates"]["swap"]["checks"]["order_gap_two_column"]["status"] == "fail"
    assert s["gates"]["loopy"]["checks"]["bad_rate"]["status"] == "fail"
    assert a["loopy"]["bad_by_reason"] == {"loop": 1}
    d = s["decision"]
    assert d["status"] == "winner" and d["winner"] == "good" and d["notes"]
    assert d["comparisons"]["good - baseline"]["ci95"][1] < 0
    assert s["set_b"]["good"]["unknown_syllable_rate"] == pytest.approx(2 / 8)
    rep = tmp_path / "report"
    assert (rep / "report.html").read_text(encoding="utf-8").count("<details>") == 2 + 10
    assert "page_id" in (rep / "per_page.csv").read_text(encoding="utf-8").splitlines()[0]
    # giới hạn giờ chạy: 1e6 trang / (2 trang/s x 4 GPU) / 3600 ~ 34,7 giờ > 30 -> good trượt cổng tốc độ -> phương án C
    s = score_bakeoff(tmp_path, ["baseline", "good", "swap", "loopy"], gates=Gates(hour_budget=30))
    h = s["hours"]["good"]
    assert s["gates"]["good"]["corpus_hours"] == h["corpus_hours"] == pytest.approx(1e6 / (2.0 * 4) / 3600, rel=0.05)
    assert h["corpus_gpu_hours"] == pytest.approx(4 * h["corpus_hours"])
    assert h["corpus_hours_by_source"]["stbook"] == pytest.approx(h["corpus_hours"])
    assert h["bakeoff_pages"] == 32 and h["bakeoff_run_hours"] == pytest.approx(32 / 2.0 / 3600)
    assert s["decision"]["status"] == "option_c"
    rows = (tmp_path / "report/hours.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("engine,bakeoff_pages,bakeoff_run_hours") and len(rows) == 5
    assert "Số giờ chạy" in (tmp_path / "report/report.html").read_text(encoding="utf-8")


def test_hoa_cer_chon_engine_nhanh_hon(tmp_path):
    """Hai ứng viên cùng sai đúng một ký tự mỗi trang (hiệu CER = 0, khoảng tin cậy chứa 0): chọn engine nhanh hơn."""
    refs = _fake_bakeoff(tmp_path)
    ids = [f"A{i}" for i in range(len(refs))]
    _write_run(tmp_path, "baseline", {p: t[:-5] for p, t in zip(ids, refs)})
    _write_run(tmp_path, "slow", {p: t[:-1] for p, t in zip(ids, refs)}, pps=1.0)
    _write_run(tmp_path, "fast", {p: t[:-1] + "x" for p, t in zip(ids, refs)}, pps=3.0)
    d = score_bakeoff(tmp_path, ["baseline", "slow", "fast"])["decision"]
    assert d["winner"] == "fast" and d["tied_with"]


def test_trang_hong_va_cat_cut(tmp_path):
    """Lý do trang hỏng; finish_reason=length tính là cắt cụt; tốc độ bỏ trang khởi động."""
    sc = page_scores("", "abc")
    assert page_bad_reason(None, sc) == "no_output"
    assert page_bad_reason({"text": ""}, sc) == "empty"
    assert page_bad_reason({"text": "ab"}, page_scores("abcdef", "abc")) == "too_long"
    refs = _fake_bakeoff(tmp_path)
    ids = [f"A{i}" for i in range(len(refs))]
    _write_run(tmp_path, "cut", dict(zip(ids, refs)), pps=4.0, finish="length")
    s = score_bakeoff(tmp_path, ["cut"])
    assert s["set_a"]["cut"]["trunc_rate"] == 1.0
    t = throughput(tmp_path / "runs/cut", warmup=5)
    assert t["pages"] == len(refs) - 5 and t["pages_per_s"] == pytest.approx(4.0) and not t["steady"]


def test_openai_engine_goi_api(tmp_path, monkeypatch):
    """Thân yêu cầu có ảnh base64 + text_prefix + lời nhắc, temperature 0; đọc nội dung, finish_reason, usage."""
    img = tmp_path / "p.jpg"
    Image.new("RGB", (10, 10)).save(img)
    sent = {}

    class Resp:
        """Phản hồi giả."""

        def raise_for_status(self):
            """Không lỗi."""

        def json(self):
            """Thân phản hồi chat completions."""
            return {"choices": [{"message": {"content": "Xin chào"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    cfg = {"type": "openai", "base_url": "http://x/v1", "model": "m", "served_name": "model", "prompt": "Đọc.",
           "text_prefix": "<img>", "max_tokens": 99, "output": "markdown"}
    e = eng.OpenAIEngine(cfg)
    monkeypatch.setattr(e.session, "post", lambda url, json, timeout: sent.update(url=url, body=json) or Resp())
    res = e.run(img)
    assert res == {"raw": "Xin chào", "finish_reason": "length", "prompt_tokens": 7, "completion_tokens": 3}
    body = sent["body"]
    assert sent["url"] == "http://x/v1/chat/completions" and body["model"] == "model" and body["temperature"] == 0
    content = body["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,") and content[1]["text"] == "<img>Đọc."


def test_run_engine_chay_tiep_va_ghi_loi(tmp_path, monkeypatch):
    """Trang lỗi ghi trường error và được làm lại ở lần chạy sau; trang đã xong không chạy lại."""
    _fake_bakeoff(tmp_path, n=3)
    calls = []

    class Flaky:
        """Engine giả: lần đầu lỗi ở trang A1."""

        def __init__(self, cfg):
            """Không nạp gì."""

        def run(self, path):
            """Trả tên file làm nội dung."""
            calls.append(path.stem)
            if path.stem == "A1" and calls.count("A1") == 1:
                raise RuntimeError("hết bộ nhớ")
            return {"raw": f"**{path.stem}**", "finish_reason": None}

    monkeypatch.setitem(eng.ENGINE_TYPES, "fake", Flaky)
    r = eng.run_engine(tmp_path, "f", {"type": "fake", "output": "markdown"})
    assert r["pages"] == 5 and r["errors"] == 1
    r = eng.run_engine(tmp_path, "f", {"type": "fake", "output": "markdown"})
    assert r["pages"] == 1 and r["errors"] == 0 and calls.count("A1") == 2
    outs = [json.loads(x) for x in (tmp_path / "runs/f/outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert outs[-1]["text"] == "A1" and outs[-1]["error"] is None
    sessions = json.loads((tmp_path / "runs/f/run_info.json").read_text(encoding="utf-8"))["sessions"]
    assert len(sessions) == 2 and all("ended" in x and "seconds" in x for x in sessions)
    with pytest.raises(ValueError):
        eng.run_engine(tmp_path, "g", {"type": "lạ"})
