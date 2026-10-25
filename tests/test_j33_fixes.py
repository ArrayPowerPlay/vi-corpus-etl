"""
Kiểm các sửa J-33 cho VJOL, rút ra từ kết quả chạy lại S-17 (vjol_10k_v5): khôi phục "ư" cho font gạch nối (A), trang không có lớp chữ nhưng có nội dung / trang trắng (B), câu rác
của OCR VLM (C), năm mặc định 1970 (D), tiêu đề tạp chí chen giữa câu (E), trang ban biên tập (F), dòng ngắn lặp trong bảng (G), phép đo họ lỗi đọc font và độ lệch tạp chí (H, J).
Không cần mạng, GPU hay mô hình (OCR được thay bằng giả).
"""

import json
from pathlib import Path

import pymupdf

from tests.test_garbled import ocr_engine
from tests.test_pipeline import vi_text
from tests.test_ocr_vlm import make_cascade, ok
from tests.test_vjol import scan_pdf, write_ckpt
from vi_corpus.common.ocr_cascade import CascadePageOcr
from vi_corpus.common.registry import SOURCES
from vi_corpus.common.garbled import GARBLED_REASONS, REASON_BLANK, REASON_NO_TEXT
from vi_corpus.common.ocr_noise import strip_vlm_noise
from vi_corpus.common.pdf_text import extract_pdf, page_has_ink
from vi_corpus.pipeline.lines import clean_lines_in_doc
from vi_corpus.vjol.audit import journal_balance, misread_hits, year_checks
from vi_corpus.vjol.clean import clean_paper_pages, drop_front_matter, is_editorial_page, strip_running_headers
from vi_corpus.vjol.hyphen_fix import restore_u_in_pages, restore_u_in_text
from vi_corpus.vjol.records import iter_records, normalize_year

# Tập âm tiết giả đủ cho các ví dụ thật của J-33 A.
VOCAB = frozenset("trường người được nhưng tư thư viện hướng nước dưới cư dân liệu môi con và các nam á kinh tế xã hội".split())


# ---------------------------------------------------------------------------------------------------------------- A: khôi phục "ư"


def test_khoi_phuc_u_cac_vi_du_that():
    """Các ví dụ thật của J-33: "môi tr-ờng", "con ng-ời", "đ-ợc", "Nh-ng", "t- liệu", "h- ớng", "D-ỚI", "N-ớc"."""
    cases = {"môi tr-ờng": "môi trường", "con ng-ời": "con người", "đ-ợc": "được", "Nh-ng": "Nhưng", "t- liệu": "tư liệu", "h- ớng": "hướng",
             "D-ỚI": "DƯỚI", "N-ớc": "Nước", "c- dân": "cư dân", "th- viện": "thư viện"}
    vocab = VOCAB | {"đ" + "ược"}
    for src, want in cases.items():
        fixed, n = restore_u_in_text(f"Một {src} khác.", vocab)
        assert fixed == f"Một {want} khác." and n == 1, (src, fixed)


def test_khong_sua_gach_noi_that():
    """"Nam-Á", "C-section", "kinh tế-xã hội", "x-ray", "e-mail", "N- TiO2" giữ nguyên dù có âm tiết tương tự trong tập."""
    for text in ("Nam-Á", "C-section delivery", "kinh tế-xã hội", "x-ray", "e-mail", "N- TiO2 samples", "COVID-19", "t-test"):
        assert restore_u_in_text(text, VOCAB) == (text, 0), text


def test_chi_sua_khi_ket_qua_la_am_tiet_that():
    """Kết quả không có trong tập âm tiết thì không sửa ("tr-ờng" khi tập thiếu "trường")."""
    assert restore_u_in_text("tr-ờng", frozenset({"khác"})) == ("tr-ờng", 0)
    assert restore_u_in_text("tr-ờng", VOCAB)[0] == "trường"


def test_chi_sua_khi_ca_bai_co_du_cho():
    """Bài chỉ có 1-2 chỗ lẻ (gạch nối thật) không bị sửa; bài có từ 3 chỗ trở lên (font hỏng) được sửa ở mọi trang, kể cả trang chỉ có 1 chỗ."""
    few = ["Cuộc sống ở tr-ờng học.", "Nội dung khác."]
    assert restore_u_in_pages(few, VOCAB) == (few, 0)
    many = ["Môi tr-ờng sống của con ng-ời.", "Điều này đ-ợc nêu rõ.", "Nh-ng tr-ờng hợp khác."]
    out, n = restore_u_in_pages(many, VOCAB | {"đ" + "ược"})
    assert n == 5 and out[0] == "Môi trường sống của con người." and out[2] == "Nhưng trường hợp khác."
    assert restore_u_in_pages(many, None) == (many, 0)  # thiếu tập âm tiết thì bỏ bước


def test_clean_paper_pages_khoi_phuc_va_dem():
    """clean_paper_pages khôi phục "ư" khi có vocab và ghi số chỗ sửa vào thống kê; không có vocab thì giữ nguyên."""
    pages = [f"Môi tr-ờng sống của con ng-ời. {vi_text(40, 1)}", f"Nh-ng tr-ờng hợp khác thì sao. {vi_text(40, 2)}", f"Điều đ-ợc nêu. {vi_text(40, 3)}"]
    vocab = VOCAB | {"đ" + "ược"}
    out, st = clean_paper_pages(pages, None, vocab)
    assert st["hyphen_restored"] == 5 and "trường" in out[0] and "-" not in "".join(out)
    out2, st2 = clean_paper_pages(pages, None, None)
    assert st2["hyphen_restored"] == 0 and "tr-ờng" in out2[0]


# ---------------------------------------------------------------------------------------------------------------- C: câu rác của VLM


def test_loc_cau_rac_cua_vlm():
    """Câu "There is no text in the image." bị bỏ ở đầu, giữa, cuối dòng và nguyên dòng; câu văn bình thường nhắc "no text" giữ nguyên."""
    assert strip_vlm_noise("There is no text in the image.") == ("", 1)
    assert strip_vlm_noise("There is no text on this page image.\nChương 1") == ("Chương 1", 1)
    assert strip_vlm_noise("Mở đầu. There is no text in the image. Nội dung tiếp") == ("Mở đầu. Nội dung tiếp", 1)
    assert strip_vlm_noise("The image is blank.")[1] == 1
    text = "The reviewers found there is no text evidence for this claim, and no content analysis was done."
    assert strip_vlm_noise(text) == (text, 0)
    assert strip_vlm_noise("Trang này nói về hình ảnh và văn bản.") == ("Trang này nói về hình ảnh và văn bản.", 0)


def test_clean_paper_pages_bo_cau_rac_vlm():
    """Câu rác của VLM lọt vào giữa bài (checkpoint cũ) cũng bị bỏ lúc làm sạch, kèm số đếm."""
    pages = ["Phần mở đầu của bài. There is no text in the image. Phần tiếp theo của bài.", vi_text(30, 1), vi_text(30, 2)]
    out, st = clean_paper_pages(pages)
    assert "no text" not in " ".join(out) and st["vlm_noise_removed"] == 1


# ---------------------------------------------------------------------------------------------------------------- B: trang không có lớp chữ


def vector_pdf(path: Path) -> None:
    """PDF 3 trang: trang 0 chữ, trang 1 chỉ có nét vẽ (không chữ, không ảnh nhúng), trang 2 trắng."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_htmlbox(pymupdf.Rect(30, 30, 565, 810), f"<p style='font-size:9pt'>{vi_text(40, 1)}</p>")
    page = doc.new_page()
    for y in range(100, 700, 12):
        page.draw_rect(pymupdf.Rect(60, y, 500, y + 6), color=(0, 0, 0), fill=(0, 0, 0))
    doc.new_page()
    doc.save(path)
    doc.close()


def test_trang_net_ve_duoc_ocr_trang_trang_ghi_ly_do(tmp_path):
    """rescue_empty: trang chỉ có nét vẽ được OCR vẽ lại cả trang (reason no_text_layer), trang trắng là empty kèm blank_page; tắt rescue_empty thì y như trước (empty, không lý do)."""
    pdf = tmp_path / "v.pdf"
    vector_pdf(pdf)
    with pymupdf.open(pdf) as doc:
        assert not page_has_ink(doc[2]) and page_has_ink(doc[1]) and not doc[1].get_images()
    seen: list = []
    pages = extract_pdf(pdf, ocr_engine(seen), detect_garbled=True, rescue_empty=True)
    assert [p["method"] for p in pages] == ["text", "ocr_vlm", "empty"]
    assert pages[1]["reason"] == REASON_NO_TEXT and pages[1]["text"] and "garbled_text" not in pages[1]
    assert pages[2]["reason"] == REASON_BLANK and pages[2]["text"] == "" and len(seen) == 1
    assert REASON_NO_TEXT not in GARBLED_REASONS and REASON_BLANK not in GARBLED_REASONS
    legacy = extract_pdf(pdf, lambda: None, detect_garbled=True)
    assert [p["method"] for p in legacy] == ["text", "empty", "empty"] and "reason" not in legacy[1]


def test_trang_net_ve_khong_ocr_duoc_thanh_needs_ocr(tmp_path):
    """Không có bộ OCR thì trang có nội dung nhưng không có lớp chữ là needs_ocr kèm lý do (không im lặng thành empty)."""
    pdf = tmp_path / "v.pdf"
    vector_pdf(pdf)
    pages = extract_pdf(pdf, lambda: None, detect_garbled=True, rescue_empty=True)
    assert [p["method"] for p in pages] == ["text", "needs_ocr", "empty"] and pages[1]["reason"] == REASON_NO_TEXT


def test_trang_chi_con_cau_rac_thanh_blank(tmp_path):
    """Trang scan mà VLM chỉ trả "There is no text in the image." thành text rỗng, status blank, ghi vlm_noise_removed."""
    pdf = tmp_path / "s.pdf"
    scan_pdf(pdf, 2)
    get_ocr = lambda: CascadePageOcr(make_cascade(lambda im: ok("There is no text in the image. " * 12)), concurrency=1)  # noqa: E731
    pages = extract_pdf(pdf, get_ocr, rescue_empty=True)
    assert all(p["text"] == "" and p["status"] == "blank" and p["vlm_noise_removed"] >= 1 for p in pages)


# ---------------------------------------------------------------------------------------------------------------- D: năm 1970


def test_nam_1970_la_thieu_nam():
    """normalize_year bỏ 1970 (mốc 0 Unix), giá trị rỗng / không phải số / ngoài khoảng; năm hợp lệ giữ nguyên, nhận cả chuỗi."""
    assert normalize_year(1970) is None and normalize_year("1970") is None
    assert normalize_year(None) is None and normalize_year("") is None and normalize_year("abc") is None and normalize_year(1200) is None
    assert normalize_year(2019) == 2019 and normalize_year(" 2015 ") == 2015


def test_ban_ghi_khong_gan_nam_1970(tmp_path):
    """Bài ở thư mục 1970 mà metadata không có năm thì year là None (không phải 1970); bài thư mục 2015 mang year 2015."""
    for year_dir, sha in ((1970, "a" * 64), (2015, "b" * 64)):
        write_ckpt(tmp_path, sha, f"{year_dir}/{year_dir}00_10.pdf", [vi_text(40, 1), vi_text(40, 2), vi_text(40, 3)])
    rows = {json.loads(r["meta"])["year_dir"]: json.loads(r["meta"]) for r in iter_records(SOURCES["vjol"], tmp_path, with_meta=True)}
    assert rows[1970]["year"] is None and rows[2015]["year"] == 2015


# ---------------------------------------------------------------------------------------------------------------- E: tiêu đề chen giữa câu


def test_cat_tieu_de_hoc_duoc_o_moi_vi_tri():
    """Tiêu đề lặp ở đầu các trang sau được học rồi cắt cả khi dính giữa đoạn và ở trang đầu (không lặp theo vị trí); câu trích dẫn có dấu câu khác giữ nguyên."""
    header = "Tạp chí Dân tộc học số 3 - 2005"
    pages = [f"Mở đầu bài viết về dân tộc. {header} Nội dung chính của trang đầu tiên.",
             f"{header}\nTrang hai nói về văn hóa và phong tục. {header.replace('3', '4')} Đoạn tiếp theo bị chen tiêu đề.",
             f"{header}\nTrang ba bàn về lễ hội truyền thống của người Mạ.",
             f"{header}\nTrang bốn kết luận: cần bảo tồn. Xem thêm Tạp chí Dân tộc học, số 3 (2005)."]
    out, cut = strip_running_headers(pages)
    joined = "\n".join(out)
    assert "Tạp chí Dân tộc học số" not in joined and cut >= 5
    assert "Mở đầu bài viết về dân tộc. Nội dung chính của trang đầu tiên." in out[0]
    assert "Đoạn tiếp theo bị chen tiêu đề." in out[1] and "Tạp chí Dân tộc học, số 3 (2005)" in out[3]


# ---------------------------------------------------------------------------------------------------------------- F: trang ban biên tập


EDITORIAL = ("TẠP CHÍ KHOA HỌC\nHỘI ĐỒNG BIÊN TẬP\nGS.TS Nguyễn Văn An - Chủ tịch hội đồng\nTổng biên tập: PGS.TS Trần Thị Bình\nThư ký tòa soạn: Lê Văn Cường\n"
             "Giấy phép xuất bản số 123/GP-BTTTT\nISSN 1859-0000\nTòa soạn: 12 Phố Huế, Hà Nội")


def test_nhan_trang_ban_bien_tap():
    """Trang ngắn có >= 3 loại chức danh / mục là trang ban biên tập; trang bài dài hoặc chỉ nhắc một chức danh thì không."""
    assert is_editorial_page(EDITORIAL)
    assert not is_editorial_page("Theo ban biên tập của nhiều tạp chí, bài này " + "nội dung bình thường của bài báo " * 60)
    assert not is_editorial_page("Tổng biên tập quyết định đăng bài này.")
    body = vi_text(40, 1)
    out, st = drop_front_matter([EDITORIAL, body, body])
    assert out[0] == "" and st["editorial_pages"] == 1 and out[1] == body
    out, st = drop_front_matter([body, body, EDITORIAL])  # trang cuối
    assert out[2] == "" and st["editorial_pages"] == 1


# ---------------------------------------------------------------------------------------------------------------- G: ô bảng bị xoá dòng lặp


def test_dong_ngan_lap_trong_bang_duoc_giu():
    """Ô bảng lặp ("Tự túc lương thực.") nằm giữa các ô ngắn khác được giữ khi keep_cells; tiêu đề lặp kề đoạn dài và mặc định (keep_cells=False) vẫn bị xoá."""
    cells = "\n\n".join(["Tự túc lương thực.", "Long Nh.", "Tự túc lương thực.", "Tân L.", "Tự túc lương thực.", "Biên Ch.", "Tự túc lương thực."])
    kept, removed = clean_lines_in_doc(cells, 10, 3, keep_cells=True)
    assert kept.count("Tự túc lương thực.") == 4 and not removed
    dropped, removed = clean_lines_in_doc(cells, 10, 3)
    assert dropped.count("Tự túc lương thực.") == 1 and len(removed) == 3
    para = "Đây là một đoạn văn dài hơn nhiều so với một ô bảng, bàn về những điều cần thiết cho bài viết hôm nay."
    heading = f"Kết luận\n\n{para}\n\nKết luận\n\n{para} thêm\n\nKết luận\n\n{para} nữa"
    kept, removed = clean_lines_in_doc(heading, 10, 3, keep_cells=True)
    assert kept.count("Kết luận") == 1 and len(removed) == 2


# ---------------------------------------------------------------------------------------------------------------- H, J: đo


def test_do_ho_loi_doc_font():
    """misread_hits đếm từ không có trong tập âm tiết nhưng sửa được bằng một phép thay ê <- ấ / ờ, â <- õ, á <- ỏ; từ đúng và từ lạ không sửa được thì không tính."""
    vocab = frozenset("nghiên cứu bệnh nhân khánh hướng".split())
    assert misread_hits("NGHIấN cứu, bệnh nhõn Khỏnh", vocab) == 3
    assert misread_hits("nghiên cứu bệnh nhân khánh", vocab) == 0
    assert misread_hits("xyzấ qwe", vocab) == 0


def test_do_lech_tap_chi_va_nam():
    """journal_balance báo tạp chí lớn nhất, tỷ lệ loại theo đoạn và theo tạp chí; year_checks đếm đoạn thiếu năm, năm 1970 và năm có hơn nửa đoạn không có metadata."""
    def row(journal, band, year=2019, year_dir=2019, found=True):
        """Bản ghi tối thiểu có meta."""
        return {"quality_band": band, "meta": json.dumps({"journal_slug": journal, "year": year, "year_dir": year_dir, "metadata_found": found})}

    rows = [row("big", "A") for _ in range(30)] + [row("big", "D") for _ in range(10)] + [row("small", "D") for _ in range(20)]
    groups = {"big": {"chunks": 40, "dropped_rate": 0.25}, "small": {"chunks": 20, "dropped_rate": 1.0}}
    jb = journal_balance(rows, groups, min_group=20)
    assert jb["top"][0] == ["big", round(40 / 60, 4)] and jb["max_share"] == round(40 / 60, 4)
    assert jb["dropped_rate_micro"] == 0.5 and jb["dropped_rate_macro"] == 0.625 and jb["macro_journals"] == 2
    yrows = [row("a", "A", year=None, year_dir=1970, found=False) for _ in range(25)] + [row("a", "A", year=1970) for _ in range(2)] + [row("a", "A") for _ in range(10)]
    yc = year_checks(yrows)
    assert yc["missing"] == 25 and yc["year_1970"] == 2 and yc["default_1970"] == 27 and yc["no_metadata_years"] == {"1970": 1.0}


# ---------------------------------------------------------------------------------------------------------------- end to end


def test_pipeline_font_gach_noi_het_lost_letters_khi_co_tap_am_tiet(tmp_path):
    """
    Checkpoint VJOL mang lỗi "ư" -> "-" qua ingest + pipeline: có syllables.json thì bản ghi được khôi phục và không bị mã lost_letters (band D); không có file đó thì chữ giữ nguyên và
    bị lost_letters (tức bước khôi phục là thứ cứu được đoạn). Tập âm tiết nằm trong vân tay đầu vào của ingest.
    """
    from vi_corpus.pipeline.ingest import input_files
    from vi_corpus.pipeline.runner import run_pipeline
    from vi_corpus.vjol.extract import SYLLABLES_REL
    from tests.test_pipeline import make_cfg
    from vi_corpus.pipeline import io as pio

    bad = "Môi tr-ờng sống, con ng-ời, đ-ợc nêu, Nh-ng tr-ờng hợp, h-ớng dẫn. " * 3
    for k in range(3):
        write_ckpt(tmp_path, f"{k}" * 64, f"2019/{k}00_1.pdf", [vi_text(8, 10 * k) + " " + bad, vi_text(8, 10 * k + 1) + " " + bad, vi_text(8, 10 * k + 2) + " " + bad])
    cfg = make_cfg(mix={"vjol": 6}, chunk_target_tokens=400, chunk_max_tokens=600, chunk_min_tokens=80, embed_spec=None)

    run_off = tmp_path / "run_off"
    run_pipeline(SOURCES, tmp_path, run_off, cfg)
    off = pio.read_rows(run_off / "04_quality.parquet")
    assert any("tr-ờng" in r["text"] for r in off) and all("lost_letters" in r["reason_codes"] for r in off)

    syl = tmp_path / SYLLABLES_REL
    syl.parent.mkdir(parents=True)
    words = sorted(VOCAB | {"đ" + "ược", "hướng", "dẫn", "sống", "hợp", "nêu"})
    syl.write_text(json.dumps({"syllables": words}, ensure_ascii=False), encoding="utf-8")
    assert syl in input_files(SOURCES["vjol"], tmp_path, chunked=True)
    run_on = tmp_path / "run_on"
    run_pipeline(SOURCES, tmp_path, run_on, cfg)
    on = pio.read_rows(run_on / "04_quality.parquet")
    assert on and all("tr-ờng" not in r["text"] and "ng-ời" not in r["text"] and "lost_letters" not in r["reason_codes"] for r in on)
    assert json.loads(on[0]["meta"])["hyphen_restored"] > 0


def test_so_sanh_hai_lan_chay():
    """compare_audits ghi hiệu mới - cũ cho số đoạn, tỷ lệ lỗi lọt và năm 1970; lỗi chỉ có ở một bên ghi "-"."""
    from vi_corpus.vjol.audit import compare_audits

    old = {"chunks": 100, "kept": 90, "dropped": 10, "error_rates": {"hyphen_lost": 0.04, "phone": 0.0}, "single_rule_drops": {"lost_letters": 7}, "anomalies": [1, 2],
           "year_checks": {"missing": 0, "year_1970": 33}, "journal_balance": {"max_share": 0.266, "dropped_rate_micro": 0.1, "dropped_rate_macro": None}}
    new = {"chunks": 100, "kept": 98, "dropped": 2, "error_rates": {"hyphen_lost": 0.0, "font_misread": 0.01}, "single_rule_drops": {}, "anomalies": [],
           "year_checks": {"missing": 33, "year_1970": 0}, "journal_balance": {"max_share": 0.266, "dropped_rate_micro": 0.02, "dropped_rate_macro": 0.05}}
    table = compare_audits(old, new, "v5", "v6")
    assert "lỗi lọt: hyphen_lost" in table and "-4.00%" in table and "lỗi lọt: font_misread" in table
    line = next(ln for ln in table.splitlines() if ln.startswith("năm: year=1970"))
    assert line.split("|")[1].strip() == "33" and line.split("|")[2].strip() == "0" and line.split("|")[3].strip() == "-33"
