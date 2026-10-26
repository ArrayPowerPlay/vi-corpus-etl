"""
Test sửa theo nguyên nhân gốc cho pipeline VJOL (J-25, J-26): danh sách từ không dấu sinh từ dữ liệu (A), tiêu đề trang lặp (C), mục / khối tham khảo theo mẫu đầu dòng (E),
khuôn mẫu "ngày ..." và số điện thoại có mã nước (F), font mất ư / ơ thành "-" (B), bộ nhớ đệm OCR theo trang (Bước 2), từ lạ đếm theo tạp chí (D1), perplexity (D3),
producers_docs.csv và nối theo producer (D2), kiểm lần chạy / bộ đáp án / tập chữ đúng (Bước 0). Không cần mạng, GPU hay mô hình.
"""

import csv
import json
import pytest

from tests.test_ocr_vlm import GOOD, make_cascade, ok
from tests.test_pipeline import make_cfg, vi_text
from tests.test_vjol import REPO, page_no, scan_pdf, text_pdf, write_ckpt
from vi_corpus.common.garbled import (DEFAULT_THRESHOLDS, UNACCENTED_WORDS, hyphen_lost, load_unaccented, page_measures, page_signals, plan_garbled,
                                      unaccented_ratio, unaccented_tag)
from vi_corpus.common.ocr_cache import OcrCache, ocr_signature
from vi_corpus.common.ocr_cascade import CascadeOcrGetter, CascadePageOcr
from vi_corpus.common.pdf_text import extract_pdf
from vi_corpus.common.references import is_reference_entry, reference_share
from vi_corpus.common.state import write_json_atomic
from vi_corpus.common.unaccented import build_unaccented, save_unaccented
from vi_corpus.pipeline import io as pio
from vi_corpus.pipeline import quality as Q
from vi_corpus.pipeline.config import PROFILES
from vi_corpus.vjol import audit as A
from vi_corpus.vjol import extract as vx
from vi_corpus.vjol.clean import (clean_paper_pages, is_ref_heading, heading_key, mask_phones, strip_boilerplate, strip_running_headers,
                                  strip_trailing_references)
from vi_corpus.vjol.diagnose import chunks_by_producer, pdf_producers
from vi_corpus.vjol.ocr_seed import seed_cache

ECON = " ".join([
    "Doanh nghiệp vừa và nhỏ cần được cung cấp vốn vay với lãi suất ưu đãi để mở rộng sản xuất kinh doanh trong giai đoạn khó khăn.",
    "Chính sách cung tiền của ngân hàng trung ương tác động trực tiếp đến lạm phát và lãi suất thị trường liên ngân hàng.",
    "Doanh thu thuần của các công ty niêm yết tăng đều qua các quý nhờ nhu cầu tiêu dùng nội địa phục hồi mạnh mẽ.",
    "Quan hệ cung cầu hàng hóa nông sản phụ thuộc nhiều vào thời tiết, chi phí vận chuyển và các hiệp định thương mại.",
    "Nghiên cứu sử dụng mô hình hồi quy dữ liệu bảng để đánh giá ảnh hưởng của vốn con người đến tăng trưởng kinh tế khu vực.",
    "Kết quả ước lượng cho thấy doanh số bán lẻ trực tuyến có mối tương quan dương với thu nhập bình quân đầu người.",
    "Các tổ chức tín dụng phải duy trì tỷ lệ an toàn vốn tối thiểu theo quy định để bảo đảm khả năng chi trả lâu dài.",
    "Chính phủ ban hành nhiều giải pháp giảm thuế, giãn nợ và hỗ trợ lãi suất nhằm tháo gỡ khó khăn cho người lao động.",
    "Hiệu quả sử dụng vốn đầu tư công còn thấp do thủ tục kéo dài và việc giải ngân chậm ở nhiều địa phương.",
    "Từ những phân tích trên, bài viết đề xuất một số khuyến nghị chính sách nhằm nâng cao năng lực cạnh tranh của doanh nghiệp.",
])
MED = " ".join([
    "Bệnh nhân được chẩn đoán xác định bằng xét nghiệm huyết thanh học và chụp cắt lớp vi tính vùng ngực.",
    "Phác đồ điều trị phối hợp hai loại kháng sinh trong mười bốn ngày cho kết quả lâm sàng tốt ở đa số trường hợp.",
    "Tỷ lệ biến chứng sau phẫu thuật thấp hơn đáng kể ở nhóm được chuẩn bị tiền phẫu đầy đủ theo hướng dẫn.",
    "Nghiên cứu cắt ngang được thực hiện trên một trăm hai mươi người cao tuổi sống tại khu vực nông thôn.",
    "Huyết áp tâm thu trung bình giảm rõ rệt sau tám tuần can thiệp bằng chế độ ăn giảm muối và vận động đều đặn.",
    "Các tác dụng không mong muốn như buồn nôn, chóng mặt và mệt mỏi đều nhẹ và tự hết khi ngừng thuốc.",
    "Việc theo dõi đường huyết tại nhà giúp người bệnh điều chỉnh liều insulin phù hợp với từng bữa ăn.",
    "Hồi cứu hồ sơ bệnh án cho thấy thời gian nằm viện trung bình ngắn hơn so với các báo cáo trước đây.",
    "Siêu âm tim qua thành ngực là phương pháp không xâm lấn có giá trị cao trong đánh giá chức năng thất trái.",
    "Kết luận, can thiệp sớm và giáo dục sức khỏe cộng đồng góp phần giảm gánh nặng bệnh tật lâu dài.",
])


# ---------------------------------------------------------------------------------------------------------------- A: danh sách từ không dấu


def test_danh_sach_toi_thieu_khong_con_doanh_va_cung_va_chu_dung_khong_bi_co():
    """J-25 A: 'doanh', 'cung' là từ đúng không dấu nên không nằm trong danh sách; đoạn kinh tế có dấu đầy đủ không bị gắn mất dấu."""
    assert "doanh" not in UNACCENTED_WORDS and "cung" not in UNACCENTED_WORDS
    assert unaccented_ratio(ECON)[0] == 0.0
    assert page_signals(ECON) == []
    lost = "Cua nhung nguoi nghien cuu duoc cung cap " * 10 + ECON[:200]
    assert unaccented_ratio(lost)[0] > 0.1  # J-28: mỗi từ chỉ tính tối đa 3 lần nên tỷ lệ thấp hơn bản cũ (>0,2) với đoạn lặp một câu


def test_sinh_danh_sach_tu_khong_dau_tu_du_lieu_loai_tu_dung_va_tu_nuoc_ngoai():
    """Danh sách sinh từ kho: 'cua', 'nguoi' vào; 'doanh', 'cung' (dạng trần chiếm đa số) và 'the' (từ tiếng Anh) không vào; văn bản không dấu hoàn toàn không làm bẩn số đếm."""
    clean = "Của những người nghiên cứu được thế giới biết đến. Doanh nghiệp cung cấp dịch vụ, doanh số và cung cầu tăng nhanh. Thể chế và thế hệ trẻ. " * 10
    bare_noise = ["khong dau hoan toan cua nguoi " * 30] * 5  # văn bản không dấu: bị loại khỏi phép đếm
    english = ["The economy and the market of the world. " * 30] * 5
    res = build_unaccented([clean] * 40 + bare_noise + english, min_accented=20, top=50)
    words = set(res["words"])
    assert {"cua", "nguoi", "duoc"} <= words
    assert not words & {"doanh", "cung", "the", "khong dau"}
    assert res["docs"] == 40 and res["counts"]["cua"][1] == 0


def test_dung_danh_sach_sinh_ra_trong_bo_do_va_dau_vet_phien_ban(tmp_path):
    """load_unaccented đọc file; page_signals dùng danh sách truyền vào; unaccented_tag đổi khi danh sách đổi."""
    path = tmp_path / "unaccented.json"
    bare = "toi ban anh chi em ong ba co chu bac con cau".split()
    save_unaccented(path, {"words": bare, "docs": 1})
    words = load_unaccented(path)
    assert words == frozenset(bare) and load_unaccented(tmp_path / "khong_co.json") is None
    text = (" ".join(bare) + " " + vi_text(4, 3) + " ") * 10
    assert "nodiac" in page_signals(text, unaccented=words) and "nodiac" not in page_signals(text)
    assert unaccented_tag(words) != unaccented_tag(frozenset({"toi"})) and unaccented_tag(None) == ""
    v = vx.version_for(None, words)
    ver = vx.EXTRACT_VERSION
    assert v == f"{ver}+unacc{unaccented_tag(words)}" and vx.version_is_current(v) and vx.version_is_current(f"{ver}+vocab") and not vx.version_is_current("3+vocab")
    assert v + "+vocab" in vx.accepted_versions(frozenset({"a"}), words) and v in vx.accepted_versions(None, words)
    assert ver not in vx.accepted_versions(frozenset({"a"}), words)  # danh sách khác thì checkpoint cũ bị làm lại


# ---------------------------------------------------------------------------------------------------------------- B: font mất ư / ơ


def test_gach_noi_thay_u_o_bat_duoc_va_khong_bat_tu_ghep_hop_le():
    """'đ-ợc', 'ng-ời', 'th- viện' là chữ mất ư / ơ; 'e-mail', 'x-ray', 'k-means', 'N-acetyl', 'co-operate' thì không."""
    bad = "Để đ-ợc th- viện và ng-ời Việt tr-ờng học kh-ơng ph-ơng"
    assert hyphen_lost(bad)[2] == 6
    for ok_text in ("e-mail x-ray k-means N-acetyl co-operate hand-made ba-ri-e", "mô hình được ưu tiên người trường học"):
        assert hyphen_lost(ok_text)[2] == 0


def test_tin_hieu_hyphen_dua_trang_di_ocr_va_ma_cung_lost_letters():
    """Trang có nhiều chỗ mất ư / ơ bị nghi (signal hyphen); đoạn có >= 5 chỗ bị mã cứng lost_letters; đoạn sạch thì không."""
    page = (vi_text(10, 1) + " Những đ-ợc th- viện ng-ời tr-ờng ph-ơng h-ớng ") * 3
    assert "hyphen" in page_signals(page)
    assert plan_garbled([page, vi_text(40, 2), vi_text(40, 3), vi_text(40, 4)]) == ["garbled_text_layer", None, None, None]
    m = Q.compute_metrics(page)
    assert m["hyphen_lost"] >= 5
    _, codes = Q.score_metrics(m, "vi", PROFILES["vjol"], 1.0, "{}")
    assert "lost_letters" in codes
    _, clean_codes = Q.score_metrics(Q.compute_metrics(ECON), "vi", PROFILES["vjol"], 1.0, "{}")
    assert "lost_letters" not in clean_codes and "lost_diacritics" not in clean_codes


# ---------------------------------------------------------------------------------------------------------------- C: tiêu đề trang lặp


def rand_line(seed: int, n: int = 12) -> str:
    """Dòng gồm n từ giả toàn chữ cái, khác nhau giữa các seed (để thân bài các trang không tình cờ trùng tiền tố)."""
    import random

    rng = random.Random(seed)
    return " ".join("".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(3, 7))) for _ in range(n))


def test_bo_tieu_de_trang_lap_ke_ca_khi_dinh_dau_dong_va_giu_than_bai():
    """Tiêu đề 'Tạp chí Dân tộc học số N - 2005' dính trước thân bài ở mọi trang bị cắt (không cần tên tạp chí trong metadata); thân bài của từng trang giữ nguyên."""
    body = lambda i: "\n".join(rand_line(i * 100 + k) for k in range(8))  # noqa: E731
    pages = [f"Tạp chí Dân tộc học số {3 + i} - 2005 {rand_line(i)}\n{body(i)}\n{40 + i}" for i in range(6)]
    out, n = strip_running_headers(pages)
    assert n >= 6 and all("Tạp chí Dân tộc học" not in p for p in out)
    assert out[2].startswith(rand_line(2)) and rand_line(203) in out[2]
    short, k = strip_running_headers(pages[:2])  # dưới 3 trang thì không đủ bằng chứng
    assert k == 0 and short == pages[:2]
    cleaned, stats = clean_paper_pages(pages, None)
    assert stats["running_headers"] >= 6 and not any("Tạp chí Dân tộc học" in p for p in cleaned)


def test_tieu_de_trang_khong_an_nham_van_ban_khong_lap():
    """Các trang có dòng đầu khác nhau hoàn toàn thì không bị cắt gì."""
    pages = [f"{rand_line(i)}\n{rand_line(i + 50)}\n{rand_line(i + 90)}" for i in range(1, 7)]
    assert strip_running_headers(pages) == (pages, 0)


# ---------------------------------------------------------------------------------------------------------------- E: tham khảo


def test_muc_tham_khao_nhan_theo_mau_dau_dong_khong_nhan_tho_co_trich_dan():
    """Mục tham khảo phải bắt đầu bằng mẫu tham khảo; thơ / văn xuôi có '(Họ, 1997, tr.623)' không bị nhận."""
    entries = ["[1] Nguyễn Văn A (2010), Kinh tế học, NXB Giáo dục, tr. 12.", "Smith, J. K. (1999). Journal of Economics, 12, pp. 3-9.",
               "Nguyen V.A., Tran T.B. (2004). Kinh tế vĩ mô. NXB Chính trị quốc gia, tr. 5."]
    assert all(is_reference_entry(e) for e in entries)
    assert not is_reference_entry("Một câu thơ buồn bã rơi xuống (Hồ, 1997, tr.623) rồi tan trong gió chiều.")
    assert not is_reference_entry("Tài liệu tham khảo 2015")
    lines = [entries[0], "tiếp theo của mục một vì dài quá", entries[1], "Bài báo thân bài bình thường dài hơn ba mươi từ " * 6]
    assert reference_share(lines) == pytest.approx(0.75)


def test_tieu_de_tham_khao_song_ngu_co_dau_gach_cheo_va_cat_khoi_cuoi_bai_khong_tieu_de():
    """'TÀI LIỆU THAM KHẢO/ REFERENCES' là tiêu đề; khối >= 6 mục tham khảo ở cuối bài không có tiêu đề bị cắt, thân bài giữ nguyên."""
    assert is_ref_heading(heading_key("TÀI LIỆU THAM KHẢO/ REFERENCES")) and not is_ref_heading(heading_key("Tài liệu / phụ lục khác"))
    body = [f"Đoạn thân bài số {i} nói về một vấn đề khác của nghiên cứu này rất cụ thể {i * 13}." for i in range(12)]
    refs = [f"[{i}] Nguyễn Văn {chr(65 + i)} (20{10 + i}), Nghiên cứu số {i}, NXB Giáo dục, tr. {i + 3}." for i in range(1, 9)]
    pages = ["\n".join(body[:6]), "\n".join(body[6:] + refs)]
    out, n = strip_trailing_references(pages)
    assert n == 8 and "[1] Nguyễn" not in "\n".join(out) and "Đoạn thân bài số 11" in out[1]
    only_body, k = strip_trailing_references(["\n".join(body)])
    assert k == 0 and only_body == ["\n".join(body)]


# ---------------------------------------------------------------------------------------------------------------- F: khuôn mẫu và điện thoại


def test_dong_ngay_chung_va_so_dien_thoai_co_ma_nuoc_khong_dau_cong():
    """'Ngày phản biện: 12/05/2015' bị bỏ; '84-983226880' bị che; số liệu bảng '84 500 000 000', '84.567.890.123' không bị che."""
    out, stats = strip_boilerplate(["Ngày phản biện: 12/05/2015\nNgày hoàn thiện 3.4.2015\nNgày 5 tháng 6 năm 2010 chính phủ ban hành nghị quyết mới về giáo dục\nNội dung"])
    assert out[0].splitlines()[0].startswith("Ngày 5 tháng 6 năm 2010") and "Nội dung" in out[0] and stats["boilerplate_lines"] == 2
    line, n = mask_phones("Liên hệ 84-983226880 hoặc 0912345678; vốn 84 500 000 000 đồng, dân số 84.567.890.123")
    assert n == 2 and "84 500 000 000" in line and "84.567.890.123" in line and "983226880" not in line


# ---------------------------------------------------------------------------------------------------------------- Bước 2: bộ nhớ đệm OCR


def _getter(scripts, calls):
    """CascadeOcrGetter giả (đã có CascadePageOcr dựng sẵn) và đếm số trang gửi cho engine."""
    def engine(im):
        calls.append(page_no(im))
        return ok(f"Trang {page_no(im)}. " + GOOD)

    cascade = make_cascade(engine)
    getter = CascadeOcrGetter([], 2, enabled=True)
    getter._ocr = CascadePageOcr(cascade, concurrency=2)
    return getter


def test_cache_ocr_khong_ocr_lai_trang_da_co_va_khong_goi_get_ocr(tmp_path):
    """Lần 1 OCR 3 trang scan và ghi cache; lần 2 (cùng chữ ký) không gọi engine lẫn get_ocr; đổi chữ ký thì OCR lại."""
    pdf = tmp_path / "raw/vjol/2019/7_1.pdf"
    scan_pdf(pdf, 3)
    sig = ocr_signature(["t0"], False)
    calls: list[int] = []
    getter = _getter(None, calls)
    first = extract_pdf(pdf, getter, ocr_cache=OcrCache(tmp_path / "c.json", sig))
    assert sorted(calls) == [0, 1, 2] and [p["method"] for p in first] == ["ocr_vlm"] * 3

    def must_not_build():
        raise AssertionError("get_ocr không được gọi khi mọi trang đã có trong cache")

    second = extract_pdf(pdf, must_not_build, ocr_cache=OcrCache(tmp_path / "c.json", sig))
    assert second == first and sorted(calls) == [0, 1, 2]
    again = extract_pdf(pdf, _getter(None, calls), ocr_cache=OcrCache(tmp_path / "c.json", ocr_signature(["t0"], True)))
    assert len(calls) == 6 and again == first  # chữ ký khác (có xoay thử) -> bỏ cache cũ, OCR lại


def test_extract_file_dung_cache_theo_sha_khi_getter_co_chu_ky(tmp_path):
    """extract_file tự đặt cache theo sha256 PDF khi get_ocr có signature(); đổi EXTRACT_VERSION rồi trích lại không OCR lại trang scan."""
    raw = tmp_path / "raw/vjol"
    scan_pdf(raw / "2019/7_1.pdf", 2)
    calls: list[int] = []
    getter = _getter(None, calls)
    getter.tiers = []  # chữ ký rỗng nhưng có
    assert vx.run_extract(raw, tmp_path, getter) == 0 and len(calls) == 2
    ckpt = next((tmp_path / vx.TEXT_DIR).glob("*.json"))
    data = json.loads(ckpt.read_text(encoding="utf-8"))
    write_json_atomic(ckpt, {**data, "extract_version": "3+vocab"})  # giả như checkpoint của phiên bản cũ
    assert vx.run_extract(raw, tmp_path, _getter(None, calls)) == 0
    assert len(calls) == 2  # cache trúng, không OCR lại
    assert json.loads(ckpt.read_text(encoding="utf-8"))["extract_version"].startswith(vx.EXTRACT_VERSION)


def test_nap_cache_tu_checkpoint_cu_bo_trang_co_the_chua_xoay_thu(tmp_path):
    """seed_cache: trang ocr_vlm thành mục cache theo cờ vẽ lại; với chữ ký có :rot thì bỏ qua trang baseline / flagged."""
    pages = [{"text": "a", "method": "ocr_vlm", "tier": "t0", "status": "ok"},
             {"text": "b", "method": "ocr_vlm", "tier": "baseline", "status": "ok"},
             {"text": "", "method": "ocr_vlm", "tier": "t1", "status": "flagged", "rejected_text": "x"},
             {"text": "c", "method": "ocr_vlm", "tier": "t0", "status": "ok", "reason": "garbled_text_layer", "signals": ["tone"], "garbled_text": "zz"},
             {"text": "d", "method": "text"}]
    write_ckpt(tmp_path, "ab" * 32, "raw/vjol/2019/7_1.pdf", [])
    ck = vx.checkpoint_path(tmp_path, "ab" * 32)
    write_json_atomic(ck, {**json.loads(ck.read_text(encoding="utf-8")), "pages": pages})
    st = seed_cache(tmp_path, ocr_signature(["t0", "t1", "baseline"], True))
    assert st["pages_seeded"] == 2 and st["pages_skipped_rotation"] == 2
    cache = OcrCache.for_pdf(tmp_path, "ab" * 32, ocr_signature(["t0", "t1", "baseline"], True))
    assert cache.get(0, False)["text"] == "a" and cache.get(3, True)["text"] == "c" and "signals" not in cache.get(3, True)
    assert cache.get(1, False) is None and cache.get(3, False) is None
    st2 = seed_cache(tmp_path, ocr_signature(["t0", "t1", "baseline"], False))
    assert st2["pages_seeded"] == 4 and st2["pages_skipped_rotation"] == 0


# ---------------------------------------------------------------------------------------------------------------- D1, D3: quality


def _row(i, key, text, parent, journal=None, lang="vi"):
    """Một đoạn giả đã qua language, có meta tạp chí."""
    return {"doc_id": f"{key}:{i}#c0", "parent_doc_id": parent, "source_key": key, "text": text, "language": lang, "lang_score": 1.0, "lang_mix": "{}",
            "meta": json.dumps({"journal_slug": journal, "article_id": str(i)}) if journal else None}


def _vocab_cfg(tmp_path, base):
    """RunConfig với tập âm tiết dựng từ `base`."""
    from tests.test_j22_fixes import make_vocab

    vocab = make_vocab(base)
    syl = tmp_path / "syllables.json"
    syl.write_text(json.dumps({"syllables": sorted(vocab)}), encoding="utf-8")
    return make_cfg(syllables=str(syl))


def test_tu_la_dem_theo_tap_chi_loi_lap_trong_mot_tap_chi_van_la_la(tmp_path):
    """J-25 D1: font lỗi lặp lại ở 5 bài của CÙNG một tạp chí vẫn bị coi là từ lạ (đếm theo bài thì lọt); thuật ngữ gặp ở >= 3 tạp chí thì không."""
    base = vi_text(8, 5)
    cfg = _vocab_cfg(tmp_path, base)
    garbage = " ".join(f"xqz{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(60))
    rows = [_row(i, "vjol", base + " " + garbage, f"P{i}", "badj") for i in range(5)]  # cùng một lỗi lặp ở 5 bài của một tạp chí
    rows += [_row(10 + j, "vjol", base + " mekong", f"Q{j}", f"j{j}") for j in range(10)]  # 10 tạp chí, "mekong" ở cả 10
    Q.annotate_quality(rows, cfg)
    stats = Q.annotate_odd_words(rows, cfg)
    assert stats["unit"] == "journal" and stats["journals"] == 11
    assert all("odd_words" in r["reason_codes"] for r in rows[:5]) and all("odd_words" not in r["reason_codes"] for r in rows[5:])
    by_doc = [_row(i, "vjol", base + " " + garbage, f"P{i}", None) for i in range(5)]  # không có tạp chí: quay về đếm theo bài, lặp ở 5 bài nên coi là từ thật
    Q.annotate_quality(by_doc, cfg)
    assert Q.annotate_odd_words(by_doc, cfg)["unit"] == "doc"
    assert all("odd_words" not in r["reason_codes"] for r in by_doc)


class FakeScorer:
    """Bộ chấm perplexity giả: giá trị cho sẵn theo thứ tự."""

    def __init__(self, values):
        """Nhận danh sách giá trị."""
        self.values = values

    def score(self, texts):
        """Trả giá trị cho sẵn (cùng độ dài texts)."""
        return list(self.values)[:len(texts)]


def test_perplexity_la_tin_hieu_thu_hai_khong_phai_cong_duy_nhat(tmp_path):
    """J-25 D3: perplexity cao một mình chỉ gắn high_perplexity (trừ điểm); kèm từ lạ vừa phải (>= ODD_SOFT) mới thành odd_words (loại); tắt khi chưa đặt mô hình."""
    base = vi_text(8, 5)
    cfg = _vocab_cfg(tmp_path, base)
    cfg = type(cfg)(**{**cfg.__dict__, "perplexity_model": "fake"})
    rows = [_row(i, "vjol", base, f"P{i}", f"j{i}") for i in range(3)]
    Q.annotate_quality(rows, cfg)
    for r, odd in zip(rows, (0.0, 0.1, 0.1)):  # từ lạ vừa phải ở hai đoạn sau
        r["quality_metrics"] = json.dumps({**json.loads(r["quality_metrics"]), "odd_word": odd})
    stats = Q.annotate_perplexity(rows, cfg, FakeScorer([400.0, 30.0, 400.0]))
    assert stats["checked"] == 3 and stats["high"] == 2 and stats["flagged"] == 1
    assert "high_perplexity" in rows[0]["reason_codes"] and "odd_words" not in rows[0]["reason_codes"]  # perplexity cao một mình: chỉ trừ điểm
    assert "odd_words" not in rows[1]["reason_codes"] and "high_perplexity" not in rows[1]["reason_codes"]  # từ lạ vừa phải một mình: không loại
    assert "odd_words" in rows[2]["reason_codes"] and rows[2]["quality_band"] == "D"  # hai tín hiệu cùng lúc
    assert Q.annotate_perplexity(rows, make_cfg(), None)["skipped"] == "perplexity_model chưa đặt"


def test_ref_line_dung_mau_dau_dong_nen_tho_khong_bi_gan_nhan():
    """Đoạn thơ trích '(Họ, 1997, tr.623)' không có nhãn reference_list; danh mục thật có."""
    poem = "\n".join(f"Câu thơ số {i} buồn man mác (Hồ, 199{i}, tr.62{i}) rồi tan." for i in range(8))
    refs = "\n".join(f"[{i}] Nguyễn Văn {chr(65 + i)} (20{10 + i}), Kinh tế học, NXB Giáo dục, tr. {i + 3}." for i in range(8))
    assert Q.compute_metrics(poem)["ref_line"] == 0.0 and Q.compute_metrics(refs)["ref_line"] >= Q.REF_LINE_TAG


# ---------------------------------------------------------------------------------------------------------------- D2: producers


def test_producers_docs_mot_dong_moi_bai_va_noi_voi_doan_xau(tmp_path):
    """producers_docs.csv có một dòng mỗi bài; chunks_by_producer đếm đoạn nghi ngờ còn lọt theo producer qua meta.article_id."""
    import pymupdf

    root = tmp_path
    pdf = root / "raw/vjol/2019/7_1.pdf"
    text_pdf(pdf, [vi_text(6, 1)])
    with pymupdf.open(pdf) as d:
        d.set_metadata({"producer": "iLovePDF", "creator": "Word"})
        d.saveIncr()
    write_ckpt(root, "ab" * 32, "raw/vjol/2019/7_1.pdf", [vi_text(8, 2)])
    pdf_producers(root)
    with open(root / "processed/vjol_diagnose/producers_docs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["article_id"] == "7" and rows[0]["producer"] == "iLovePDF"
    run = root / "processed/pipeline_runs/r"
    good = {"doc_id": "vjol:x:0#c0", "source_key": "vjol", "text": vi_text(4, 3), "quality_band": "A", "reason_codes": [], "meta": json.dumps({"article_id": "7"}),
            "quality_metrics": json.dumps({"odd_word": 0.0})}
    bad = {**good, "doc_id": "vjol:x:0#c1", "quality_metrics": json.dumps({"odd_word": 0.3, "unaccented": 0.05})}
    pio.write_rows(run / "05_dedup.parquet", [good, bad])
    out = chunks_by_producer(root, run)
    with open(out, encoding="utf-8") as f:
        res = list(csv.DictReader(f))
    assert res[0]["producer"] == "iLovePDF" and res[0]["chunks"] == "2" and res[0]["bad_chunks"] == "1" and "unaccented" in res[0]["top_causes"]
    with pytest.raises(FileNotFoundError):
        chunks_by_producer(tmp_path / "khac", run)


# ---------------------------------------------------------------------------------------------------------------- Bước 0: kiểm, bộ đáp án, tập chữ đúng


def _quality_row(i, text, band="A", journal="j1", year=2010, codes=(), **metrics):
    """Bản ghi giả đã qua quality, có meta tạp chí / năm."""
    return {"doc_id": f"vjol:{i}:0#c0", "parent_doc_id": f"vjol:{i}:0", "source_key": "vjol", "text": text, "quality_band": band, "reason_codes": list(codes),
            "quality_metrics": json.dumps(metrics), "meta": json.dumps({"article_id": str(i), "journal_slug": journal, "year": year})}


def test_audit_bao_loi_con_lot_tap_chi_bat_thuong_va_luat_don():
    """audit_rows: đếm email / '-' thay ư / tiêu đề trang trong đoạn giữ; nhóm tạp chí bị loại vượt 3 lần trung vị và >= 15% được báo; đếm đoạn chỉ bị một luật."""
    rows = [_quality_row(i, vi_text(3, i), journal=f"j{i % 5}") for i in range(100)]
    rows[0]["text"] += " liên hệ abc@x.vn"
    rows[1]["text"] += " Tạp chí Dân tộc học số 3 - 2005 "
    rows[2]["quality_metrics"] = json.dumps({"hyphen_lost": 2})
    rows += [_quality_row(1000 + i, vi_text(3, i), band="D", journal="econst", codes=["lost_diacritics"]) for i in range(30)]
    rows += [_quality_row(2000 + i, vi_text(3, i), band="A", journal="econst") for i in range(30)]
    res = A.audit_rows(rows, min_group=20)
    assert res["error_counts"]["email"] == 1 and res["error_counts"]["mid_header"] == 1 and res["error_counts"]["hyphen_lost"] == 1
    assert res["single_rule_drops"] == {"lost_diacritics": 30}
    assert [a["key"] for a in res["anomalies"] if a["level"] == "journal"] == ["econst"]
    assert res["samples"]["kept:email"][0]["doc_id"] == "vjol:0:0#c0"


def test_bo_dap_an_chia_hai_nua_theo_bai_va_do_bo_do():
    """split_of ổn định theo bài (mọi đoạn của một bài cùng nửa); evaluate_labels tính precision / recall theo nhãn và dương tính giả trên nhãn clean."""
    assert A.split_of("123", 0) == A.split_of("123", 0) and {A.split_of(i, 0) for i in range(40)} == {"tune", "holdout"}
    rows = [_quality_row(i, vi_text(3, i), journal=f"j{i % 7}", odd_word=0.2 if i % 10 == 0 else 0.0) for i in range(200)]
    items = A.make_labeling_sample(rows, n=60, seed=1)
    assert len(items) == 60 and all(it["split"] == A.split_of(it["article_id"], 1) for it in items)
    for it in items:
        it["label"] = "font_misread" if json.loads(it["quality_metrics"]).get("odd_word", 0) >= 0.2 else "clean"
    res = A.evaluate_labels(items)
    assert res["detectors"]["font_misread"]["tune"]["precision"] in (1.0, None) and res["clean_false_flags"]["tune"]["font_misread"] == 0


def test_tap_chu_dung_khong_bi_co_va_phat_hien_loi_neu_co():
    """check_clean_set: đoạn kinh tế có dấu đầy đủ (có 'doanh', 'cung') đạt '0 cờ'; đoạn mất dấu hoặc mất ư/ơ bị báo."""
    ok_items = [{"doc_id": "a", "text": ECON}, {"doc_id": "b", "text": MED}]
    assert A.check_clean_set(ok_items) == []
    bad = A.check_clean_set([{"doc_id": "x", "text": ("Cua nhung nguoi nghien cuu duoc " * 6 + vi_text(10, 1))},
                             {"doc_id": "y", "text": (vi_text(10, 1) + " đ-ợc th- viện ng-ời tr-ờng ph-ơng h-ớng ") * 3}])
    assert {b["doc_id"] for b in bad} == {"x", "y"}
    rows = [_quality_row(i, ECON if i % 2 else MED, vi_diacritic=0.25) for i in range(10)]
    assert 0 < len(A.make_clean_set(rows, n=5)) <= 5


# ---------------------------------------------------------------------------------------------------------------- dòng lệnh


def test_dong_lenh_audit_groundtruth_seed_cache_va_diagnose(tmp_path):
    """Các script mới chạy được qua dòng lệnh trên dữ liệu giả: audit_run, groundtruth (sample / clean-set / check-clean), seed_ocr_cache, build_unaccented --help."""
    import subprocess
    import sys

    run = tmp_path / "run"
    rows = [_quality_row(i, ECON if i % 2 else MED, journal=f"j{i % 4}", vi_diacritic=0.25) for i in range(40)]
    pio.write_rows(run / "04_quality.parquet", rows)

    def sh(*args):
        """Chạy một script của repo, trả về tiến trình hoàn tất."""
        return subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True, timeout=120)

    p = sh("scripts/vjol/audit_run.py", "--run-dir", str(run), "--min-group", "5")
    assert p.returncode in (0, 1) and (run / "audit/audit.json").is_file() and (run / "audit/audit.md").is_file(), p.stderr
    p = sh("scripts/vjol/groundtruth.py", "sample", "--run-dir", str(run), "--n", "20")
    assert p.returncode == 0 and (run / "groundtruth/labeling_sample.csv").is_file(), p.stderr
    p = sh("scripts/vjol/groundtruth.py", "clean-set", "--run-dir", str(run), "--n", "10")
    assert p.returncode == 0, p.stderr
    p = sh("scripts/vjol/groundtruth.py", "check-clean", "--clean-set", str(run / "groundtruth/clean_set.jsonl"), "--data-root", str(tmp_path))
    assert p.returncode == 0 and "0 bị gắn cờ" in p.stdout, p.stdout + p.stderr
    p = sh("scripts/vjol/seed_ocr_cache.py", "--data-root", str(tmp_path))
    assert p.returncode == 0 and "pages_seeded" in p.stdout, p.stderr
    p = sh("scripts/build_unaccented.py", "--help")
    assert p.returncode == 0
