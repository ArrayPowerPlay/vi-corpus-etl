"""
Test biên do reviewer viết cho các bản sửa F1-F3 (kho dấu vân tay vòng 1, vòng 2, MinHash cả văn bản). Không cần mạng.
"""

import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tests.test_pipeline import SOURCES, _row, make_cfg, vi_text
from vi_corpus.pipeline import dedup
from vi_corpus.pipeline.dedup import (
    dedup_rows,
    has_index,
    index_rows,
    minhash,
    write_index,
)
from vi_corpus.pipeline.dedup_global import (
    apply_verdict,
    global_verdict,
    read_index,
    write_verdict,
)


def _naive_minhash(text: str) -> np.ndarray:
    """MinHash tính một lần trên mọi shingle (không chia khối) để so với bản chia khối."""
    words = text.lower().split()
    n = max(len(words) - dedup.SHINGLE + 1, 1)
    x = np.array([zlib.crc32(" ".join(words[i:i + dedup.SHINGLE]).encode("utf-8")) for i in range(n)], dtype=np.uint64)
    return ((dedup._A[:, None] * x[None, :] + dedup._B[:, None]) & dedup._MASK).min(axis=1)


def test_minhash_chia_khoi_bang_tinh_mot_lan():
    """Chia khối SHINGLE_BLOCK không đổi kết quả, kể cả khi số shingle không chia hết cho cỡ khối."""
    text = " ".join(vi_text(1, i) for i in range(1500))  # > 2 khối
    assert len(text.split()) > 2 * dedup.SHINGLE_BLOCK
    assert np.array_equal(minhash(text), _naive_minhash(text))


def test_minhash_dung_bien_toi_thieu():
    """Đúng MIN_FUZZY_WORDS từ thì có chữ ký; ít hơn một từ thì None."""
    words = [f"w{i}" for i in range(dedup.MIN_FUZZY_WORDS)]
    assert minhash(" ".join(words)) is not None
    assert minhash(" ".join(words[:-1])) is None


def test_ten_run_co_ky_tu_glob_khong_xoa_file_run_khac(tmp_path):
    """Tên run do người dùng đặt (--run-name) có '[' ']' không được khớp nhầm file của run khác khi dọn / kiểm."""
    cfg = make_cfg()
    rows, _ = dedup_rows([_row("sea:1", vi_text(30, 1))], cfg)
    write_index(tmp_path, index_rows(rows, cfg, "r1"), "r1", ["sea_pile_v2"])
    other = tmp_path / "state" / "dedup_index" / "sea_pile_v2" / "r1.parquet"
    assert other.exists()
    st, _ = dedup_rows([_row("stbook:1", vi_text(30, 2), key="stbook")], cfg)
    write_index(tmp_path, index_rows(st, cfg, "r[12]"), "r[12]", ["stbook"])
    assert other.exists(), "file kho của run 'r1' bị xoá bởi run 'r[12]'"
    assert has_index(tmp_path, "r[12]")


def test_read_index_file_kieu_cu_khong_co_alive_written_at(tmp_path):
    """File kho thiếu cột alive / written_at / rights_status vẫn đọc được, coi là còn sống và cũ nhất."""
    d = tmp_path / "sea_pile_v2"
    d.mkdir(parents=True)
    sig = minhash(vi_text(30, 3)).astype(np.uint32).tobytes()
    pq.write_table(pa.table({"doc_id": ["a"], "source_key": ["sea_pile_v2"], "dedup_group": ["cpt"],
                             "doc_sha256": ["h"], "minhash": [sig], "quality_score": [90.0], "priority": [1]}),
                   d / "old.parquet")
    items = read_index(tmp_path)
    assert [it["doc_id"] for it in items] == ["a"]
    verdict, stats = global_verdict(items, 0.8)
    assert verdict[0]["status"] == "kept" and stats["cpt"]["largest_family"] == 1


def test_vong_2_enforce_chan_het_va_van_ban_ngan():
    """enforce chặn mọi văn bản: không lỗi, largest_family = 0; văn bản ngắn (minhash None) chỉ so exact."""
    items = [{"doc_id": "a", "source_key": "stbook", "dedup_group": "cpt", "doc_sha256": "h1", "minhash": None,
              "quality_score": 90.0, "priority": 2, "rights_status": "unknown"}]
    verdict, stats = global_verdict(items, 0.8, rights_gate="enforce")
    assert verdict[0]["status"] == "rights" and stats["cpt"]["largest_family"] == 0
    items += [{**items[0], "doc_id": "b", "quality_score": 50.0}]
    verdict, stats = global_verdict(items, 0.8)
    v = {x["doc_id"]: x for x in verdict}
    assert v["a"]["status"] == "kept" and v["b"]["status"] == "duplicate" and v["b"]["dup_kind"] == "exact"


def test_apply_verdict_doan_sach_theo_parent(tmp_path):
    """Verdict theo văn bản áp vào mọi đoạn (parent_doc_id) của sách; đoạn đã bị loại ở vòng 1 không bị đụng."""
    rows = [_row(f"B#c{i}", vi_text(30, 50 + i), key="stbook") for i in range(3)]
    for i, r in enumerate(rows):
        r["parent_doc_id"] = "B"
        r["status"], r["dedup_family_id"] = "kept", "fam_old"
    rows[2]["status"] = "rejected:quality"
    path = tmp_path / "v.parquet"
    write_verdict(path, [{"doc_id": "B", "source_key": "stbook", "dedup_group": "cpt", "status": "duplicate",
                          "dup_kind": "fuzzy", "dup_of": "A", "dedup_family_id": "fam_A", "family_size": 2}])
    out = apply_verdict(rows, path)
    assert [r["status"] for r in rows] == ["rejected:duplicate", "rejected:duplicate", "rejected:quality"]
    assert rows[0]["dup_kind"] == "global_fuzzy" and rows[0]["dedup_family_id"] == "fam_A"
    assert out == {"rejected": 2, "refamily": 2, "missing": 0, "superseded": 0}


def test_runner_bat_lai_dedup_index_ghi_kho_khong_chay_lai_dedup(tmp_path):
    """Chạy với dedup_index=False rồi bật lại: kho được ghi từ 05_dedup.parquet, stage dedup không chạy lại."""
    from tests.test_finalize_parsers import _web_root
    from vi_corpus.pipeline.runner import run_pipeline

    root = _web_root(tmp_path, n=10)
    run = root / "processed" / "run"
    kw = {"mix": {"sea_pile_v2": 10}, "embed_spec": None}
    run_pipeline(SOURCES, root, run, make_cfg(**kw, dedup_index=False), with_report=False)
    assert not has_index(root, "run")
    mtime = (run / "05_dedup.parquet").stat().st_mtime_ns
    run_pipeline(SOURCES, root, run, make_cfg(**kw), with_report=False)
    assert (run / "05_dedup.parquet").stat().st_mtime_ns == mtime
    assert has_index(root, "run")
    n_docs = len({r.get("parent_doc_id") or r["doc_id"]
                  for r in pq.read_table(run / "05_dedup.parquet").to_pylist()})
    table = pq.read_table(root / "state" / "dedup_index" / "sea_pile_v2" / "run.parquet")
    assert table.num_rows == n_docs  # mọi văn bản: còn sống + bia mộ


def test_kho_khong_cu_hon_05_dedup_sau_khi_tat_bat_dedup_index(tmp_path):
    """Dedup chạy lại (đổi tham số) với dedup_index=False rồi bật lại: kho của run phải khớp 05_dedup.parquet mới, không
    giữ dòng của cấu hình cũ."""
    from tests.test_finalize_parsers import _web_root, prose
    from vi_corpus.pipeline.runner import run_pipeline

    root = _web_root(tmp_path, n=10, gen=prose)
    run = root / "processed" / "run"
    kw = {"mix": {"sea_pile_v2": 10}, "embed_spec": None}
    run_pipeline(SOURCES, root, run, make_cfg(**kw), with_report=False)
    assert read_index(root / "state" / "dedup_index")
    tight = {**kw, "keep_bands": ("Z",)}  # mọi văn bản bị loại chất lượng
    run_pipeline(SOURCES, root, run, make_cfg(**tight, dedup_index=False), with_report=False)
    run_pipeline(SOURCES, root, run, make_cfg(**tight), with_report=False)
    kept = [r for r in pq.read_table(run / "05_dedup.parquet").to_pylist() if r["status"] == "kept"]
    assert kept == []
    assert read_index(root / "state" / "dedup_index") == [], "kho còn văn bản sống của cấu hình dedup cũ"


def test_vong_2_ghi_dropped_va_apply_dem_superseded(tmp_path):
    """Văn bản lần chạy mới nhất đã loại (bia mộ) ra status "dropped"; áp verdict vào lần chạy cũ còn giữ nó thì đoạn
    giữ nguyên, đếm superseded (không đếm missing)."""
    cfg = make_cfg()
    text = vi_text(30, 61)
    old, _ = dedup_rows([_row("sea:1", text, 90)], cfg)
    write_index(tmp_path, index_rows(old, cfg, "a"), "a", ["sea_pile_v2"])
    new, _ = dedup_rows([_row("sea:1", text, 10, band="D")], cfg)
    write_index(tmp_path, index_rows(new, cfg, "b"), "b", ["sea_pile_v2"])
    items = read_index(tmp_path / "state" / "dedup_index", with_dropped=True)
    verdict, stats = global_verdict(items, 0.8)
    assert [v["status"] for v in verdict] == ["dropped"] and stats["cpt"]["dropped"] == 1
    path = tmp_path / "v.parquet"
    write_verdict(path, verdict)
    out = apply_verdict(old, path)
    assert old[0]["status"] == "kept" and out["superseded"] == 1 and out["missing"] == 0


def test_xoa_ket_qua_dedup_thi_xoa_kho_cua_run(tmp_path):
    """Đổi tham số dedup rồi chạy --until quality: 05_dedup bị xoá thì kho của run cũng bị xoá (không để kho cũ sống)."""
    from tests.test_finalize_parsers import _web_root, prose
    from vi_corpus.pipeline.runner import run_pipeline

    root = _web_root(tmp_path, n=10, gen=prose)
    run = root / "processed" / "run"
    kw = {"mix": {"sea_pile_v2": 10}, "embed_spec": None}
    run_pipeline(SOURCES, root, run, make_cfg(**kw), with_report=False)
    assert has_index(root, "run")
    run_pipeline(SOURCES, root, run, make_cfg(**kw, fuzzy_threshold=0.9), with_report=False, until="quality")
    assert not (run / "05_dedup.parquet").exists() and not has_index(root, "run")


def test_c07_menu_web_co_dau_gach_dung_khong_phai_bang():
    """C-07 chỉ miễn văn bản có code / bảng: dòng menu web "A | B | C" (breadcrumb, chân trang) không phải bảng, nên mục A
    vẫn xoá dòng ngắn lặp trong bài web có menu."""
    from vi_corpus.pipeline.lines import has_code_or_table

    web = ("Trang chủ | Tin tức | Thể thao\n" + "\n".join(f"{s}\nXem thêm" for s in [vi_text(3, i) for i in range(4)])
           + "\nGiới thiệu | Liên hệ | Quảng cáo")
    assert not has_code_or_table(web)


def test_c07_van_xuoi_tieng_anh_xuong_dong_khong_phai_code():
    """Giáo trình tiếng Anh (giữ theo P-08) trích từ PDF xuống dòng giữa câu: dòng bắt đầu bằng "for" / "if" / "case" và
    kết thúc bằng ")" / ":" không phải code."""
    from vi_corpus.pipeline.lines import has_code_or_table

    prose = ("The model is trained in two stages. The first stage is described below, and the second\n"
             "for the remaining parameters (see Section 3.2)\n"
             "is fine-tuned on labelled data. The loss is minimised, but only\n"
             "if the validation error decreases (Figure 4)\n"
             "and the learning rate is reduced. We consider the special\n"
             "case of a single layer, which gives the following result:\n"
             "the gradient is bounded.")
    assert not has_code_or_table(prose)


JAVA = ("Lớp Hình chữ nhật trong Java:\n"
        "public class HinhChuNhat {\n    private double dai;\n    private double rong;\n\n"
        "    public double dienTich() {\n        return dai * rong;\n    }\n\n"
        "    public double chuVi() {\n        return 2 * (dai + rong);\n    }\n\n"
        "    public String toString() {\n        return \"HCN\";\n    }\n}")
PYTHON = ("Hàm tính tổng trong Python:\n"
          "def tong(n):\n    s = 0\n    for i in range(n):\n        s += i\n    return s\n\n"
          "class Diem:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y")


def test_c07_code_java_va_python_thuc_te_duoc_nhan_dien():
    """C-07: đoạn Java / Python thường gặp trong giáo trình (thân hàm có gán, return không có ";" kiểu Python) phải được
    nhận là code."""
    from vi_corpus.pipeline.lines import has_code_or_table

    assert has_code_or_table(JAVA)
    assert has_code_or_table(PYTHON)


def test_c07_code_java_khong_mat_dau_dong_ngoac():
    """Văn bản có code Java không bị mục A xoá dòng "}" lặp (lý do của quyết định C-07)."""
    from vi_corpus.pipeline.lines import clean_rows

    rows = [{"doc_id": "j", "source_key": "stbook", "text": JAVA}]
    clean_rows(rows, make_cfg())
    assert rows[0]["text"].count("}") == JAVA.count("}")
