"""
Danh sách nguồn ứng viên cần kiểm chất lượng (docs/sources/SOURCE_CANDIDATES.md) và cách đọc mẫu của từng nguồn.

Mỗi ProbeSpec mô tả một nguồn: đọc ở đâu (kind + location), trường văn bản tiếng Việt, trường bản gốc tiếng Anh (nếu là dữ liệu
dịch máy), các cột số đo sẵn có của bộ dữ liệu cần giữ lại để so sánh, và hồ sơ đo (web hay tài liệu).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeSpec:
    """
    Một nguồn cần lấy mẫu.

    Attributes:
        name:        Tên ngắn, dùng làm tên thư mục kết quả.
        kind:        "hf_parquet" (đọc row group của file parquet trên HF), "hf_rows" (API datasets-server, cho bộ JSON),
                     "hplt" (đầu các shard jsonl.zst của HPLT 3.0, đã xếp theo điểm WDS).
        location:    hf_parquet: đường dẫn HfFileSystem tới thư mục chứa parquet; hf_rows: id dataset; hplt: danh sách URL shard.
        text_field:  Trường văn bản tiếng Việt cần đo. Có thể là "a+b" để nối nhiều trường (hỏi + đáp).
        en_field:    Trường bản gốc tiếng Anh (dữ liệu dịch máy) để so độ dài và tìm phần chưa dịch; rỗng nếu không có.
        extra:       Cột số đo sẵn có của bộ dữ liệu cần giữ (điểm edu, điểm OCR, extractor...).
        web:         True thì đo theo hồ sơ web (bật kiểm tra boilerplate của pipeline).
        note:        Ghi chú hiển thị trong báo cáo.
    """

    name: str
    kind: str
    location: str | tuple[str, ...]
    text_field: str = "text"
    en_field: str = ""
    extra: tuple[str, ...] = ()
    web: bool = False
    note: str = ""


_HPLT = "https://data.hplt-project.org/three/sorted/vie_Latn/"

SPECS: dict[str, ProbeSpec] = {s.name: s for s in [
    ProbeSpec("finepdfs_edu_vi", "hf_parquet", "datasets/HuggingFaceFW/finepdfs-edu/data/vie_Latn/train",
              extra=("extractor", "token_count", "fw_edu_scores", "duplicate_count", "is_truncated", "url"),
              note="FinePDFs-Edu, tập con lọc giáo dục"),
    ProbeSpec("finepdfs_vi", "hf_parquet", "datasets/HuggingFaceFW/finepdfs/data/vie_Latn/train",
              extra=("extractor", "token_count", "fw_edu_scores", "ocr_quality_scores", "dclm_scores", "duplicate_count",
                     "is_truncated", "url"),
              note="FinePDFs đầy đủ"),
    ProbeSpec("fineweb2_hq_vi", "hf_parquet", "datasets/epfml/FineWeb2-HQ/vie_Latn", extra=("quality_score", "url"), web=True,
              note="Top 10% FineWeb-2 theo classifier kiến thức"),
    ProbeSpec("fineweb2_vi", "hf_parquet", "datasets/HuggingFaceFW/fineweb-2/data/vie_Latn/train", extra=("url",), web=True,
              note="Mốc so sánh: FineWeb-2 ngẫu nhiên (đã chọn ở Q7)"),
    ProbeSpec("hplt3_vi_wds10", "hplt", (_HPLT + "10_1.jsonl.zst",), web=True, note="HPLT 3.0, bin WDS 10 (cao nhất)"),
    ProbeSpec("hplt3_vi_wds9", "hplt", (_HPLT + "9_1.jsonl.zst",), web=True, note="HPLT 3.0, bin WDS 9"),
    ProbeSpec("hplt3_vi_wds5", "hplt", (_HPLT + "5_1.jsonl.zst",), web=True, note="HPLT 3.0, bin WDS 5 (thấp nhất)"),
    ProbeSpec("cosmopedia_vi_math", "hf_parquet", "datasets/thangvip/cosmopedia_vi_math", text_field="vi_text", en_field="text",
              extra=("format", "audience"), note="Cosmopedia dịch máy, phần toán"),
    ProbeSpec("cosmopedia_vi_khan", "hf_parquet", "datasets/thangvip/cosmopedia_vi_khanacademy", text_field="vi_text",
              en_field="text", extra=("format", "audience"), note="Cosmopedia dịch máy, kiểu Khan Academy"),
    ProbeSpec("vi_pubmed", "hf_parquet", "datasets/VietAI/vi_pubmed", text_field="vi", en_field="en",
              note="Tóm tắt PubMed dịch máy"),
    ProbeSpec("metamath_vi", "hf_rows", "5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated",
              text_field="query_vi+response_vi", en_field="query_en+response_en", note="SFT toán, Google Translate"),
    ProbeSpec("orca_math_vi", "hf_rows", "5CD-AI/Vietnamese-microsoft-orca-math-word-problems-200k-gg-translated",
              text_field="question_vi+answer_vi", en_field="question_en+answer_en", note="SFT toán, Google Translate"),
]}
