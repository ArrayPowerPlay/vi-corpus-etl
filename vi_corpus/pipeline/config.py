"""
Cấu hình của pipeline: cơ cấu mẫu giữa các nguồn, hồ sơ theo nguồn (ngưỡng chất lượng, ngôn ngữ cho phép)
và các tham số làm sạch dòng, cắt đoạn, nhận diện ngôn ngữ, chấm chất lượng, dedup, gom cụm.

Mọi ngưỡng nằm ở đây (không rải trong code các stage) để chỉnh một chỗ và ghi lại được vào manifest của lần chạy.
Các ngưỡng là điểm khởi đầu, chưa hiệu chỉnh trên dữ liệu thật (R-11: hiệu chỉnh bằng điểm LLM trên ~50.000 mẫu).
"""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SourceProfile:
    """
    Hồ sơ xử lý riêng của một nguồn.

    Attributes:
        chunked:          Nguồn là sách / tài liệu dài: luôn cắt thành đoạn ở stage prepare và lấy mẫu theo số đoạn.
        split_long:       Nguồn không chunked: văn bản dài hơn chunk_max_tokens vẫn được cắt theo cùng quy tắc (D-10).
                          Tắt với hội thoại (SEA-Instruct) vì cắt sẽ làm vỡ cặp hỏi - đáp.
        min_words:        Số từ tối thiểu; ít hơn thì loại (too_short).
        max_words:        Số từ tối đa; nhiều hơn thì loại (too_long).
        min_alpha:        Tỷ lệ chữ cái tối thiểu trong ký tự không phải khoảng trắng.
        web_checks:       Bật các kiểm tra riêng của văn bản web (boilerplate, URL, dòng gạch đầu / dấu chấm lửng).
        in_doc_line_clean: Chạy xoá dòng lặp trong văn bản (D-04 mục A). Tắt với hội thoại SFT (F-01): dòng "$$", "\\[",
                          nhãn markdown lặp là nội dung; vòng lặp do mô hình sinh đã bị repeated_ngrams / duplicate_lines
                          loại cả mẫu ở stage quality.
        cross_line_clean: Xoá dòng lặp ở nhiều văn bản của cùng nguồn (D-04 mục C, chỉ dành cho nguồn web).
        allowed_langs:    Ngôn ngữ chấp nhận; ngoài danh sách thì loại (lang_not_allowed).
        min_lang_score:   Ngưỡng lang_score (độ thuần x độ tin cậy, D-02); thấp hơn thì gắn mã low_lang_score (trừ điểm nhẹ).
        dedup_group:      Nhóm dedup: "cpt" dedup toàn cục giữa mọi nguồn, "sft" (hỏi - đáp) chỉ dedup trong nhóm của mình (Q4).
    """

    chunked: bool = False
    split_long: bool = True
    min_words: int = 50
    max_words: int = 100_000
    min_alpha: float = 0.55
    web_checks: bool = False
    in_doc_line_clean: bool = True
    cross_line_clean: bool = False
    allowed_langs: tuple[str, ...] = ("vi",)
    min_lang_score: float = 0.65
    dedup_group: str = "cpt"


PROFILES: dict[str, SourceProfile] = {
    "sea_pile_v2": SourceProfile(web_checks=True, cross_line_clean=True),
    "sea_lion_pile_v1": SourceProfile(web_checks=True, cross_line_clean=True),
    "sea_instruct_2602": SourceProfile(split_long=False, min_words=5, min_alpha=0.4, in_doc_line_clean=False,
                                       dedup_group="sft"),
    "stbook": SourceProfile(chunked=True, min_words=80),
    # giữ sách tiếng Anh (P-08); ngưỡng lang_score thấp hơn vì giáo trình hay trộn Việt - Anh
    "giao_trinh": SourceProfile(chunked=True, min_words=80, allowed_langs=("vi", "en"), min_lang_score=0.5),
}

# Cơ cấu mặc định khi không chỉ định --mix (tỷ lệ, sẽ được quy về --total).
DEFAULT_MIX: dict[str, int] = {"sea_pile_v2": 30, "sea_lion_pile_v1": 30, "sea_instruct_2602": 15, "stbook": 25}

# Các giá trị rights_status được coi là đã rõ quyền; còn lại ("unknown", ...) bị gắn rights_gate="quarantine".
# Theo R-01 cổng quyền hiện chỉ gắn nhãn (rights_gate="tag"), không loại, không chặn dedup / knowledge unit.
OPEN_RIGHTS = frozenset({"open", "cleared"})

# Thứ tự ưu tiên giữ bản khi trùng giữa các nguồn (R-15: VJOL -> SEA -> stbook -> các dataset khác). Số nhỏ = ưu tiên
# cao. Cùng mức thì xét điểm chất lượng rule (R-19). Thứ tự trong nhóm SEA (v2 / v1) và trong nhóm "các dataset khác"
# chưa chốt (R-15 a, b) nên để cùng mức. Nguồn không có ở đây dùng DEFAULT_PRIORITY.
SOURCE_PRIORITY: dict[str, int] = {"vjol": 0, "sea_pile_v2": 1, "sea_lion_pile_v1": 1, "sea_instruct_2602": 1,
                                   "stbook": 2}
DEFAULT_PRIORITY = 3

DEFAULT_TOKENIZER = "hf:Qwen/Qwen3-0.6B"  # tokenizer chính thức để đếm KPI và cắt đoạn (D-01, R-35)
DEFAULT_LANG_MODEL = "fasttext:lid.176"  # fastText lid.176.bin (D-02, R-20)


def scale_mix(weights: dict[str, int], total: int) -> dict[str, int]:
    """
    Quy các trọng số về số mẫu nguyên có tổng đúng bằng `total` (phương pháp phần dư lớn nhất).

    Raises:
        ValueError: nếu tổng trọng số bằng 0.
    """
    s = sum(weights.values())
    if s <= 0:
        raise ValueError("Tổng trọng số cơ cấu mẫu phải > 0")
    raw = {k: total * v / s for k, v in weights.items()}
    out = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


@dataclass(frozen=True)
class RunConfig:
    """
    Tham số của một lần chạy pipeline.

    Attributes:
        mix:                  Số mẫu cần lấy từ mỗi nguồn (nguồn chunked: số đoạn sau khi cắt).
        seed:                 Hạt giống ngẫu nhiên; cùng seed + cùng dữ liệu thì cùng mẫu.
        rows_per_file:        Số dòng lấy từ mỗi file khi lấy mẫu nguồn SEA (để mẫu trải đều trên nhiều file).
        tokenizer:            Bộ đếm token (vi_corpus.pipeline.tokens): "hf:<repo>", "tiktoken:<enc>" hoặc "words".
        compare_tokenizers:   Các tokenizer khác chỉ để so tổng token của phần giữ lại trong báo cáo (không đổi cách cắt).
        chunk_target_tokens:  Số token mục tiêu của một đoạn (D-10).
        chunk_max_tokens:     Số token tối đa của một đoạn.
        chunk_min_tokens:     Đoạn cuối ngắn hơn mức này thì gộp vào đoạn trước (nếu không vượt chunk_max_tokens).
        line_short_words:     Dòng ít hơn số từ này là "dòng ngắn" (D-04 mục A).
        line_repeat_min:      Dòng ngắn lặp từ số lần này trở lên trong một văn bản thì chỉ giữ lần đầu.
        cross_line_min_docs:  D-04 mục C: dòng xuất hiện ở ít nhất max(số này, cross_line_min_frac x số văn bản) văn bản
                              của cùng nguồn web thì bị xoá.
        cross_line_min_frac:  Xem cross_line_min_docs.
        lang_model:           Bộ nhận diện ngôn ngữ: "fasttext:lid.176" (tự tải), "fasttext:<đường dẫn .bin>" hoặc
                              "heuristic" (đếm chữ có dấu + stopword, không cần mô hình).
        lang_max_segments:    Số đoạn văn tối đa đưa vào bộ nhận diện mỗi văn bản (rải đều), để giữ chi phí cố định.
        keep_bands:           Các band chất lượng được giữ lại trong clean corpus.
        rights_gate:          "tag" chỉ gắn nhãn quarantine (mặc định, R-01); "enforce" thì loại luôn (dùng lúc phát hành).
        fuzzy_threshold:      Ngưỡng Jaccard (ước lượng từ MinHash) để coi hai văn bản là gần trùng (R-34: giữ cấu hình A).
        source_priority:      Mức ưu tiên giữ bản khi trùng giữa nguồn (xem SOURCE_PRIORITY).
        contained_frac:       L1 (D-03, G-05 bước 3): văn bản có >= tỉ lệ này số đoạn văn dài nằm trong một văn bản lớn
                              hơn thì bị coi là trùng bao hàm (dup_kind "contained").
        contained_min_paras:  Chỉ xét bao hàm cho văn bản có ít nhất chừng này đoạn văn dài.
        para_min_words:       Đoạn văn (ngăn bởi dòng trống) có ít nhất chừng này từ mới được băm cho L1.
        dedup_index:          Ghi kho dấu vân tay dedup vòng 1 (D-09) vào <data_root>/state/dedup_index/<nguồn>/.
        embed_spec:           Bộ nhúng cho bản đồ embedding: "hf:<model_id>", "tfidf" (CPU, không cần mô hình) hoặc None để
                              bỏ qua embed / reduce / bản đồ.
        embed_prompt:         Tiền tố gắn trước văn bản khi nhúng (E5 cần "passage: ").
        embed_batch_size:     Số văn bản mỗi lần chạy mô hình nhúng.
        embed_max_seq:        Số token tối đa khi nhúng.
        embed_scope:          "all" nhúng mọi bản ghi (bản đồ thấy cả bản bị loại, dùng cho chạy mẫu); "kept" chỉ nhúng phần
                              còn sống sau lọc rẻ (R-05, dùng khi chạy lớn).
        prefer_gpu:           Giảm chiều bằng cuML (GPU) nếu cài; không thì scikit-learn / umap-learn trên CPU.
        cluster_space:        Không gian gom cụm HDBSCAN (R-28): "raw" = embedding gốc, "pca50" / "umap10" = giảm chiều
                              trung gian khi raw quá chậm. Không bao giờ gom trên toạ độ 2D.
        hdbscan_min_cluster_size: None = mặc định ViLA max(2, min(20, n // 10)) (R-29).
        hdbscan_min_samples:  None = để HDBSCAN tự đặt; chỉ truyền khi đặt rõ (R-29).
        cpt_context_tokens:   Độ dài tối đa một khối CPT khi nối các đoạn liền nhau còn giữ (D-10). Chưa chốt theo mô hình đích.
        benchmarks_dir:       Thư mục tập benchmark cho quét rò rỉ 13-gram; rỗng / không có = danh sách rỗng (R-16).
        global_verdict:       Đường dẫn verdict.parquet của dedup vòng 2 (D-09); có thì áp vào trước khi sinh knowledge unit.
        profiles:             Hồ sơ theo nguồn.
    """

    mix: dict[str, int]
    seed: int = 42
    rows_per_file: int = 250
    tokenizer: str = DEFAULT_TOKENIZER
    compare_tokenizers: tuple[str, ...] = ()
    chunk_target_tokens: int = 1024
    chunk_max_tokens: int = 2048
    chunk_min_tokens: int = 256
    line_short_words: int = 10
    line_repeat_min: int = 3
    cross_line_min_docs: int = 20
    cross_line_min_frac: float = 0.0001
    lang_model: str = DEFAULT_LANG_MODEL
    lang_max_segments: int = 50
    keep_bands: tuple[str, ...] = ("A", "B", "C")
    rights_gate: str = "tag"
    fuzzy_threshold: float = 0.8
    source_priority: dict[str, int] = field(default_factory=lambda: dict(SOURCE_PRIORITY))
    contained_frac: float = 0.8
    contained_min_paras: int = 3
    para_min_words: int = 20
    dedup_index: bool = True
    embed_spec: str | None = "hf:intfloat/multilingual-e5-base"
    embed_prompt: str = "passage: "
    embed_batch_size: int = 64
    embed_max_seq: int = 512
    embed_scope: str = "all"
    prefer_gpu: bool = True
    cluster_space: str = "raw"
    hdbscan_min_cluster_size: int | None = None
    hdbscan_min_samples: int | None = None
    cpt_context_tokens: int = 4096
    benchmarks_dir: str | None = "configs/benchmarks"
    global_verdict: str | None = None
    profiles: dict[str, SourceProfile] = field(default_factory=lambda: dict(PROFILES))

    def profile(self, source_key: str) -> SourceProfile:
        """Hồ sơ của một nguồn; nguồn lạ dùng hồ sơ mặc định."""
        return self.profiles.get(source_key, SourceProfile())

    def priority(self, source_key: str) -> int:
        """Mức ưu tiên giữ bản khi trùng của một nguồn (số nhỏ = ưu tiên cao)."""
        return self.source_priority.get(source_key, DEFAULT_PRIORITY)

    def to_dict(self) -> dict:
        """Dạng dict (JSON được) để ghi vào manifest."""
        return asdict(self)
