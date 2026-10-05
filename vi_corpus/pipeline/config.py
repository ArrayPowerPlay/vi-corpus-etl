"""
Cấu hình của pipeline: cơ cấu mẫu giữa các nguồn, hồ sơ theo nguồn (ngưỡng chất lượng, ngôn ngữ cho phép)
và các tham số cắt đoạn, chấm chất lượng, dedup.

Mọi ngưỡng nằm ở đây (không rải trong code các stage) để chỉnh một chỗ và ghi lại được vào manifest của lần chạy.
"""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SourceProfile:
    """
    Hồ sơ xử lý riêng của một nguồn.

    Attributes:
        chunked:       Nguồn là sách / tài liệu dài, cắt thành đoạn ở stage chunk (mỗi đoạn là một mẫu).
        min_words:     Số từ tối thiểu; ít hơn thì loại (too_short).
        max_words:     Số từ tối đa; nhiều hơn thì loại (too_long).
        min_alpha:     Tỷ lệ chữ cái tối thiểu trong ký tự không phải khoảng trắng.
        web_checks:    Bật các kiểm tra riêng của văn bản web (boilerplate, URL, dòng gạch đầu / dấu chấm lửng).
        allowed_langs: Ngôn ngữ chấp nhận; ngoài danh sách thì loại (lang_not_allowed).
    """

    chunked: bool = False
    min_words: int = 50
    max_words: int = 100_000
    min_alpha: float = 0.55
    web_checks: bool = False
    allowed_langs: tuple[str, ...] = ("vi",)


PROFILES: dict[str, SourceProfile] = {
    "sea_pile_v2": SourceProfile(web_checks=True),
    "sea_lion_pile_v1": SourceProfile(web_checks=True),
    "sea_instruct_2602": SourceProfile(min_words=5, min_alpha=0.4),
    "stbook": SourceProfile(chunked=True, min_words=80),
    "giao_trinh": SourceProfile(chunked=True, min_words=80, allowed_langs=("vi", "en")),  # giữ sách tiếng Anh
}

# Cơ cấu mặc định khi không chỉ định --mix (tỷ lệ, sẽ được quy về --total).
DEFAULT_MIX: dict[str, int] = {"sea_pile_v2": 30, "sea_lion_pile_v1": 30, "sea_instruct_2602": 15, "stbook": 25}

# Các giá trị rights_status được coi là đã rõ quyền; còn lại ("unknown", ...) bị gắn rights_gate="quarantine".
OPEN_RIGHTS = frozenset({"open", "cleared"})


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
        mix:               Số mẫu cần lấy từ mỗi nguồn (nguồn chunked: số đoạn sau khi cắt).
        seed:              Hạt giống ngẫu nhiên; cùng seed + cùng dữ liệu thì cùng mẫu.
        rows_per_file:     Số dòng lấy từ mỗi file khi lấy mẫu nguồn SEA (để mẫu trải đều trên nhiều file).
        chunk_target_words: Số từ mục tiêu của một đoạn (cắt ở ranh giới đoạn văn).
        chunk_max_words:   Số từ tối đa của một đoạn.
        chunk_min_words:   Đoạn cuối ngắn hơn mức này thì gộp vào đoạn trước.
        keep_bands:        Các band chất lượng được giữ lại trong clean corpus.
        rights_gate:       "tag" chỉ gắn nhãn quarantine (mặc định, vì mọi nguồn đang unknown); "enforce" thì loại luôn.
        fuzzy_threshold:   Ngưỡng Jaccard (ước lượng từ MinHash) để coi hai văn bản là gần trùng.
        tokenizer:         Tên/đường dẫn tokenizer HF để đếm token thật; None thì token_count = số từ (ước lượng thấp).
        embed_spec:        Bộ nhúng cho bản đồ embedding: "hf:<model_id>", "tfidf" (CPU, không cần mô hình) hoặc None để bỏ qua
                           embed / reduce / bản đồ.
        embed_prompt:      Tiền tố gắn trước văn bản khi nhúng (E5 cần "passage: ").
        embed_batch_size:  Số văn bản mỗi lần chạy mô hình nhúng.
        embed_max_seq:     Số token tối đa khi nhúng.
        prefer_gpu:        Giảm chiều bằng cuML (GPU) nếu cài; không thì scikit-learn / umap-learn trên CPU.
        profiles:          Hồ sơ theo nguồn.
    """

    mix: dict[str, int]
    seed: int = 42
    rows_per_file: int = 250
    chunk_target_words: int = 600
    chunk_max_words: int = 900
    chunk_min_words: int = 80
    keep_bands: tuple[str, ...] = ("A", "B", "C")
    rights_gate: str = "tag"
    fuzzy_threshold: float = 0.8
    tokenizer: str | None = None
    embed_spec: str | None = "hf:intfloat/multilingual-e5-base"
    embed_prompt: str = "passage: "
    embed_batch_size: int = 64
    embed_max_seq: int = 512
    prefer_gpu: bool = True
    profiles: dict[str, SourceProfile] = field(default_factory=lambda: dict(PROFILES))

    def profile(self, source_key: str) -> SourceProfile:
        """Hồ sơ của một nguồn; nguồn lạ dùng hồ sơ mặc định."""
        return self.profiles.get(source_key, SourceProfile())

    def to_dict(self) -> dict:
        """Dạng dict (JSON được) để ghi vào manifest."""
        return asdict(self)
