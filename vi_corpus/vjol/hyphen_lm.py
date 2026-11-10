"""
Chọn cách đọc cho chỗ "ph-ơng / l-ơng / ch-ơng" mơ hồ của VJOL bằng mô hình ngôn ngữ nhỏ (J-57 (2)).

Font đời cũ làm mất chữ "ư" cùng dấu thanh của "ơ" đứng sau, nên "ph-ơng" có thể là phương / phường / phưởng / phướng / phượng. `hyphen_fix` dựng sẵn các cách đọc có trong tập
âm tiết; ở đây mỗi cách đọc được đặt vào câu thật (ngữ cảnh trái + phải) và chấm bằng tổng log xác suất của cả chuỗi. Chỉ trả về cách đọc tốt nhất khi nó hơn cách nhì ít nhất `margin` nat
(mặc định 4,0, tức khoảng 55 lần); chưa đủ chắc thì trả None và chữ hỏng được giữ nguyên (đoạn sẽ bị `lost_letters` loại như trước).

Dùng cùng loại mô hình với perplexity của stage quality (Qwen/Qwen3-0.6B). Mô hình nạp ở lần gọi đầu tiên; kết quả nhớ theo (trái, cách đọc, phải) nên cùng một cụm chỉ chấm một lần.
"""

import logging
import unicodedata

logger = logging.getLogger("vi_corpus")

DEFAULT_MARGIN = 4.0  # chênh lệch log xác suất tối thiểu giữa cách đọc tốt nhất và nhì
PREFIX = "Bài báo khoa học tiếng Việt.\n"  # mồi cố định để token đầu của ngữ cảnh cũng được chấm (Qwen không có BOS)
MAX_CACHE = 200_000


class LmChooser:
    """Bộ chọn cách đọc bằng mô hình ngôn ngữ nhân quả; gọi như hàm: chooser(left, candidates, right) -> cách đọc hoặc None."""

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", device: str = "cuda", margin: float = DEFAULT_MARGIN) -> None:
        """
        Args:
            model_id: Mô hình trên HuggingFace.
            device:   "cuda" | "cpu" | "auto" (auto = cuda nếu có, ghi cảnh báo khi rơi xuống cpu; cuda mà không có GPU thì báo lỗi).
            margin:   Chênh lệch log xác suất tối thiểu để chấp nhận cách đọc tốt nhất.
        """
        self.model_id, self.device, self.margin = model_id, device, margin
        self._model = None
        self._tok = None
        self._cache: dict[tuple, str | None] = {}
        self.calls = self.chosen = 0

    def _load(self) -> None:
        """Nạp tokenizer và mô hình (bf16 trên GPU)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                logger.warning("[hyphen_lm] không có GPU, chạy trên CPU (chậm)")
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("hyphen_lm: device=cuda nhưng không thấy GPU (dùng device=cpu hoặc auto nếu chủ ý)")
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._tok.padding_side = "right"
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16 if device == "cuda" else None).to(device).eval()
        self.device = device

    def score(self, texts: list[str]) -> list[float]:
        """Tổng log xác suất của từng chuỗi (không tính token đầu tiên của mồi, vì không có gì đứng trước); cùng mồi nên so sánh được giữa các cách đọc."""
        import torch

        if self._model is None:
            self._load()
        enc = self._tok([PREFIX + t for t in texts], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self._model(**enc).logits[:, :-1].float()
        targets = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none").view(targets.shape)
        return [-float(nll[row][mask[row]].sum()) for row in range(len(texts))]

    def __call__(self, left: str, candidates: list[str], right: str) -> str | None:
        """
        Cách đọc tốt nhất trong ngữ cảnh, hoặc None khi chưa hơn cách nhì `margin`.

        Args:
            left:       Văn bản đứng trước chỗ mơ hồ (cắt ở ranh giới từ để không bắt đầu giữa chữ).
            candidates: Các cách đọc (ít nhất 2).
            right:      Văn bản đứng sau chỗ mơ hồ.
        """
        if len(candidates) < 2:
            return None
        left = left.split(" ", 1)[1] if " " in left[:20] and len(left) > 20 else left
        key = (unicodedata.normalize("NFC", left), tuple(candidates), unicodedata.normalize("NFC", right))
        if key in self._cache:
            return self._cache[key]
        self.calls += 1
        scores = self.score([key[0] + c + key[2] for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        picked = candidates[order[0]] if scores[order[0]] - scores[order[1]] >= self.margin else None
        self.chosen += picked is not None
        if len(self._cache) >= MAX_CACHE:
            self._cache.clear()
        self._cache[key] = picked
        return picked
