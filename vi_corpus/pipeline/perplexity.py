"""
Perplexity của đoạn văn theo một mô hình ngôn ngữ nhỏ (mặc định Qwen3-0.6B), tín hiệu thứ hai cho stage quality (J-25 D3).

Chữ hỏng (lỗi font, OCR cũ, mất dấu nhẹ) làm perplexity tăng mạnh so với tiếng Việt đúng. Tín hiệu này KHÔNG phải cổng duy nhất: bảng số, công thức, danh sách loài, tên riêng
cũng có perplexity cao, nên quality chỉ dùng nó kèm các tín hiệu khác (xem vi_corpus.pipeline.quality.annotate_perplexity). Đo trên MAX_TOKENS token đầu của đoạn; số tuyệt đối chỉ có nghĩa
khi so cùng mô hình, cùng độ dài. Cần transformers + torch (group curator); nên chạy trên GPU (mặc định device="cuda", không lặng lẽ rơi xuống CPU như embed "auto" từng làm, J-24).
"""

import logging
import math

logger = logging.getLogger("vi_corpus")

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 512
BATCH_SIZE = 16


class LmScorer:
    """Chấm perplexity từng văn bản bằng mô hình ngôn ngữ nhân quả của HuggingFace; nạp mô hình ở lần gọi score đầu tiên."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cuda", max_tokens: int = MAX_TOKENS, batch_size: int = BATCH_SIZE) -> None:
        """
        Args:
            model_id:   Mô hình trên HuggingFace.
            device:     "cuda" | "cpu" | "auto" (auto = cuda nếu có, ghi cảnh báo khi rơi xuống cpu).
            max_tokens: Số token đầu của văn bản được chấm.
            batch_size: Số văn bản mỗi lô (đã sắp theo độ dài để giảm padding).
        """
        self.model_id, self.device, self.max_tokens, self.batch_size = model_id, device, max_tokens, batch_size
        self._model = None
        self._tok = None

    def _load(self) -> None:
        """Nạp tokenizer và mô hình (bf16 trên GPU); device="cuda" mà không có GPU thì báo lỗi."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                logger.warning("[perplexity] không có GPU, chạy trên CPU (chậm)")
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("perplexity: device=cuda nhưng không thấy GPU (dùng device=cpu hoặc auto nếu chủ ý)")
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._tok.padding_side = "right"
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16 if device == "cuda" else None).to(device).eval()
        self.device = device

    def score(self, texts: list[str]) -> list[float | None]:
        """
        Perplexity của MAX_TOKENS token đầu của từng văn bản; None cho văn bản ngắn hơn 8 token (không đủ để kết luận).

        Returns:
            Danh sách cùng độ dài và thứ tự `texts`.
        """
        import torch

        if self._model is None:
            self._load()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: list[float | None] = [None] * len(texts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self._tok([texts[i] or "" for i in idx], truncation=True, max_length=self.max_tokens, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits[:, :-1].float()
            targets = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].bool()
            nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none").view(targets.shape)
            for row, i in enumerate(idx):
                n = int(mask[row].sum())
                if n >= 8:
                    out[i] = math.exp(float(nll[row][mask[row]].mean()))
        return out
