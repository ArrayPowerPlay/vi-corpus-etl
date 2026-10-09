"""
Đếm token bằng tokenizer thật (D-01, R-35): mặc định tokenizer của Qwen3 (`hf:Qwen/Qwen3-0.6B`).

Chuỗi `spec` chọn bộ đếm:
- "hf:<repo hoặc đường dẫn>": thư viện `tokenizers` (Rust), chỉ nạp tokenizer.json (không cần transformers / torch),
  đếm theo lô bằng encode_batch. Tải từ Hugging Face lần đầu (cache của huggingface_hub).
- "tiktoken:<encoding>" (vd "tiktoken:o200k_base"): cần tự cài tiktoken (không có trong dependency của repo).
- "words": số từ tách bằng khoảng trắng (tiếng Việt: số âm tiết). Không cần mạng, dùng cho test / chạy thử; thấp hơn
  số token thật (đo M-01: Qwen3 ~1,37 token / từ) nên không dùng để đo KPI.

Mọi bộ đếm có cùng giao diện TokenCounter: count, count_batch, hard_split (cắt cứng theo token, dùng khi cắt đoạn).
Bộ đếm được nạp một lần mỗi tiến trình (cache theo spec).
"""

from functools import lru_cache

BATCH = 256  # số văn bản mỗi lần encode_batch


class TokenCounter:
    """Bộ đếm token theo spec (xem docstring module)."""

    def __init__(self, spec: str) -> None:
        """
        Nạp tokenizer theo spec.

        Raises:
            ValueError: spec không thuộc "hf:", "tiktoken:", "words".
            ImportError: thiếu thư viện tiktoken khi dùng "tiktoken:".
        """
        self.spec = spec
        self._tok = self._enc = None
        if spec.startswith("hf:"):
            from tokenizers import Tokenizer

            name = spec[3:]
            self._tok = Tokenizer.from_file(name) if name.endswith(".json") else Tokenizer.from_pretrained(name)
        elif spec.startswith("tiktoken:"):
            import tiktoken

            self._enc = tiktoken.get_encoding(spec[len("tiktoken:"):])
        elif spec != "words":
            raise ValueError(f"tokenizer không hợp lệ: {spec!r} (dùng 'hf:<repo>', 'tiktoken:<enc>' hoặc 'words')")

    def count_batch(self, texts: list[str]) -> list[int]:
        """Số token của từng văn bản (không thêm token đặc biệt)."""
        texts = [t or "" for t in texts]
        if self._tok is not None:
            out: list[int] = []
            for i in range(0, len(texts), BATCH):
                out += [len(e.ids) for e in self._tok.encode_batch(texts[i:i + BATCH], add_special_tokens=False)]
            return out
        if self._enc is not None:
            return [len(x) for x in self._enc.encode_ordinary_batch(texts)]
        return [len(t.split()) for t in texts]

    def count(self, text: str) -> int:
        """Số token của một văn bản."""
        return self.count_batch([text])[0]

    def hard_split(self, text: str, max_tokens: int) -> list[str]:
        """
        Cắt cứng text thành các phần <= max_tokens token (mức cắt cuối cùng của D-10, khi không có ranh giới câu).

        Cắt tại offset ký tự của token nên không làm hỏng ký tự; phần rỗng sau khi strip bị bỏ.
        """
        if self._tok is not None:
            offsets = self._tok.encode(text, add_special_tokens=False).offsets
            starts = [offsets[i][0] for i in range(0, len(offsets), max_tokens)]
            parts = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
        elif self._enc is not None:
            ids = self._enc.encode_ordinary(text)
            parts = [self._enc.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)]
        else:
            words = text.split()
            parts = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
        return [p.strip() for p in parts if p.strip()]


@lru_cache(maxsize=8)
def get_counter(spec: str) -> TokenCounter:
    """Bộ đếm dùng chung trong tiến trình cho một spec (nạp tokenizer một lần)."""
    return TokenCounter(spec)
