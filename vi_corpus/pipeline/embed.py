"""
Stage 9 (embed): vector hóa văn bản để vẽ bản đồ embedding (kiểm tra bằng mắt các cụm, nguồn, bản bị loại).

Hai bộ nhúng, chọn bằng chuỗi `spec`:
- "hf:<model_id>" (mặc định intfloat/multilingual-e5-base): mô hình HuggingFace chạy trên GPU, trung bình hóa token +
  chuẩn hóa L2 (cùng cách với packages/embedder của ViLA). Hàm thuần theo từng lô nên chạy song song nhiều GPU được
  (mỗi actor Curator giữ một bản mô hình trên một GPU, xem vi_corpus.pipeline.curator.EmbedStage).
- "tfidf": TF-IDF + SVD 64 chiều bằng scikit-learn, chạy CPU, không cần tải mô hình. Phải fit trên TOÀN BỘ corpus nên
  không chạy song song được; dùng khi thử trên máy không có GPU / không có mạng.
Embedding lưu riêng ở <run_dir>/06_embeddings.parquet (doc_id, embedding): schema bản ghi chính không đổi.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SPEC = "hf:intfloat/multilingual-e5-base"
DEFAULT_PROMPT = "passage: "  # họ E5 cần tiền tố này cho văn bản (document); mô hình khác đặt "" bằng --embed-prompt
MAX_CHARS = 6000  # cắt văn bản trước khi token hóa, đủ cho 512 token mà không tốn thời gian tokenizer
EMB_SCHEMA = pa.schema([("doc_id", pa.string()), ("embedding", pa.list_(pa.float32()))])


class HfEmbedder:
    """Bộ nhúng HuggingFace: nạp mô hình một lần lên một thiết bị, nhúng từng lô văn bản."""

    def __init__(self, model_id: str, device: str = "auto", max_seq_length: int = 512, batch_size: int = 64,
                 prompt: str = DEFAULT_PROMPT) -> None:
        """
        Nạp tokenizer và mô hình (fp16 trên GPU, fp32 trên CPU).

        Args:
            model_id:       Tên / đường dẫn mô hình HF.
            device:         "auto" (cuda nếu có, ngược lại cpu), "cuda" hoặc "cpu". Trong actor Ray có 1 GPU được cấp,
                            "cuda" luôn là GPU đó (CUDA_VISIBLE_DEVICES do Ray đặt).
            max_seq_length: Số token tối đa mỗi văn bản (phần dư bị cắt).
            batch_size:     Số văn bản mỗi lần chạy mô hình.
            prompt:         Tiền tố gắn trước mỗi văn bản.
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device).eval()
        self.max_seq_length, self.batch_size, self.prompt = max_seq_length, batch_size, prompt

    def embed(self, texts: list[str]) -> np.ndarray:
        """Nhúng danh sách văn bản thành ma trận (n, dim) float32 đã chuẩn hóa L2."""
        import torch

        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = [self.prompt + (t or "")[:MAX_CHARS] for t in texts[i:i + self.batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_seq_length,
                                 return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out.append(torch.nn.functional.normalize(vec.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


def embed_tfidf(texts: list[str], dim: int = 64) -> np.ndarray:
    """TF-IDF (20.000 từ, sublinear) + SVD `dim` chiều, chuẩn hóa L2. Cần toàn bộ corpus cùng lúc."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    x = TfidfVectorizer(max_features=20_000, sublinear_tf=True, min_df=2 if len(texts) > 50 else 1).fit_transform(
        [(t or "")[:MAX_CHARS * 4] for t in texts])
    k = max(2, min(dim, x.shape[0] - 1, x.shape[1] - 1))
    return normalize(TruncatedSVD(n_components=k, random_state=0).fit_transform(x)).astype(np.float32)


def embed_rows(rows: list[dict], spec: str = DEFAULT_SPEC, device: str = "auto", prompt: str = DEFAULT_PROMPT,
               batch_size: int = 64, max_seq_length: int = 512) -> list[dict]:
    """
    Nhúng văn bản của các bản ghi trong MỘT tiến trình (một thiết bị).

    Returns:
        Danh sách {"doc_id", "embedding": list[float]} cùng thứ tự `rows`.

    Raises:
        ValueError: nếu spec không phải "tfidf" hoặc "hf:<model_id>".
    """
    texts = [r["text"] for r in rows]
    if spec == "tfidf":
        mat = embed_tfidf(texts)
    elif spec.startswith("hf:"):
        mat = HfEmbedder(spec[3:], device, max_seq_length, batch_size, prompt).embed(texts)
    else:
        raise ValueError(f"spec embedding không hợp lệ: {spec!r} (dùng 'tfidf' hoặc 'hf:<model_id>')")
    return [{"doc_id": r["doc_id"], "embedding": v.tolist()} for r, v in zip(rows, mat)]


def write_embeddings(path: Path, items: list[dict]) -> int:
    """Ghi (doc_id, embedding) ra Parquet nguyên tử. Trả về số dòng."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(items, schema=EMB_SCHEMA), tmp)
    tmp.replace(path)
    return len(items)


def read_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    """Đọc file embedding thành (danh sách doc_id, ma trận (n, dim) float32)."""
    table = pq.read_table(path)
    ids = table.column("doc_id").to_pylist()
    return ids, np.array(table.column("embedding").to_pylist(), dtype=np.float32).reshape(len(ids), -1)
