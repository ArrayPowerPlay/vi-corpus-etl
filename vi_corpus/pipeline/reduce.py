"""
Stage 10 (reduce): giảm chiều embedding xuống 2D (PCA, UMAP) và gom cụm (HDBSCAN) để vẽ bản đồ.

Cùng cách làm với packages/reducer của ViLA: ưu tiên cuML trên GPU khi có (RAPIDS), nếu không thì scikit-learn /
umap-learn trên CPU. Fit MỘT lần trên toàn bộ ma trận (không fit theo phân vùng) để toạ độ và cluster_id so sánh được
giữa mọi bản ghi. UMAP thiếu thư viện thì bỏ qua (chỉ còn PCA), không làm hỏng pipeline. Kết quả ở
<run_dir>/07_reduced.parquet: doc_id, pca_x, pca_y, umap_x, umap_y, cluster_id (-1 = nhiễu).
"""

import logging
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("vi_corpus")


def have_cuml() -> bool:
    """True nếu import được cuml (có RAPIDS trên máy)."""
    try:
        import cuml  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 — thiếu CUDA / thư viện đều coi là không có
        return False


def reduce_pca(matrix: np.ndarray, prefer_gpu: bool) -> np.ndarray:
    """PCA xuống 2 chiều (cuML nếu prefer_gpu và có, ngược lại scikit-learn)."""
    if prefer_gpu and have_cuml():
        import cupy as cp
        from cuml.decomposition import PCA
        return PCA(n_components=2).fit_transform(cp.asarray(matrix)).get()
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=0).fit_transform(matrix)


def reduce_umap(matrix: np.ndarray, prefer_gpu: bool) -> np.ndarray | None:
    """UMAP xuống 2 chiều (cuML nếu prefer_gpu và có, ngược lại umap-learn). Trả None nếu không có thư viện UMAP."""
    n_neighbors = max(2, min(15, len(matrix) - 1))
    if prefer_gpu and have_cuml():
        import cupy as cp
        from cuml.manifold import UMAP
        return UMAP(n_components=2, n_neighbors=n_neighbors).fit_transform(cp.asarray(matrix)).get()
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba"))  # tránh lỗi cache numba chỉ-đọc
    try:
        import umap
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không import được umap-learn (%s); bỏ qua UMAP", exc)
        return None
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=0).fit_transform(matrix)


def cluster(coords: np.ndarray, min_cluster_size: int = 25) -> np.ndarray:
    """HDBSCAN (scikit-learn) trên toạ độ 2D; trả nhãn cụm, -1 là nhiễu. Ít điểm hơn 2 * min_cluster_size thì mọi điểm -1."""
    if len(coords) < 2 * min_cluster_size:
        return np.full(len(coords), -1)
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(coords)


def reduce_embeddings(ids: list[str], matrix: np.ndarray, prefer_gpu: bool = True) -> list[dict]:
    """
    Giảm chiều + gom cụm, trả về các dòng {doc_id, pca_x, pca_y, umap_x, umap_y, cluster_id}.

    Cụm tính trên UMAP nếu có, ngược lại trên PCA. Dưới 3 điểm thì không giảm chiều được: trả về toạ độ 0.
    """
    n = len(ids)
    if n < 3:
        return [{"doc_id": i, "pca_x": 0.0, "pca_y": 0.0, "umap_x": None, "umap_y": None, "cluster_id": -1} for i in ids]
    pca = reduce_pca(matrix, prefer_gpu)
    umap_xy = reduce_umap(matrix, prefer_gpu)
    labels = cluster(umap_xy if umap_xy is not None else pca)
    return [{"doc_id": ids[i], "pca_x": float(pca[i, 0]), "pca_y": float(pca[i, 1]),
             "umap_x": float(umap_xy[i, 0]) if umap_xy is not None else None,
             "umap_y": float(umap_xy[i, 1]) if umap_xy is not None else None,
             "cluster_id": int(labels[i])} for i in range(n)]


REDUCED_SCHEMA = pa.schema([("doc_id", pa.string()), ("pca_x", pa.float64()), ("pca_y", pa.float64()),
                            ("umap_x", pa.float64()), ("umap_y", pa.float64()), ("cluster_id", pa.int64())])


def write_reduced(path: Path, items: list[dict]) -> int:
    """Ghi kết quả giảm chiều ra Parquet nguyên tử. Trả về số dòng."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(items, schema=REDUCED_SCHEMA), tmp)
    tmp.replace(path)
    return len(items)


def read_reduced(path: Path) -> list[dict]:
    """Đọc 07_reduced.parquet thành list[dict]."""
    return pq.read_table(path).to_pylist()
