"""
Stage reduce: gom cụm (HDBSCAN) trên embedding GỐC và giảm chiều xuống 2D (PCA, UMAP) chỉ để vẽ bản đồ.

Cùng cách làm với packages/reducer của ViLA: gom cụm trên ma trận gốc, độc lập với mọi phép chiếu (R-28; UMAP làm méo
mật độ nên HDBSCAN trên toạ độ 2D có thể tạo hoặc xé cụm giả). PCA 2D và UMAP 2D cùng tô theo một nhãn cụm. Khi gom trên
vector nhiều chiều quá chậm thì chọn không gian trung gian bằng cluster_space ("pca50" hoặc "umap10"); 2D không bao giờ
được dùng để gom cụm. Tham số HDBSCAN cấu hình được (R-29): min_cluster_size mặc định theo ViLA
max(2, min(20, n // 10)), min_samples chỉ truyền khi đặt rõ; tham số, số chiều gom cụm, số cụm và tỉ lệ nhiễu được ghi
vào manifest.

Ưu tiên cuML trên GPU khi có (RAPIDS), nếu không thì scikit-learn / umap-learn trên CPU. Fit MỘT lần trên toàn bộ ma
trận (không fit theo phân vùng) để toạ độ và cluster_id so sánh được giữa mọi bản ghi. UMAP thiếu thư viện thì bỏ qua
(chỉ còn PCA). Kết quả ở <run_dir>/07_reduced.parquet: doc_id, pca_x, pca_y, umap_x, umap_y, cluster_id (-1 = nhiễu).
Giới hạn: HDBSCAN của scikit-learn trên vector nhiều chiều không scale tới hàng triệu đoạn; khi chạy lớn phải fit trên
mẫu phân tầng rồi gán nhãn phần còn lại, hoặc dùng cuML HDBSCAN.
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


def reduce_umap(matrix: np.ndarray, prefer_gpu: bool, n_components: int = 2) -> np.ndarray | None:
    """UMAP xuống n_components chiều (cuML nếu prefer_gpu và có, ngược lại umap-learn). None nếu không có thư viện UMAP."""
    n_neighbors = max(2, min(15, len(matrix) - 1))
    if prefer_gpu and have_cuml():
        import cupy as cp
        from cuml.manifold import UMAP
        return UMAP(n_components=n_components, n_neighbors=n_neighbors).fit_transform(cp.asarray(matrix)).get()
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba"))  # tránh lỗi cache numba chỉ-đọc
    try:
        import umap
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không import được umap-learn (%s); bỏ qua UMAP", exc)
        return None
    return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=0).fit_transform(matrix)


CLUSTER_SPACES = ("raw", "pca50", "umap10")


def default_min_cluster_size(n: int) -> int:
    """min_cluster_size mặc định theo ViLA: max(2, min(20, n // 10))."""
    return max(2, min(20, n // 10))


def cluster_matrix(matrix: np.ndarray, space: str, prefer_gpu: bool) -> np.ndarray:
    """
    Ma trận dùng để gom cụm theo cluster_space: "raw" = embedding gốc, "pca50" = PCA 50 chiều, "umap10" = UMAP 10 chiều.

    Raises:
        ValueError: space không thuộc CLUSTER_SPACES.
        RuntimeError: "umap10" mà không có thư viện UMAP.
    """
    if space == "raw":
        return matrix
    if space == "pca50":
        k = min(50, matrix.shape[1], len(matrix) - 1)
        from sklearn.decomposition import PCA
        return PCA(n_components=k, random_state=0).fit_transform(matrix)
    if space == "umap10":
        out = reduce_umap(matrix, prefer_gpu, n_components=min(10, len(matrix) - 2))
        if out is None:
            raise RuntimeError("cluster_space=umap10 cần umap-learn (uv sync --group viz) hoặc cuML")
        return out
    raise ValueError(f"cluster_space không hợp lệ: {space!r}; chọn một trong {CLUSTER_SPACES}")


def cluster(matrix: np.ndarray, min_cluster_size: int, min_samples: int | None = None,
            prefer_gpu: bool = False) -> np.ndarray:
    """
    HDBSCAN trên ma trận (cuML nếu prefer_gpu và có, ngược lại scikit-learn); trả nhãn cụm, -1 là nhiễu.

    Ít điểm hơn 2 * min_cluster_size thì mọi điểm -1. min_samples None = để thư viện tự đặt (bằng min_cluster_size).
    """
    if len(matrix) < 2 * min_cluster_size:
        return np.full(len(matrix), -1)
    kwargs = {"min_cluster_size": min_cluster_size}
    if min_samples is not None:
        kwargs["min_samples"] = min_samples
    if prefer_gpu and have_cuml():
        import cupy as cp
        from cuml.cluster import HDBSCAN as CuHDBSCAN
        return CuHDBSCAN(**kwargs).fit_predict(cp.asarray(matrix)).get()
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(copy=True, **kwargs).fit_predict(matrix)


def reduce_embeddings(ids: list[str], matrix: np.ndarray, prefer_gpu: bool = True, cluster_space: str = "raw",
                      min_cluster_size: int | None = None, min_samples: int | None = None) -> tuple[list[dict], dict]:
    """
    Gom cụm trên không gian cluster_space + giảm chiều 2D để vẽ.

    Returns:
        (các dòng {doc_id, pca_x, pca_y, umap_x, umap_y, cluster_id}, thông tin gom cụm cho manifest:
        {"space", "dim", "min_cluster_size", "min_samples", "clusters", "noise_rate", "umap"}).
        Dưới 3 điểm thì không giảm chiều được: toạ độ 0, mọi điểm là nhiễu.
    """
    n = len(ids)
    mcs = min_cluster_size if min_cluster_size is not None else default_min_cluster_size(n)
    info = {"space": cluster_space, "dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0, "min_cluster_size": mcs,
            "min_samples": min_samples}
    if n < 3:
        return ([{"doc_id": i, "pca_x": 0.0, "pca_y": 0.0, "umap_x": None, "umap_y": None, "cluster_id": -1} for i in ids],
                {**info, "clusters": 0, "noise_rate": 1.0, "umap": False})
    space = cluster_matrix(matrix, cluster_space, prefer_gpu)
    labels = cluster(space, mcs, min_samples, prefer_gpu)
    pca = reduce_pca(matrix, prefer_gpu)
    umap_xy = reduce_umap(matrix, prefer_gpu)
    info.update(dim=int(space.shape[1]), clusters=len(set(labels.tolist()) - {-1}),
                noise_rate=round(float(np.mean(labels == -1)), 4), umap=umap_xy is not None)
    return [{"doc_id": ids[i], "pca_x": float(pca[i, 0]), "pca_y": float(pca[i, 1]),
             "umap_x": float(umap_xy[i, 0]) if umap_xy is not None else None,
             "umap_y": float(umap_xy[i, 1]) if umap_xy is not None else None,
             "cluster_id": int(labels[i])} for i in range(n)], info


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
