"""
Outlier Scoring Module
Implements Euclidean trend distance and advanced outlier detectors (PCA, kNN, CBLOF).
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from trendsurf.trend import Trend


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Computes Euclidean distance between two vectors:
    sqrt(sum((t_i - c_i)^2))
    """
    diff = np.asarray(v1, dtype=float) - np.asarray(v2, dtype=float)
    return float(np.sqrt(np.sum(diff ** 2)))


def rmsd_score(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Root Mean Square Deviation between two trend vectors.
    """
    diff = np.asarray(v1, dtype=float) - np.asarray(v2, dtype=float)
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_euclidean_score(
    target_trend: Union[Trend, np.ndarray],
    reference_trend: Union[Trend, np.ndarray],
    use_normalized: bool = True
) -> float:
    """
    Calculates Euclidean outlier score between target trend and reference trend.
    """
    t_vec = target_trend.normalized_values if isinstance(target_trend, Trend) and use_normalized else (
        target_trend.raw_values if isinstance(target_trend, Trend) else np.asarray(target_trend)
    )
    r_vec = reference_trend.normalized_values if isinstance(reference_trend, Trend) and use_normalized else (
        reference_trend.raw_values if isinstance(reference_trend, Trend) else np.asarray(reference_trend)
    )

    if len(t_vec) != len(r_vec):
        raise ValueError(f"Vector dimensions mismatch: {len(t_vec)} vs {len(r_vec)}")

    return euclidean_distance(t_vec, r_vec)


def compute_pca_outlier_score(
    target_trend: Union[Trend, np.ndarray],
    reference_trend: Union[Trend, np.ndarray],
    sibling_vectors: Optional[List[np.ndarray]] = None,
    n_components: int = 2
) -> float:
    """
    Computes PCA reconstruction error outlier score.
    If sibling_vectors are provided, fits PCA on siblings;
    otherwise computes projection error against reference trend.
    """
    t_vec = target_trend.normalized_values if isinstance(target_trend, Trend) else np.asarray(target_trend)
    r_vec = reference_trend.normalized_values if isinstance(reference_trend, Trend) else np.asarray(reference_trend)

    if sibling_vectors is not None and len(sibling_vectors) >= 5:
        X = np.array(sibling_vectors)
        k = min(n_components, X.shape[0] - 1, X.shape[1])
        if k < 1:
            return euclidean_distance(t_vec, r_vec)
        pca = PCA(n_components=k)
        pca.fit(X)
        t_proj = pca.inverse_transform(pca.transform(t_vec.reshape(1, -1)))
        recon_err = float(np.linalg.norm(t_vec - t_proj[0]))
        return recon_err
    else:
        # Fallback to Euclidean difference
        return euclidean_distance(t_vec, r_vec)


def compute_knn_outlier_score(
    target_trend: Union[Trend, np.ndarray],
    sibling_vectors: List[np.ndarray],
    k: int = 5
) -> float:
    """
    Computes k-NN outlier score as the average distance to the k-nearest sibling trends.
    """
    t_vec = target_trend.normalized_values if isinstance(target_trend, Trend) else np.asarray(target_trend)
    if not sibling_vectors or len(sibling_vectors) < 2:
        return 0.0

    X = np.array(sibling_vectors)
    n_neighbors = min(k, len(sibling_vectors))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X)
    distances, _ = nbrs.kneighbors(t_vec.reshape(1, -1))
    return float(np.mean(distances[0]))


def compute_cblof_score(
    target_trend: Union[Trend, np.ndarray],
    sibling_vectors: List[np.ndarray],
    n_clusters: int = 3
) -> float:
    """
    Computes cluster-based outlier score (distance to closest cluster center).
    """
    t_vec = target_trend.normalized_values if isinstance(target_trend, Trend) else np.asarray(target_trend)
    if not sibling_vectors or len(sibling_vectors) < n_clusters:
        return 0.0

    X = np.array(sibling_vectors)
    k = min(n_clusters, len(sibling_vectors) // 2)
    if k < 1:
        k = 1
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    centers = kmeans.cluster_centers_
    dists = [np.linalg.norm(t_vec - c) for c in centers]
    return float(np.min(dists))


def score_trend(
    target_trend: Trend,
    reference_trend: Trend,
    method: str = "euclidean",
    sibling_vectors: Optional[List[np.ndarray]] = None,
    use_normalized: bool = True
) -> float:
    """
    Unified scoring entry point.
    """
    method = method.lower()
    if method == "euclidean":
        return compute_euclidean_score(target_trend, reference_trend, use_normalized=use_normalized)
    elif method == "pca":
        return compute_pca_outlier_score(target_trend, reference_trend, sibling_vectors=sibling_vectors)
    elif method == "knn":
        if sibling_vectors and len(sibling_vectors) > 1:
            return compute_knn_outlier_score(target_trend, sibling_vectors)
        return compute_euclidean_score(target_trend, reference_trend, use_normalized=use_normalized)
    elif method == "cblof":
        if sibling_vectors and len(sibling_vectors) > 2:
            return compute_cblof_score(target_trend, sibling_vectors)
        return compute_euclidean_score(target_trend, reference_trend, use_normalized=use_normalized)
    else:
        return compute_euclidean_score(target_trend, reference_trend, use_normalized=use_normalized)
