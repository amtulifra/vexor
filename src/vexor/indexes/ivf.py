"""
IVF — Inverted File Index with online k-means centroid updates.

Improvements over a static IVF:
  - Online centroid updates via mini-batch gradient descent so centroids
    track a streaming data distribution without full retraining.
  - Adaptive nprobe that scales with query difficulty (residual to nearest centroid).
  - Bitmap-intersected cluster selection: clusters whose member bitmap does not
    overlap with the filter predicate are skipped entirely.
"""

from __future__ import annotations
import numpy as np
from typing import Any

from vexor.distance.kernels import batch_cosine, batch_l2, batch_inner_product
from vexor.filtering.bitmap import BitmapIndex, Filter
from vexor.filtering.adaptive import adaptive_nprobe
from vexor.hooks.base import VexorHook
from vexor.hooks.noop import NoopHook


_BATCH_FN = {
    "cosine": batch_cosine,
    "l2": batch_l2,
    "inner_product": batch_inner_product,
}

_ONLINE_LR_INIT = 0.05
_ONLINE_LR_DECAY = 0.001


class IVFIndex:
    """
    IVF approximate k-NN index.

    Parameters
    ----------
    nlist : int
        Number of Voronoi cells (centroids).
    nprobe : int
        Number of cells to search per query. Higher = better recall, slower.
    metric : str
        Distance metric.
    online_updates : bool
        If True, centroids are updated incrementally as new vectors are added.
    """

    def __init__(
        self,
        nlist: int = 256,
        nprobe: int = 8,
        metric: str = "cosine",
        online_updates: bool = True,
        hook: VexorHook | None = None,
    ) -> None:
        if metric not in _BATCH_FN:
            raise ValueError(f"Unknown metric '{metric}'")
        self._nlist = nlist
        self._nprobe = nprobe
        self._metric = metric
        self._online_updates = online_updates
        self._hook: VexorHook = hook or NoopHook()
        self._batch_dist = _BATCH_FN[metric]

        self._centroids: np.ndarray | None = None
        self._inverted_lists: list[list[int]] = [[] for _ in range(nlist)]
        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []
        self._cluster_bitmaps: list[BitmapIndex] = [BitmapIndex() for _ in range(nlist)]
        self._global_bitmap = BitmapIndex()
        self._is_trained = False
        self._insert_count = 0

    @property
    def size(self) -> int:
        return len(self._vectors)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, vectors: np.ndarray, n_iter: int = 50) -> None:
        """Train centroids via k-means++ on the provided vectors."""
        matrix = vectors.astype(np.float32)
        rng = np.random.default_rng(0)
        self._centroids = _kmeans_plus_plus(matrix, self._nlist, rng)

        for _ in range(n_iter):
            assignments = self._assign_all(matrix)
            new_centroids = np.zeros_like(self._centroids)
            counts = np.zeros(self._nlist, dtype=np.int32)
            for i, c in enumerate(assignments):
                new_centroids[c] += matrix[i]
                counts[c] += 1
            for c in range(self._nlist):
                if counts[c] > 0:
                    old = self._centroids[c].copy()
                    self._centroids[c] = new_centroids[c] / counts[c]
                    self._hook.on_centroid_update(c, old, self._centroids[c])
                else:
                    random_vec = matrix[rng.integers(len(matrix))]
                    self._centroids[c] = random_vec

        self._is_trained = True

    def add(self, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> int:
        if not self._is_trained:
            raise RuntimeError("Call train() before add().")

        vec = vector.astype(np.float32)
        vec_id = len(self._vectors)
        meta = metadata or {}
        self._vectors.append(vec)
        self._metadata.append(meta)

        centroid_id = self._nearest_centroid(vec)
        self._inverted_lists[centroid_id].append(vec_id)
        self._cluster_bitmaps[centroid_id].add(vec_id, meta)
        self._global_bitmap.add(vec_id, meta)
        self._hook.on_cluster_assign(vec_id, centroid_id)

        if self._online_updates:
            self._incremental_update(centroid_id, vec)

        self._insert_count += 1
        return vec_id

    def search(
        self,
        query: np.ndarray,
        k: int,
        nprobe: int | None = None,
        filter: Filter | None = None,
        use_adaptive_nprobe: bool = True,
    ) -> list[tuple[int, float]]:
        if not self._is_trained or self._centroids is None:
            return []

        query = query.astype(np.float32)
        centroid_dists = self._batch_dist(query, self._centroids)

        effective_nprobe = nprobe or self._nprobe
        if use_adaptive_nprobe:
            nearest_dist = float(centroid_dists.min())
            mean_dist = float(centroid_dists.mean())
            effective_nprobe = adaptive_nprobe(effective_nprobe, self._nlist,
                                               nearest_dist, mean_dist)

        probed_centroids = np.argsort(centroid_dists)[:effective_nprobe]

        if filter:
            filter_bitmap = self._global_bitmap.query(filter)
        else:
            filter_bitmap = None

        candidates: list[tuple[float, int]] = []
        for c_id in probed_centroids:
            if filter_bitmap is not None:
                cluster_ids = set(self._cluster_bitmaps[c_id]._all_ids)
                if not (cluster_ids & set(filter_bitmap)):
                    continue

            for vec_id in self._inverted_lists[c_id]:
                if filter_bitmap is not None and vec_id not in filter_bitmap:
                    continue
                d = float(self._batch_dist(query, self._vectors[vec_id].reshape(1, -1))[0])
                candidates.append((d, vec_id))

        candidates.sort(key=lambda x: x[0])
        return [(vid, d) for d, vid in candidates[:k]]

    def _nearest_centroid(self, vec: np.ndarray) -> int:
        dists = self._batch_dist(vec, self._centroids)
        return int(np.argmin(dists))

    def _assign_all(self, matrix: np.ndarray) -> np.ndarray:
        dists = np.stack([self._batch_dist(matrix[i], self._centroids) for i in range(len(matrix))])
        return np.argmin(dists, axis=1)

    def _incremental_update(self, centroid_id: int, vec: np.ndarray) -> None:
        lr = _ONLINE_LR_INIT / (1.0 + self._insert_count * _ONLINE_LR_DECAY)
        old = self._centroids[centroid_id].copy()
        self._centroids[centroid_id] += lr * (vec - self._centroids[centroid_id])
        self._hook.on_centroid_update(centroid_id, old, self._centroids[centroid_id])


def _kmeans_plus_plus(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ initialization: spread initial centroids to speed convergence."""
    n = len(data)
    k = min(k, n)
    first = int(rng.integers(n))
    centroids = [data[first].copy()]

    for _ in range(1, k):
        # Squared distance from each point to its nearest chosen centroid
        min_sq_dists = np.full(n, np.inf, dtype=np.float64)
        for c in centroids:
            diff = data - c
            sq = np.einsum("ij,ij->i", diff, diff).astype(np.float64)
            np.minimum(min_sq_dists, sq, out=min_sq_dists)
        total = min_sq_dists.sum()
        if total == 0.0:
            break
        probs = min_sq_dists / total
        idx = int(rng.choice(n, p=probs))
        centroids.append(data[idx].copy())

    return np.array(centroids, dtype=np.float32)
