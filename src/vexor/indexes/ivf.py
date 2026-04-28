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
import heapq
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
        self._centroid_neighbors: np.ndarray | None = None  # (nlist, k_graph)

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
            # Vectorized centroid update via scatter-add
            new_centroids = np.zeros_like(self._centroids)
            counts = np.bincount(assignments, minlength=self._nlist).astype(np.int32)
            np.add.at(new_centroids, assignments, matrix)
            for c in range(self._nlist):
                if counts[c] > 0:
                    old = self._centroids[c].copy()
                    self._centroids[c] = new_centroids[c] / counts[c]
                    self._hook.on_centroid_update(c, old, self._centroids[c])
                else:
                    self._centroids[c] = matrix[rng.integers(len(matrix))]

        self._is_trained = True
        self._build_centroid_graph()

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

        probed_centroids = self._beam_search_centroids(centroid_dists, effective_nprobe)

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

    def _all_centroid_dists(self, matrix: np.ndarray) -> np.ndarray:
        """Vectorized (N, nlist) distance matrix from every row to every centroid."""
        C = self._centroids
        if self._metric == "l2":
            x_sq = np.einsum("ij,ij->i", matrix, matrix)[:, None]
            c_sq = np.einsum("ij,ij->i", C, C)
            return x_sq + c_sq - 2.0 * (matrix @ C.T)
        if self._metric == "cosine":
            x_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
            c_norm = np.linalg.norm(C, axis=1, keepdims=True)
            x_unit = matrix / np.where(x_norm == 0, 1e-10, x_norm)
            c_unit = C / np.where(c_norm == 0, 1e-10, c_norm)
            return 1.0 - x_unit @ c_unit.T
        # inner_product
        return 1.0 - matrix @ C.T

    def _assign_all(self, matrix: np.ndarray) -> np.ndarray:
        return np.argmin(self._all_centroid_dists(matrix), axis=1).astype(np.int32)

    def _build_centroid_graph(self, k: int = 8) -> None:
        """Precompute k nearest centroid neighbors for beam search during query."""
        dists = self._all_centroid_dists(self._centroids)
        np.fill_diagonal(dists, np.inf)
        k = min(k, self._nlist - 1)
        self._centroid_neighbors = np.argsort(dists, axis=1)[:, :k]

    def _beam_search_centroids(self, centroid_dists: np.ndarray, nprobe: int) -> list[int]:
        """
        Greedy beam search over the centroid proximity graph.

        Starts at the nearest centroid and expands via precomputed neighbors,
        always visiting the cheapest unexplored centroid next. Near cluster
        boundaries this finds better probing sets than a raw distance sort.
        """
        if self._centroid_neighbors is None:
            return list(np.argsort(centroid_dists)[:nprobe])

        start = int(np.argmin(centroid_dists))
        visited: set[int] = {start}
        heap: list[tuple[float, int]] = [(float(centroid_dists[start]), start)]
        probed: list[int] = []

        while heap and len(probed) < nprobe:
            _, best = heapq.heappop(heap)
            probed.append(best)
            for nb in self._centroid_neighbors[best]:
                nb = int(nb)
                if nb not in visited:
                    visited.add(nb)
                    heapq.heappush(heap, (float(centroid_dists[nb]), nb))

        return probed

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
