"""
IVFPQ — IVF with Product Quantization on residuals.

Pipeline:
  1. Coarse IVF: assign vector to nearest centroid.
  2. Compute residual (vector minus centroid).
  3. PQ-encode the residual (not the raw vector — residuals have lower variance,
     better codebook utilization).
  4. Search: build ADC table for residual, score all PQ codes in the nprobe
     clusters, rerank top candidates with exact float32 distance.
"""

from __future__ import annotations
import numpy as np
from typing import Any

from vexor.quantization.pq import ProductQuantizer
from vexor.distance.kernels import batch_cosine, batch_l2, batch_inner_product
from vexor.filtering.bitmap import BitmapIndex, Filter
from vexor.hooks.base import VexorHook
from vexor.hooks.noop import NoopHook


_BATCH_FN = {
    "cosine": batch_cosine,
    "l2": batch_l2,
    "inner_product": batch_inner_product,
}

_RERANK_CANDIDATES = 4096


def _kmeans_plus_plus(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(data)
    k = min(k, n)
    centroids = [data[int(rng.integers(n))].copy()]
    for _ in range(1, k):
        min_sq = np.full(n, np.inf, dtype=np.float64)
        for c in centroids:
            diff = data - c
            sq = np.einsum("ij,ij->i", diff, diff).astype(np.float64)
            np.minimum(min_sq, sq, out=min_sq)
        total = min_sq.sum()
        if total == 0.0:
            break
        centroids.append(data[int(rng.choice(n, p=min_sq / total))].copy())
    return np.array(centroids, dtype=np.float32)


class IVFPQIndex:
    """
    IVFPQ approximate k-NN index.

    Parameters
    ----------
    nlist : int   Number of IVF clusters (coarse quantizer).
    nprobe : int  Clusters to search per query.
    M : int       PQ subspaces. Memory per vector = M bytes.
    K : int       Codebook size per subspace (256 = 1 byte per subspace).
    metric : str  Distance metric for coarse assignment and exact reranking.
    """

    def __init__(
        self,
        nlist: int = 256,
        nprobe: int = 8,
        M: int = 8,
        K: int = 256,
        metric: str = "l2",
        hook: VexorHook | None = None,
    ) -> None:
        if metric not in _BATCH_FN:
            raise ValueError(f"Unknown metric '{metric}'")
        self._nlist = nlist
        self._nprobe = nprobe
        self._M = M
        self._K = K
        self._metric = metric
        self._hook: VexorHook = hook or NoopHook()
        self._batch_dist = _BATCH_FN[metric]

        self._coarse_centroids: np.ndarray | None = None
        self._pq = ProductQuantizer(M=M, K=K, hook=hook)
        self._inverted_lists: list[list[int]] = [[] for _ in range(nlist)]
        self._codes: dict[int, np.ndarray] = {}
        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []
        self._bitmap = BitmapIndex()
        self._is_trained = False

    @property
    def size(self) -> int:
        return len(self._vectors)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, vectors: np.ndarray, n_iter: int = 50) -> None:
        """Train coarse centroids and PQ codebooks."""
        matrix = vectors.astype(np.float32)
        rng = np.random.default_rng(0)

        centroids = _kmeans_plus_plus(matrix, self._nlist, rng)

        n = len(matrix)
        for _ in range(n_iter):
            dists = np.stack([self._batch_dist(matrix[i], centroids) for i in range(n)])
            assignments = np.argmin(dists, axis=1)
            new_c = np.zeros_like(centroids)
            counts = np.zeros(len(centroids), dtype=np.int32)
            for i, c in enumerate(assignments):
                new_c[c] += matrix[i]
                counts[c] += 1
            for c in range(len(centroids)):
                if counts[c] > 0:
                    centroids[c] = new_c[c] / counts[c]

        self._coarse_centroids = centroids

        assignments = np.array([int(np.argmin(self._batch_dist(v, centroids))) for v in matrix])
        residuals = matrix - centroids[assignments]
        self._pq.train(residuals)
        self._is_trained = True

    def add(self, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> int:
        if not self._is_trained:
            raise RuntimeError("Call train() before add().")

        vec = vector.astype(np.float32)
        meta = metadata or {}
        vec_id = len(self._vectors)
        self._vectors.append(vec)
        self._metadata.append(meta)
        self._bitmap.add(vec_id, meta)

        c_id = int(np.argmin(self._batch_dist(vec, self._coarse_centroids)))
        residual = vec - self._coarse_centroids[c_id]
        code = self._pq.encode(residual.reshape(1, -1))[0]
        self._codes[vec_id] = code
        self._inverted_lists[c_id].append(vec_id)
        self._hook.on_pq_encode(vec_id, code)
        return vec_id

    def search(
        self,
        query: np.ndarray,
        k: int,
        nprobe: int | None = None,
        filter: Filter | None = None,
    ) -> list[tuple[int, float]]:
        if not self._is_trained:
            return []

        query = query.astype(np.float32)
        effective_nprobe = nprobe or self._nprobe

        centroid_dists = self._batch_dist(query, self._coarse_centroids)
        probed_ids = np.argsort(centroid_dists)[:effective_nprobe]

        if filter:
            filter_bitmap = self._bitmap.query(filter)
        else:
            filter_bitmap = None

        all_candidates: list[tuple[float, int]] = []

        for c_id in probed_ids:
            q_residual = query - self._coarse_centroids[c_id]
            table = self._pq.build_lookup_table(q_residual)

            list_ids = self._inverted_lists[c_id]
            if not list_ids:
                continue

            if filter_bitmap is not None:
                list_ids = [i for i in list_ids if i in filter_bitmap]

            if not list_ids:
                continue

            codes_batch = np.stack([self._codes[i] for i in list_ids])
            approx_dists = self._pq.adc_distance(codes_batch, table)

            for j, vec_id in enumerate(list_ids):
                all_candidates.append((float(approx_dists[j]), vec_id))

        all_candidates.sort(key=lambda x: x[0])
        rerank_pool = all_candidates[:_RERANK_CANDIDATES]

        if not rerank_pool:
            return []

        exact_results: list[tuple[int, float]] = []
        for _, vec_id in rerank_pool:
            d = float(self._batch_dist(query, self._vectors[vec_id].reshape(1, -1))[0])
            exact_results.append((vec_id, d))

        exact_results.sort(key=lambda x: x[1])
        return exact_results[:k]
