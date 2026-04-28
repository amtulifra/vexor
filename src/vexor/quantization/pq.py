"""
Product Quantizer (PQ).

Splits each D-dimensional vector into M equal subvectors of D/M dimensions.
Trains a K-centroid codebook per subspace. Encodes vectors as M bytes.

A 768-dim float32 vector (3072 bytes) with M=32 subspaces → 32 bytes (96× reduction).

Asymmetric Distance Computation (ADC): at query time, precompute a (M, K)
lookup table of subspace distances. Approximate distance to any encoded
vector is then M indexed lookups — no multiplication needed.
"""

from __future__ import annotations
import numpy as np
from vexor.hooks.base import VexorHook
from vexor.hooks.noop import NoopHook


class ProductQuantizer:
    """
    Parameters
    ----------
    M : int
        Number of subspaces. Must divide the vector dimension evenly.
    K : int
        Number of centroids per subspace. K=256 lets each code fit in 1 byte.
    n_iter : int
        K-means iterations for codebook training.
    """

    def __init__(self, M: int = 8, K: int = 256, n_iter: int = 25,
                 hook: VexorHook | None = None) -> None:
        self._M = M
        self._K = K
        self._n_iter = n_iter
        self._hook: VexorHook = hook or NoopHook()
        self._codebooks: np.ndarray | None = None  # shape: (M, K, Ds)
        self._Ds: int = 0
        self._is_trained = False

    @property
    def M(self) -> int:
        return self._M

    @property
    def K(self) -> int:
        return self._K

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, vectors: np.ndarray) -> None:
        n, D = vectors.shape
        if D % self._M != 0:
            raise ValueError(f"dim {D} must be divisible by M={self._M}")
        self._Ds = D // self._M
        self._codebooks = np.zeros((self._M, self._K, self._Ds), dtype=np.float32)

        for m in range(self._M):
            sub = vectors[:, m * self._Ds : (m + 1) * self._Ds].copy()
            self._codebooks[m] = _kmeans(sub, self._K, self._n_iter)

        self._is_trained = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode to (N, M) uint8 codes."""
        if not self._is_trained:
            raise RuntimeError("Call train() first.")
        n = len(vectors)
        codes = np.empty((n, self._M), dtype=np.uint8)
        for m in range(self._M):
            sub = vectors[:, m * self._Ds : (m + 1) * self._Ds]
            dists = _batch_l2_to_centroids(sub, self._codebooks[m])
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)
        for i in range(n):
            self._hook.on_pq_encode(i, codes[i])
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate float32 vectors from codes. Shape: (N, D)."""
        n, M = codes.shape
        out = np.empty((n, M * self._Ds), dtype=np.float32)
        for m in range(M):
            out[:, m * self._Ds : (m + 1) * self._Ds] = self._codebooks[m][codes[:, m]]
        return out

    def build_lookup_table(self, query: np.ndarray) -> np.ndarray:
        """
        Precompute (M, K) L2 distance table from query subvectors to all centroids.
        Used for asymmetric distance computation.
        """
        table = np.zeros((self._M, self._K), dtype=np.float32)
        for m in range(self._M):
            q_sub = query[m * self._Ds : (m + 1) * self._Ds]
            diff = self._codebooks[m] - q_sub
            table[m] = np.einsum("kd,kd->k", diff, diff)
        return table

    def adc_distance(self, codes: np.ndarray, table: np.ndarray) -> np.ndarray:
        """
        Asymmetric distance from lookup table to N encoded vectors.
        Shape: (N,). Each distance is sum of M table lookups.
        """
        return table[np.arange(self._M), codes].sum(axis=1)


def _kmeans(data: np.ndarray, K: int, n_iter: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = len(data)
    K = min(K, n)
    centroids = data[rng.choice(n, size=K, replace=False)].copy()

    for _ in range(n_iter):
        dists = _batch_l2_to_centroids(data, centroids)
        assignments = np.argmin(dists, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K, dtype=np.int32)
        for i, c in enumerate(assignments):
            new_centroids[c] += data[i]
            counts[c] += 1
        for c in range(K):
            if counts[c] > 0:
                centroids[c] = new_centroids[c] / counts[c]
            else:
                centroids[c] = data[rng.integers(n)]

    return centroids.astype(np.float32)


def _batch_l2_to_centroids(sub: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Squared L2 from each row in sub to each centroid. Shape: (N, K)."""
    diff = sub[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.einsum("nkd,nkd->nk", diff, diff)
