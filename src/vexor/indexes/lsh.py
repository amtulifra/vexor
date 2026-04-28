"""
LSH — Locality Sensitive Hashing index.

Uses random hyperplane projection (sign of W·v) to hash vectors into buckets.
L independent hash tables give recall through redundancy. Search collects
candidates from matching buckets across all tables, then reranks by exact
distance — giving sub-linear query time with no index training required.

When exact-bucket hits are sparse, automatically expands to 1-bit neighbors
(flip each bit) to recover recall at the cost of more candidates.
"""

from __future__ import annotations
import numpy as np
from typing import Any

from vexor.filtering.bitmap import BitmapIndex, Filter


class LSHIndex:
    """
    Random projection LSH index.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    n_tables : int
        Number of independent hash tables. More tables → higher recall,
        more memory and insert time.
    n_hyperplanes : int
        Hyperplanes per table (hash width). More bits → finer buckets,
        fewer candidates per lookup, lower recall on small datasets.
        Typical range: 8–16.
    metric : str
        Distance metric for final reranking: 'cosine', 'l2', 'inner_product'.
    """

    def __init__(
        self,
        dim: int,
        n_tables: int = 10,
        n_hyperplanes: int = 12,
        metric: str = "cosine",
    ) -> None:
        if metric not in ("cosine", "l2", "inner_product"):
            raise ValueError(f"Unknown metric '{metric}'")

        self._dim = dim
        self._L = n_tables
        self._K = n_hyperplanes
        self._metric = metric

        rng = np.random.default_rng(42)
        planes = rng.standard_normal((n_tables, n_hyperplanes, dim)).astype(np.float32)
        norms = np.linalg.norm(planes, axis=2, keepdims=True)
        self._planes = planes / np.where(norms == 0, 1.0, norms)

        self._tables: list[dict[tuple, list[int]]] = [{} for _ in range(n_tables)]
        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []
        self._bitmap = BitmapIndex()

    @property
    def size(self) -> int:
        return len(self._vectors)

    def add(self, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> int:
        vec = vector.astype(np.float32)
        vid = len(self._vectors)
        meta = metadata or {}
        self._vectors.append(vec)
        self._metadata.append(meta)
        self._bitmap.add(vid, meta)

        for t, key in enumerate(self._hash(vec)):
            bucket = self._tables[t].setdefault(key, [])
            bucket.append(vid)

        return vid

    def search(
        self,
        query: np.ndarray,
        k: int,
        filter: Filter | None = None,
    ) -> list[tuple[int, float]]:
        if not self._vectors:
            return []

        query = query.astype(np.float32)
        filter_bitmap = self._bitmap.query(filter) if filter else None
        keys = self._hash(query)

        candidates: set[int] = set()
        for t, key in enumerate(keys):
            for vid in self._tables[t].get(key, []):
                if filter_bitmap is None or vid in filter_bitmap:
                    candidates.add(vid)

        # Expand to 1-bit neighbors when candidates are sparse
        if len(candidates) < k * 4:
            key_lists = [list(key) for key in keys]
            for t, bits in enumerate(key_lists):
                for bit in range(self._K):
                    bits[bit] = not bits[bit]
                    alt = tuple(bits)
                    for vid in self._tables[t].get(alt, []):
                        if filter_bitmap is None or vid in filter_bitmap:
                            candidates.add(vid)
                    bits[bit] = not bits[bit]  # restore

        if not candidates:
            return []

        cand_ids = list(candidates)
        mat = np.stack([self._vectors[vid] for vid in cand_ids])
        dists = self._rerank_dists(query, mat)
        order = np.argsort(dists)[:k]
        return [(cand_ids[i], float(dists[i])) for i in order]

    def _hash(self, vec: np.ndarray) -> list[tuple]:
        """Hash vec against all L tables. Returns L bucket keys."""
        projections = self._planes @ vec  # (L, K)
        return [tuple(row > 0) for row in projections]

    def _rerank_dists(self, query: np.ndarray, mat: np.ndarray) -> np.ndarray:
        if self._metric == "l2":
            diff = mat - query
            return np.einsum("ij,ij->i", diff, diff)
        if self._metric == "cosine":
            dots = mat @ query
            qnorm = float(np.linalg.norm(query))
            rnorms = np.linalg.norm(mat, axis=1)
            denom = rnorms * qnorm
            np.maximum(denom, 1e-10, out=denom)
            return 1.0 - dots / denom
        return 1.0 - mat @ query
