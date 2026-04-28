"""
Flat (brute-force) exact k-NN index.

100% recall. O(N·D) per query. Serves as the correctness ground truth
for evaluating all approximate indexes.
"""

from __future__ import annotations
import numpy as np
from typing import Any

from vexor.distance.kernels import batch_cosine, batch_l2, batch_inner_product
from vexor.filtering.bitmap import BitmapIndex, Filter
from vexor.hooks.base import VexorHook
from vexor.hooks.noop import NoopHook


_BATCH_FN = {
    "cosine": batch_cosine,
    "l2": batch_l2,
    "inner_product": batch_inner_product,
}


class FlatIndex:
    """Exact k-NN via exhaustive distance computation."""

    def __init__(self, metric: str = "cosine", hook: VexorHook | None = None) -> None:
        if metric not in _BATCH_FN:
            raise ValueError(f"Unknown metric '{metric}'. Choose from {set(_BATCH_FN)}")
        self._metric = metric
        self._batch_dist = _BATCH_FN[metric]
        self._hook: VexorHook = hook or NoopHook()

        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []
        self._bitmap = BitmapIndex()

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def size(self) -> int:
        return len(self._vectors)

    def add(self, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> int:
        vec_id = len(self._vectors)
        self._vectors.append(vector.astype(np.float32))
        meta = metadata or {}
        self._metadata.append(meta)
        self._bitmap.add(vec_id, meta)
        self._hook.on_node_insert(vec_id, 0, [])
        return vec_id

    def search(
        self,
        query: np.ndarray,
        k: int,
        filter: Filter | None = None,
    ) -> list[tuple[int, float]]:
        if not self._vectors:
            return []

        matrix = np.stack(self._vectors)
        distances = self._batch_dist(query.astype(np.float32), matrix)

        for i, d in enumerate(distances):
            self._hook.on_search_visit(i, 0, float(d))

        if filter:
            valid_ids = self._bitmap.query(filter)
            mask = np.zeros(len(self._vectors), dtype=bool)
            for vid in valid_ids:
                mask[vid] = True
            distances = np.where(mask, distances, np.inf)

        top_k = int(min(k, len(self._vectors)))
        indices = np.argpartition(distances, top_k - 1)[:top_k]
        indices = indices[np.argsort(distances[indices])]

        return [(int(i), float(distances[i])) for i in indices if not np.isinf(distances[i])]
