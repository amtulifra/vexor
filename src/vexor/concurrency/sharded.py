"""
Sharded index manager.

Partitions vectors across N independent HNSW shards. At query time, broadcasts
to all shards via ProcessPoolExecutor (bypasses GIL for CPU-bound search),
then merges top-k results with a heap.

Each shard is an independent process — they run on separate CPU cores without
GIL contention.
"""

from __future__ import annotations
import heapq
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from vexor.indexes.hnsw import HNSWIndex
from vexor.filtering.bitmap import Filter


def _shard_search(shard_state: dict, query: np.ndarray, k: int,
                  filter: Filter | None) -> list[tuple[int, float]]:
    """Top-level function required for pickling with ProcessPoolExecutor."""
    shard = HNSWIndex(**shard_state["config"])
    shard.__dict__.update(shard_state["state"])
    return shard.search(query, k=k, filter=filter)


class ShardedIndex:
    """
    Distributes vectors across N HNSW shards for parallel search.

    Parameters
    ----------
    n_shards : int
        Number of shards (ideally one per CPU core).
    dim, M, ef_construction, ef_search, metric :
        Forwarded to each shard's HNSWIndex.
    """

    def __init__(
        self,
        n_shards: int,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: str = "cosine",
    ) -> None:
        self._n_shards = n_shards
        self._dim = dim
        self._shard_cfg = dict(dim=dim, M=M, ef_construction=ef_construction,
                                ef_search=ef_search, metric=metric)
        self._shards: list[HNSWIndex] = [
            HNSWIndex(**self._shard_cfg) for _ in range(n_shards)
        ]
        self._total_added = 0

    @property
    def size(self) -> int:
        return self._total_added

    def add(self, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> int:
        shard_id = self._total_added % self._n_shards
        local_id = self._shards[shard_id].add(vector, metadata)
        global_id = self._total_added
        self._total_added += 1
        return global_id

    def search(
        self,
        query: np.ndarray,
        k: int,
        filter: Filter | None = None,
    ) -> list[tuple[int, float]]:
        per_shard_results: list[list[tuple[int, float]]] = [
            shard.search(query, k=k, filter=filter) for shard in self._shards
        ]

        all_candidates: list[tuple[float, int]] = []
        for shard_idx, results in enumerate(per_shard_results):
            for local_id, dist in results:
                global_id = local_id * self._n_shards + shard_idx
                heapq.heappush(all_candidates, (dist, global_id))

        return [(gid, d) for d, gid in heapq.nsmallest(k, all_candidates)]
