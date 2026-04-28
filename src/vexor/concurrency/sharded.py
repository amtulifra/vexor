"""
Sharded index manager.

Partitions vectors across N independent HNSW shards. At query time, broadcasts
to all shards in parallel via a forked process pool (bypasses GIL for CPU-bound
search on POSIX systems), then merges top-k results with a heap.

On Windows (no fork), falls back to serial shard search.
"""

from __future__ import annotations
import heapq
import sys
import multiprocessing as mp
import numpy as np
from typing import Any

from vexor.indexes.hnsw import HNSWIndex
from vexor.filtering.bitmap import Filter

# Fork gives zero-copy COW access to parent shard state — no serialization.
# Only available on POSIX (Linux, macOS).
_CAN_FORK = sys.platform != "win32"


def _fork_shard_search(args: tuple) -> list[tuple[int, float]]:
    shard, query, k, filter_obj = args
    return shard.search(query, k=k, filter=filter_obj)


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
        self._shards[shard_id].add(vector, metadata)
        global_id = self._total_added
        self._total_added += 1
        return global_id

    def search(
        self,
        query: np.ndarray,
        k: int,
        filter: Filter | None = None,
    ) -> list[tuple[int, float]]:
        if self._n_shards == 1:
            return self._shards[0].search(query, k=k, filter=filter)

        if _CAN_FORK:
            ctx = mp.get_context("fork")
            args = [(shard, query, k, filter) for shard in self._shards]
            with ctx.Pool(self._n_shards) as pool:
                per_shard = pool.map(_fork_shard_search, args)
        else:
            per_shard = [shard.search(query, k=k, filter=filter)
                         for shard in self._shards]

        return self._merge(per_shard, k)

    def _merge(
        self,
        per_shard: list[list[tuple[int, float]]],
        k: int,
    ) -> list[tuple[int, float]]:
        heap: list[tuple[float, int]] = []
        for shard_idx, results in enumerate(per_shard):
            for local_id, dist in results:
                global_id = local_id * self._n_shards + shard_idx
                heapq.heappush(heap, (dist, global_id))
        return [(gid, d) for d, gid in heapq.nsmallest(k, heap)]
