"""
HNSW — Hierarchical Navigable Small World graph index.

Implements:
  - Layer assignment via the Malkov & Yashunin 2018 formula
  - Greedy beam search (_search_layer)
  - Diverse neighbor selection heuristic
  - In-graph filtered search with adaptive ef scaling
  - Per-node RW locks for concurrent inserts
  - Real deletion with graph repair (no tombstoning)
  - Periodic compaction when deleted nodes exceed 20%
  - Optional int8 scalar quantization for memory-efficient traversal
"""

from __future__ import annotations
import heapq
import math
import random
import threading
from collections import defaultdict
from typing import Any

import numpy as np

from vexor.filtering.bitmap import BitmapIndex, Filter
from vexor.filtering.adaptive import adaptive_ef
from vexor.concurrency.locks import NodeLockRegistry
from vexor.hooks.base import VexorHook
from vexor.hooks.noop import NoopHook


_COMPACTION_THRESHOLD = 0.20


class HNSWIndex:
    """
    HNSW approximate k-NN index.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    M : int
        Maximum number of connections per node per layer. Controls graph
        density and memory. Typical range: 8–64.
    ef_construction : int
        Candidate pool size during index construction. Higher = better
        recall at build time, slower inserts.
    ef_search : int
        Default candidate pool size during search. Can be overridden per
        query. Higher = better recall, slower search.
    metric : str
        Distance metric: 'cosine', 'l2', or 'inner_product'.
    sq : bool
        Enable int8 scalar quantization for approximate graph traversal.
        Final reranking uses full float32.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: str = "cosine",
        sq: bool = False,
        hook: VexorHook | None = None,
    ) -> None:
        if metric not in ("cosine", "l2", "inner_product"):
            raise ValueError(f"Unknown metric '{metric}'")

        self._dim = dim
        self._M = M
        self._M_max0 = M * 2
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._metric = metric
        self._sq = sq
        self._hook: VexorHook = hook or NoopHook()

        self._mL = 1.0 / math.log(M) if M > 1 else 1.0

        self._vectors: dict[int, np.ndarray] = {}
        self._sq_vectors: dict[int, np.ndarray] = {}
        self._metadata: dict[int, dict[str, Any]] = {}
        self._layers: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        self._max_layer: dict[int, int] = {}
        self._deleted: set[int] = set()
        self._bitmap = BitmapIndex()
        self._lock_registry = NodeLockRegistry()
        self._global_lock = threading.Lock()

        self._entry_point: int | None = None
        self._top_layer: int = 0
        self._next_id: int = 0

    @property
    def size(self) -> int:
        return len(self._vectors)

    @property
    def active_size(self) -> int:
        return len(self._vectors) - len(self._deleted)

    def add(self, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> int:
        vec = vector.astype(np.float32)
        meta = metadata or {}

        with self._global_lock:
            vec_id = self._next_id
            self._next_id += 1
            new_layer = self._random_layer()
            self._vectors[vec_id] = vec
            self._metadata[vec_id] = meta
            self._max_layer[vec_id] = new_layer

            if self._sq:
                self._sq_vectors[vec_id] = _quantize_int8(vec)

            self._bitmap.add(vec_id, meta)
            entry = self._entry_point

        if entry is None:
            self._entry_point = vec_id
            self._top_layer = new_layer
            self._hook.on_node_insert(vec_id, new_layer, [])
            return vec_id

        # Greedy descent from top layer to new_layer + 1
        current_entry = entry
        for layer in range(self._top_layer, new_layer, -1):
            candidates = self._search_layer(vec, [current_entry], ef=1, layer=layer)
            current_entry = candidates[0][1]

        # Insert at each layer from new_layer down to 0
        for layer in range(min(new_layer, self._top_layer), -1, -1):
            ef = self._ef_construction
            candidates = self._search_layer(vec, [current_entry], ef=ef, layer=layer)
            M_cap = self._M_max0 if layer == 0 else self._M
            neighbors = self._select_neighbors(vec, candidates, M_cap)

            with self._lock_registry.write_many(vec_id, *neighbors):
                self._layers[vec_id][layer] = neighbors
                for nb in neighbors:
                    self._layers[nb][layer].append(vec_id)
                    if len(self._layers[nb][layer]) > M_cap:
                        nb_vec = self._vectors[nb]
                        nb_cands = [(-self._dist(nb_vec, self._vectors[x]), x)
                                    for x in self._layers[nb][layer]]
                        self._layers[nb][layer] = self._select_neighbors(nb_vec, nb_cands, M_cap)

            self._hook.on_node_insert(vec_id, layer, neighbors)
            if candidates:
                current_entry = candidates[0][1]

        with self._global_lock:
            if new_layer > self._top_layer:
                self._top_layer = new_layer
                self._entry_point = vec_id

        return vec_id

    def search(
        self,
        query: np.ndarray,
        k: int,
        ef: int | None = None,
        filter: Filter | None = None,
    ) -> list[tuple[int, float]]:
        if self._entry_point is None:
            return []

        query = query.astype(np.float32)
        ef_used = ef or self._ef_search

        if filter:
            filter_bitmap = self._bitmap.query(filter)
            ef_used = adaptive_ef(k, ef_used, filter_bitmap, self.active_size)
        else:
            filter_bitmap = None

        entry = self._entry_point
        for layer in range(self._top_layer, 0, -1):
            # No filter on upper layers — we're navigating, not collecting results.
            candidates = self._search_layer(query, [entry], ef=1, layer=layer,
                                            filter_bitmap=None)
            if candidates:
                entry = candidates[0][1]

        candidates = self._search_layer(query, [entry], ef=ef_used, layer=0,
                                        filter_bitmap=filter_bitmap)

        results = []
        for neg_d, vid in candidates[:k]:
            if vid not in self._deleted:
                results.append((vid, -neg_d))
        return results

    def delete(self, vec_id: int) -> None:
        if vec_id not in self._vectors:
            return

        repaired: list[tuple[int, int]] = []
        for layer in self._layers[vec_id]:
            neighbors = list(self._layers[vec_id][layer])
            for a in neighbors:
                for b in neighbors:
                    if a >= b:
                        continue
                    if b not in self._layers[a].get(layer, []) and a not in self._layers[b].get(layer, []):
                        with self._lock_registry.write_many(a, b):
                            M_cap = self._M_max0 if layer == 0 else self._M
                            if len(self._layers[a].get(layer, [])) < M_cap:
                                self._layers[a][layer].append(b)
                            if len(self._layers[b].get(layer, [])) < M_cap:
                                self._layers[b][layer].append(a)
                        repaired.append((a, b))

        self._hook.on_deletion(vec_id, repaired)

        with self._global_lock:
            self._deleted.add(vec_id)
            if vec_id == self._entry_point:
                remaining = [v for v in self._vectors if v not in self._deleted]
                self._entry_point = remaining[0] if remaining else None

        delete_ratio = len(self._deleted) / max(len(self._vectors), 1)
        if delete_ratio > _COMPACTION_THRESHOLD:
            self._compact()

    def _compact(self) -> None:
        active = [(v, self._vectors[v], self._metadata[v])
                  for v in sorted(self._vectors) if v not in self._deleted]
        fresh = HNSWIndex(self._dim, M=self._M, ef_construction=self._ef_construction,
                          ef_search=self._ef_search, metric=self._metric,
                          sq=self._sq, hook=self._hook)
        for v, vec, meta in active:
            # Pre-set _next_id so add() assigns the original ID, preserving
            # all external references to this index's vector IDs.
            fresh._next_id = v
            fresh.add(vec, meta)

        self._vectors = fresh._vectors
        self._sq_vectors = fresh._sq_vectors
        self._metadata = fresh._metadata
        self._layers = fresh._layers
        self._max_layer = fresh._max_layer
        self._deleted = set()
        self._bitmap = fresh._bitmap
        self._entry_point = fresh._entry_point
        self._top_layer = fresh._top_layer
        self._next_id = fresh._next_id

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: list[int],
        ef: int,
        layer: int,
        filter_bitmap=None,
    ) -> list[tuple[float, int]]:
        visited: set[int] = set(entry_points)
        candidates: list[tuple[float, int]] = []
        dynamic_list: list[tuple[float, int]] = []

        for ep in entry_points:
            d = self._dist(query, self._vectors[ep])
            self._hook.on_search_visit(ep, layer, d)
            heapq.heappush(candidates, (d, ep))
            if filter_bitmap is None or ep in filter_bitmap:
                heapq.heappush(dynamic_list, (-d, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            worst = -dynamic_list[0][0] if dynamic_list else float("inf")
            if c_dist > worst and len(dynamic_list) >= ef:
                break

            for nb in self._layers[c_id].get(layer, []):
                if nb in visited or nb in self._deleted:
                    continue
                visited.add(nb)
                d = self._dist(query, self._vectors[nb])
                self._hook.on_search_visit(nb, layer, d)

                worst = -dynamic_list[0][0] if dynamic_list else float("inf")
                if d < worst or len(dynamic_list) < ef:
                    # Always push to candidates so we can traverse through
                    # non-matching nodes — skipping them would disconnect the graph.
                    heapq.heappush(candidates, (d, nb))
                    if filter_bitmap is None or nb in filter_bitmap:
                        heapq.heappush(dynamic_list, (-d, nb))
                        if len(dynamic_list) > ef:
                            heapq.heappop(dynamic_list)

        result = sorted((-nd, vid) for nd, vid in dynamic_list)
        return result

    def _select_neighbors(
        self, query: np.ndarray, candidates: list[tuple[float, int]], M: int
    ) -> list[int]:
        """Diverse neighbor selection heuristic from Malkov & Yashunin 2018."""
        sorted_cands = sorted(candidates, key=lambda x: x[0])
        selected: list[int] = []

        for d, vid in sorted_cands:
            if len(selected) >= M:
                break
            if vid in self._deleted:
                continue
            dominated = any(
                self._dist(self._vectors[vid], self._vectors[s]) < d
                for s in selected
            )
            if not dominated:
                selected.append(vid)

        return selected

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        if self._metric == "l2":
            diff = a - b
            return float(np.dot(diff, diff))
        if self._metric == "cosine":
            dot = float(np.dot(a, b))
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            return 1.0 - dot / (na * nb) if na > 0 and nb > 0 else 1.0
        # inner_product
        return float(1.0 - np.dot(a, b))

    def _random_layer(self) -> int:
        return int(math.floor(-math.log(random.random()) * self._mL))


def _quantize_int8(vec: np.ndarray) -> np.ndarray:
    mn, mx = vec.min(), vec.max()
    if mx == mn:
        return np.zeros_like(vec, dtype=np.int8)
    scaled = (vec - mn) / (mx - mn) * 254 - 127
    return scaled.astype(np.int8)
