"""
KD-Tree index for exact k-NN in low-dimensional spaces.

Splits on the axis of highest variance at the median. At each node the union
of metadata values is tracked so entire subtrees can be pruned when a filter
predicate cannot possibly be satisfied.

Performance degrades above ~20 dimensions due to the curse of dimensionality —
every point becomes equidistant from the splitting hyperplane, eliminating
branch pruning. This failure mode is illustrated in the visualizer.
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from vexor.hooks.base import VexorHook
from vexor.hooks.noop import NoopHook
from vexor.filtering.bitmap import Filter


@dataclass
class _KDNode:
    axis: int = 0
    split_value: float = 0.0
    vec_ids: list[int] = field(default_factory=list)
    left: "_KDNode | None" = None
    right: "_KDNode | None" = None
    meta_union: dict[str, set] = field(default_factory=dict)
    depth: int = 0


class KDTreeIndex:
    """Exact k-NN via KD-tree with metadata subtree pruning."""

    _LEAF_SIZE = 20

    def __init__(self, metric: str = "l2", hook: VexorHook | None = None) -> None:
        if metric not in ("l2", "cosine", "inner_product"):
            raise ValueError(f"Unknown metric '{metric}'")
        self._metric = metric
        self._hook: VexorHook = hook or NoopHook()
        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []
        self._root: _KDNode | None = None

    @property
    def size(self) -> int:
        return len(self._vectors)

    def add(self, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> int:
        vec_id = len(self._vectors)
        self._vectors.append(vector.astype(np.float32))
        self._metadata.append(metadata or {})
        return vec_id

    def build(self) -> None:
        """Build the tree from all added vectors. Must be called before search."""
        ids = list(range(len(self._vectors)))
        matrix = np.stack(self._vectors)
        self._root = self._build_node(ids, matrix, depth=0)

    def _build_node(self, ids: list[int], matrix: np.ndarray, depth: int) -> _KDNode:
        node = _KDNode(depth=depth)
        node.meta_union = _union_metadata([self._metadata[i] for i in ids])

        if len(ids) <= self._LEAF_SIZE:
            node.vec_ids = ids
            self._hook.on_kdtree_split(depth, -1, 0.0, 0, 0)
            return node

        sub = matrix[ids]
        axis = int(np.argmax(np.var(sub, axis=0)))
        median = float(np.median(sub[:, axis]))

        left_ids = [i for i in ids if matrix[i, axis] <= median]
        right_ids = [i for i in ids if matrix[i, axis] > median]

        # Degenerate split guard — force a split if all points land on one side
        if not left_ids or not right_ids:
            node.vec_ids = ids
            return node

        self._hook.on_kdtree_split(depth, axis, median, len(left_ids), len(right_ids))
        node.axis = axis
        node.split_value = median
        node.left = self._build_node(left_ids, matrix, depth + 1)
        node.right = self._build_node(right_ids, matrix, depth + 1)
        return node

    def search(
        self,
        query: np.ndarray,
        k: int,
        filter: Filter | None = None,
    ) -> list[tuple[int, float]]:
        if self._root is None:
            raise RuntimeError("Call build() before search().")

        query = query.astype(np.float32)
        heap: list[tuple[float, int]] = []  # max-heap via negation

        def _dist(vec_id: int) -> float:
            v = self._vectors[vec_id]
            if self._metric == "l2":
                diff = query - v
                return float(np.dot(diff, diff))
            if self._metric == "cosine":
                d = float(np.dot(query, v))
                n = float(np.linalg.norm(query) * np.linalg.norm(v))
                return 1.0 - d / n if n > 0 else 1.0
            # inner_product
            return float(1.0 - np.dot(query, v))

        def _search(node: _KDNode) -> None:
            if node is None:
                return

            if filter and not _meta_satisfies_union(filter, node.meta_union):
                self._hook.on_kdtree_visit(id(node), 0.0, True)
                return

            if node.vec_ids:
                for vec_id in node.vec_ids:
                    if filter and not _meta_satisfies(filter, self._metadata[vec_id]):
                        continue
                    d = _dist(vec_id)
                    self._hook.on_kdtree_visit(vec_id, d, False)
                    if len(heap) < k:
                        heapq.heappush(heap, (-d, vec_id))
                    elif d < -heap[0][0]:
                        heapq.heapreplace(heap, (-d, vec_id))
                return

            diff = query[node.axis] - node.split_value
            closer, farther = (node.left, node.right) if diff <= 0 else (node.right, node.left)
            _search(closer)

            worst = -heap[0][0] if heap else float("inf")
            hyperplane_dist = diff * diff if self._metric == "l2" else abs(diff)
            if len(heap) < k or hyperplane_dist < worst:
                _search(farther)

        _search(self._root)
        results = [(-neg_d, vec_id) for neg_d, vec_id in heap]
        results.sort(key=lambda x: x[0])
        return [(vec_id, d) for d, vec_id in results]


def _union_metadata(metas: list[dict[str, Any]]) -> dict[str, set]:
    union: dict[str, set] = {}
    for m in metas:
        for k, v in m.items():
            if k not in union:
                union[k] = set()
            union[k].add(v)
    return union


def _meta_satisfies_union(filter: Filter, union: dict[str, set]) -> bool:
    for field, value in filter.items():
        values = union.get(field, set())
        if value not in values:
            return False
    return True


def _meta_satisfies(filter: Filter, meta: dict[str, Any]) -> bool:
    for field, value in filter.items():
        if meta.get(field) != value:
            return False
    return True
