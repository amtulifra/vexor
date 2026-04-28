from __future__ import annotations
from typing import Any
import numpy as np


class VexorHook:
    """
    Observer interface for algorithm internals.

    Override any method to receive events. All methods are no-ops by default
    so subclasses only override what they need. Core algorithms accept a hook
    instance and call these methods at key moments — the hook never affects
    algorithm correctness or output.
    """

    def on_node_insert(self, node_id: int, layer: int, neighbors: list[int]) -> None:
        pass

    def on_search_visit(self, node_id: int, layer: int, dist: float) -> None:
        pass

    def on_centroid_update(self, centroid_id: int, old: np.ndarray, new: np.ndarray) -> None:
        pass

    def on_cluster_assign(self, vec_id: int, centroid_id: int) -> None:
        pass

    def on_pq_encode(self, vec_id: int, codes: np.ndarray) -> None:
        pass

    def on_kdtree_split(self, depth: int, axis: int, value: float, n_left: int, n_right: int) -> None:
        pass

    def on_kdtree_visit(self, node_id: int, dist: float, pruned: bool) -> None:
        pass

    def on_deletion(self, node_id: int, repaired_edges: list[tuple[int, int]]) -> None:
        pass
