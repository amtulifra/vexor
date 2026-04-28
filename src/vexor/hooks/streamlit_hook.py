from __future__ import annotations
from typing import Any
import numpy as np
import streamlit as st

from vexor.hooks.base import VexorHook


class StreamlitHook(VexorHook):
    """
    Streams algorithm events to st.session_state for the live dashboard.

    Events are appended as dicts so the dashboard can replay them
    frame-by-frame or render them all at once.
    """

    def __init__(self, session_key: str = "vexor_events") -> None:
        self._key = session_key
        if self._key not in st.session_state:
            st.session_state[self._key] = []

    def _emit(self, event: dict[str, Any]) -> None:
        st.session_state[self._key].append(event)

    def on_node_insert(self, node_id: int, layer: int, neighbors: list[int]) -> None:
        self._emit({"type": "node_insert", "node_id": node_id, "layer": layer, "neighbors": neighbors})

    def on_search_visit(self, node_id: int, layer: int, dist: float) -> None:
        self._emit({"type": "search_visit", "node_id": node_id, "layer": layer, "dist": dist})

    def on_centroid_update(self, centroid_id: int, old: np.ndarray, new: np.ndarray) -> None:
        self._emit({
            "type": "centroid_update",
            "centroid_id": centroid_id,
            "old": old.tolist(),
            "new": new.tolist(),
        })

    def on_cluster_assign(self, vec_id: int, centroid_id: int) -> None:
        self._emit({"type": "cluster_assign", "vec_id": vec_id, "centroid_id": centroid_id})

    def on_pq_encode(self, vec_id: int, codes: np.ndarray) -> None:
        self._emit({"type": "pq_encode", "vec_id": vec_id, "codes": codes.tolist()})

    def on_kdtree_split(self, depth: int, axis: int, value: float, n_left: int, n_right: int) -> None:
        self._emit({"type": "kdtree_split", "depth": depth, "axis": axis, "value": value,
                    "n_left": n_left, "n_right": n_right})

    def on_kdtree_visit(self, node_id: int, dist: float, pruned: bool) -> None:
        self._emit({"type": "kdtree_visit", "node_id": node_id, "dist": dist, "pruned": pruned})

    def on_deletion(self, node_id: int, repaired_edges: list[tuple[int, int]]) -> None:
        self._emit({"type": "deletion", "node_id": node_id, "repaired_edges": repaired_edges})

    def clear(self) -> None:
        st.session_state[self._key] = []
