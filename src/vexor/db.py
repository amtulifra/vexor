"""
VectorDB — unified interface over all Vexor index types.

Provides a single entry point for adding vectors, searching with optional
metadata filters, deleting, and persisting to disk with WAL-backed durability.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import numpy as np

from vexor.indexes.flat import FlatIndex
from vexor.indexes.kdtree import KDTreeIndex
from vexor.indexes.hnsw import HNSWIndex
from vexor.indexes.ivf import IVFIndex
from vexor.indexes.ivfpq import IVFPQIndex
from vexor.storage.wal import WriteAheadLog
from vexor.storage.format import save_index, load_index
from vexor.filtering.bitmap import Filter
from vexor.hooks.base import VexorHook
from vexor.hooks.noop import NoopHook

_INDEX_TYPES = {"flat", "kdtree", "hnsw", "ivf", "ivfpq"}
_WAL_OP_INSERT = 1
_WAL_OP_DELETE = 2


class VectorDB:
    """
    Unified vector database interface.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    index_type : str
        One of 'flat', 'kdtree', 'hnsw', 'ivf', 'ivfpq'.
    metric : str
        Distance metric: 'cosine', 'l2', or 'inner_product'.
    wal_path : str | None
        If set, all inserts/deletes are written to a WAL for crash recovery.
    hook : VexorHook | None
        Observer hook for algorithm internals. Uses NoopHook by default.
    **index_kwargs
        Forwarded to the underlying index constructor.
    """

    def __init__(
        self,
        dim: int,
        index_type: str = "hnsw",
        metric: str = "cosine",
        wal_path: str | None = None,
        hook: VexorHook | None = None,
        **index_kwargs,
    ) -> None:
        if index_type not in _INDEX_TYPES:
            raise ValueError(f"Unknown index_type '{index_type}'. Choose from {_INDEX_TYPES}.")

        self._dim = dim
        self._index_type = index_type
        self._metric = metric
        self._hook: VexorHook = hook or NoopHook()
        self._wal: WriteAheadLog | None = WriteAheadLog(wal_path) if wal_path else None

        self._index = self._build_index(index_type, dim, metric, hook=self._hook, **index_kwargs)
        self._ivf_trained = False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def index_type(self) -> str:
        return self._index_type

    @property
    def size(self) -> int:
        return self._index.size

    def add(self, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> int:
        vec_id = self._index.add(vector, metadata)
        if self._wal:
            self._wal.append_insert(vec_id, vector, metadata or {})
        return vec_id

    def add_batch(self, vectors: np.ndarray, metadata: list[dict[str, Any]] | None = None) -> list[int]:
        ids = []
        for i, v in enumerate(vectors):
            meta = metadata[i] if metadata else None
            ids.append(self.add(v, meta))
        return ids

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter: Filter | None = None,
        **search_kwargs,
    ) -> list[tuple[int, float]]:
        if isinstance(self._index, KDTreeIndex) and not hasattr(self._index, "_root") or \
                (isinstance(self._index, KDTreeIndex) and self._index._root is None):
            self._index.build()
        return self._index.search(query, k=k, filter=filter, **search_kwargs)

    def delete(self, vec_id: int) -> None:
        if not hasattr(self._index, "delete"):
            raise NotImplementedError(f"{self._index_type} does not support deletion.")
        self._index.delete(vec_id)
        if self._wal:
            self._wal.append_delete(vec_id)

    def train(self, vectors: np.ndarray, **kwargs) -> None:
        """Train IVF/IVFPQ indexes before adding vectors."""
        if not hasattr(self._index, "train"):
            raise NotImplementedError(f"{self._index_type} does not require training.")
        self._index.train(vectors, **kwargs)
        self._ivf_trained = True

    def build_kdtree(self) -> None:
        if not isinstance(self._index, KDTreeIndex):
            raise NotImplementedError("build() is only relevant for the KD-tree index.")
        self._index.build()

    def save(self, path: str) -> None:
        save_index(self._index, path, self._index_type)
        if self._wal:
            self._wal.truncate()

    def recover_from_wal(self) -> int:
        if not self._wal:
            return 0
        entries = self._wal.replay()
        replayed = 0
        for entry in entries:
            if entry["op"] == _WAL_OP_INSERT:
                self._index.add(entry["vector"], entry["metadata"])
                replayed += 1
            elif entry["op"] == _WAL_OP_DELETE and hasattr(self._index, "delete"):
                self._index.delete(entry["vec_id"])
                replayed += 1
        return replayed

    @staticmethod
    def _build_index(index_type: str, dim: int, metric: str, hook: VexorHook, **kwargs):
        if index_type == "flat":
            return FlatIndex(metric=metric, hook=hook)
        if index_type == "kdtree":
            return KDTreeIndex(metric=metric, hook=hook)
        if index_type == "hnsw":
            return HNSWIndex(dim=dim, metric=metric, hook=hook, **kwargs)
        if index_type == "ivf":
            return IVFIndex(metric=metric, hook=hook, **kwargs)
        if index_type == "ivfpq":
            return IVFPQIndex(metric=metric, hook=hook, **kwargs)
        raise ValueError(f"Unknown index_type: {index_type}")
