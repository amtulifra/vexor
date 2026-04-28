"""
In-memory vector store with optional memmap persistence.

Holds the ground-truth (N, D) float32 array and per-vector metadata.
All indexes are search structures that reference IDs into this store.
"""

from __future__ import annotations
import os
import numpy as np
from typing import Any


class VectorStore:
    def __init__(self, dim: int, capacity: int = 100_000, mmap_path: str | None = None) -> None:
        self._dim = dim
        self._capacity = capacity
        self._count = 0
        self._metadata: list[dict[str, Any]] = []
        self._deleted: set[int] = set()

        if mmap_path:
            self._mmap_path = mmap_path
            self._vectors = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(capacity, dim))
        else:
            self._mmap_path = None
            self._vectors = np.empty((capacity, dim), dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def count(self) -> int:
        return self._count

    @property
    def active_count(self) -> int:
        return self._count - len(self._deleted)

    def add(self, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> int:
        if self._count >= self._capacity:
            self._grow()

        vec_id = self._count
        self._vectors[vec_id] = vector.astype(np.float32)
        self._metadata.append(metadata or {})
        self._count += 1
        return vec_id

    def get(self, vec_id: int) -> np.ndarray:
        return self._vectors[vec_id]

    def get_metadata(self, vec_id: int) -> dict[str, Any]:
        return self._metadata[vec_id]

    def delete(self, vec_id: int) -> None:
        self._deleted.add(vec_id)

    def is_deleted(self, vec_id: int) -> bool:
        return vec_id in self._deleted

    def active_ids(self) -> list[int]:
        return [i for i in range(self._count) if i not in self._deleted]

    def active_vectors(self) -> np.ndarray:
        ids = self.active_ids()
        return self._vectors[ids]

    def flush(self) -> None:
        if isinstance(self._vectors, np.memmap):
            self._vectors.flush()

    def _grow(self) -> None:
        new_capacity = self._capacity * 2
        if self._mmap_path:
            old_data = np.array(self._vectors[: self._count])
            os.unlink(self._mmap_path)
            self._vectors = np.memmap(
                self._mmap_path, dtype=np.float32, mode="w+", shape=(new_capacity, self._dim)
            )
            self._vectors[: self._count] = old_data
        else:
            new_buf = np.empty((new_capacity, self._dim), dtype=np.float32)
            new_buf[: self._count] = self._vectors[: self._count]
            self._vectors = new_buf
        self._capacity = new_capacity
