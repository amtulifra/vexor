"""Per-node RW lock registry for concurrent HNSW inserts."""

from __future__ import annotations
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator


class NodeLockRegistry:
    """
    Lazily-created per-node reentrant locks.

    Writers acquire the lock for the specific nodes they modify (new node +
    its neighbors). Readers traverse without locking — read-only graph
    traversal is safe for concurrent access since we never remove edges in
    place during a live search.
    """

    def __init__(self) -> None:
        self._locks: dict[int, threading.RLock] = defaultdict(threading.RLock)
        self._registry_lock = threading.Lock()

    def lock_for(self, node_id: int) -> threading.RLock:
        with self._registry_lock:
            return self._locks[node_id]

    @contextmanager
    def write_many(self, *node_ids: int) -> Iterator[None]:
        sorted_ids = sorted(set(node_ids))
        locks = [self.lock_for(nid) for nid in sorted_ids]
        for lk in locks:
            lk.acquire()
        try:
            yield
        finally:
            for lk in reversed(locks):
                lk.release()
