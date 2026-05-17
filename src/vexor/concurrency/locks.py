"""Lock primitives for concurrent indexing and search."""

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


class ReaderWriterLock:
    """
    Writer-preferring reader-writer lock.

    - Multiple readers can hold the lock concurrently.
    - Writers acquire exclusive access.
    - New readers wait while a writer is waiting, preventing writer starvation.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._active_readers = 0
        self._active_writer = False
        self._waiting_writers = 0

    @contextmanager
    def read_lock(self) -> Iterator[None]:
        with self._cond:
            while self._active_writer or self._waiting_writers > 0:
                self._cond.wait()
            self._active_readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._active_readers -= 1
                if self._active_readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        with self._cond:
            self._waiting_writers += 1
            while self._active_writer or self._active_readers > 0:
                self._cond.wait()
            self._waiting_writers -= 1
            self._active_writer = True
        try:
            yield
        finally:
            with self._cond:
                self._active_writer = False
                self._cond.notify_all()
