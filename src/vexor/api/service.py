"""Service layer for managing in-memory VectorDB instances."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from uuid import uuid4
import time

import numpy as np

from vexor.concurrency.locks import ReaderWriterLock
from vexor.db import VectorDB


@dataclass
class ManagedIndex:
    name: str
    db: VectorDB
    rw_lock: ReaderWriterLock
    ingest_queue: queue.Queue
    stop_event: threading.Event
    worker: threading.Thread
    jobs: dict[str, dict]
    jobs_lock: threading.Lock


class IndexRegistry:
    """Thread-safe registry of named VectorDB instances."""

    def __init__(self) -> None:
        self._indexes: dict[str, ManagedIndex] = {}
        self._lock = threading.RLock()

    def create(
        self,
        name: str,
        dim: int,
        index_type: str,
        metric: str,
        wal_path: str | None,
        index_kwargs: dict,
    ) -> VectorDB:
        with self._lock:
            if name in self._indexes:
                raise ValueError(f"Index '{name}' already exists.")
            db = VectorDB(
                dim=dim,
                index_type=index_type,
                metric=metric,
                wal_path=wal_path,
                **index_kwargs,
            )
            self._indexes[name] = self._build_managed(name, db)
            return db

    def load(self, name: str, path: str, wal_path: str | None) -> VectorDB:
        with self._lock:
            if name in self._indexes:
                raise ValueError(f"Index '{name}' already exists.")
            db = VectorDB.load(path=path, wal_path=wal_path)
            self._indexes[name] = self._build_managed(name, db)
            return db

    def get(self, name: str) -> VectorDB:
        return self._get_managed(name).db

    def _get_managed(self, name: str) -> ManagedIndex:
        with self._lock:
            managed = self._indexes.get(name)
            if managed is None:
                raise KeyError(f"Index '{name}' not found.")
            return managed

    def delete(self, name: str) -> None:
        with self._lock:
            managed = self._indexes.get(name)
            if managed is None:
                raise KeyError(f"Index '{name}' not found.")
            managed.stop_event.set()
            managed.ingest_queue.put(None)
            managed.worker.join(timeout=2.0)
            del self._indexes[name]

    def list(self) -> list[ManagedIndex]:
        with self._lock:
            return list(self._indexes.values())

    def train(self, name: str, vectors: np.ndarray, **kwargs) -> None:
        managed = self._get_managed(name)
        with managed.rw_lock.write_lock():
            managed.db.train(vectors, **kwargs)

    def add_vector(self, name: str, vector: np.ndarray, metadata: dict | None = None) -> int:
        managed = self._get_managed(name)
        with managed.rw_lock.write_lock():
            return managed.db.add(vector, metadata=metadata)

    def add_batch(
        self,
        name: str,
        vectors: np.ndarray,
        metadata: list[dict] | None = None,
    ) -> list[int]:
        managed = self._get_managed(name)
        with managed.rw_lock.write_lock():
            return managed.db.add_batch(vectors, metadata=metadata)

    def delete_vector(self, name: str, vec_id: int) -> None:
        managed = self._get_managed(name)
        with managed.rw_lock.write_lock():
            managed.db.delete(vec_id)

    def search(
        self,
        name: str,
        query: np.ndarray,
        k: int,
        filter: dict | None = None,
        **search_kwargs,
    ) -> list[tuple[int, float]]:
        managed = self._get_managed(name)
        with managed.rw_lock.read_lock():
            return managed.db.search(query, k=k, filter=filter, **search_kwargs)

    def search_batch(
        self,
        name: str,
        queries: np.ndarray,
        k: int,
        filter: dict | None = None,
        **search_kwargs,
    ) -> list[list[tuple[int, float]]]:
        managed = self._get_managed(name)
        with managed.rw_lock.read_lock():
            return managed.db.search_batch(queries, k=k, filter=filter, **search_kwargs)

    def save(self, name: str, path: str) -> None:
        managed = self._get_managed(name)
        with managed.rw_lock.write_lock():
            managed.db.save(path)

    def enqueue_ingestion_batch(
        self,
        name: str,
        vectors: np.ndarray,
        metadata: list[dict] | None = None,
    ) -> str:
        managed = self._get_managed(name)
        if metadata is not None and len(metadata) != len(vectors):
            raise ValueError("metadata length must match vectors length.")
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D matrix.")
        if vectors.shape[1] != managed.db.dim:
            raise ValueError(f"Expected vector dim {managed.db.dim}, got {vectors.shape[1]}.")

        job_id = uuid4().hex
        with managed.jobs_lock:
            managed.jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "submitted_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "count": int(len(vectors)),
                "inserted": 0,
                "error": None,
            }
        managed.ingest_queue.put((job_id, vectors, metadata))
        return job_id

    def ingestion_job_status(self, name: str, job_id: str) -> dict:
        managed = self._get_managed(name)
        with managed.jobs_lock:
            info = managed.jobs.get(job_id)
            if info is None:
                raise KeyError(f"Ingestion job '{job_id}' not found.")
            return dict(info)

    def flush_ingestion(self, name: str, timeout_s: float = 5.0) -> bool:
        managed = self._get_managed(name)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if managed.ingest_queue.unfinished_tasks == 0:
                return True
            time.sleep(0.02)
        return managed.ingest_queue.unfinished_tasks == 0

    def _build_managed(self, name: str, db: VectorDB) -> ManagedIndex:
        rw_lock = ReaderWriterLock()
        ingest_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        jobs: dict[str, dict] = {}
        jobs_lock = threading.Lock()
        worker = threading.Thread(
            target=self._ingest_worker,
            args=(db, rw_lock, ingest_queue, stop_event, jobs, jobs_lock),
            daemon=True,
            name=f"vexor-ingest-{name}",
        )
        worker.start()
        return ManagedIndex(
            name=name,
            db=db,
            rw_lock=rw_lock,
            ingest_queue=ingest_queue,
            stop_event=stop_event,
            worker=worker,
            jobs=jobs,
            jobs_lock=jobs_lock,
        )

    @staticmethod
    def _ingest_worker(
        db: VectorDB,
        rw_lock: ReaderWriterLock,
        ingest_queue: queue.Queue,
        stop_event: threading.Event,
        jobs: dict[str, dict],
        jobs_lock: threading.Lock,
    ) -> None:
        while not stop_event.is_set():
            try:
                item = ingest_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                ingest_queue.task_done()
                break

            job_id, vectors, metadata = item
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "running"
                    jobs[job_id]["started_at"] = time.time()
            try:
                with rw_lock.write_lock():
                    ids = db.add_batch(vectors, metadata=metadata)
                with jobs_lock:
                    if job_id in jobs:
                        jobs[job_id]["status"] = "completed"
                        jobs[job_id]["inserted"] = len(ids)
                        jobs[job_id]["finished_at"] = time.time()
            except Exception as exc:  # defensive: keep worker alive
                with jobs_lock:
                    if job_id in jobs:
                        jobs[job_id]["status"] = "failed"
                        jobs[job_id]["error"] = str(exc)
                        jobs[job_id]["finished_at"] = time.time()
            finally:
                ingest_queue.task_done()

    @staticmethod
    def to_array_1d(vector: list[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("Expected a 1D vector.")
        return arr

    @staticmethod
    def to_array_2d(vectors: list[list[float]]) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Expected a 2D matrix.")
        return arr
