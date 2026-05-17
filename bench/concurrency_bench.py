"""
Benchmark 4: HNSW search throughput vs thread count.

Tests two execution models:
  1. Threading (shared process) — GIL limits true parallelism for Python code.
     Shows the baseline and overhead cost of thread switching.
  2. Multiprocessing (independent processes) — bypasses the GIL, achieves
     near-linear scaling with core count for CPU-bound search.

Usage:
    python bench/concurrency_bench.py
"""

from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from vexor.indexes.hnsw import HNSWIndex


N = 10_000
DIM = 64
K = 10
N_QUERIES = 400
THREAD_COUNTS = [1, 2, 4, 8, 16]

rng = np.random.default_rng(0)
vecs = rng.standard_normal((N, DIM)).astype(np.float32)
queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)

# Set by __main__ block; worker threads read this global.
hnsw: HNSWIndex | None = None
worker_hnsw: HNSWIndex | None = None


def _thread_worker(q_slice: np.ndarray, out: list, idx: int) -> None:
    results = []
    for q in q_slice:
        results.extend(hnsw.search(q, k=K))
    out[idx] = results


def _init_worker(vectors: np.ndarray) -> None:
    """Build one HNSW index per worker process."""
    global worker_hnsw
    worker_hnsw = HNSWIndex(DIM, M=16, ef_construction=200, ef_search=50, metric="l2")
    for v in vectors:
        worker_hnsw.add(v)


def _worker_warmup(_: int) -> int:
    # Forces initializer execution for each worker.
    return 1


def _worker_search_batch(q_batch: np.ndarray) -> list:
    return [worker_hnsw.search(q, k=K) for q in q_batch]


if __name__ == "__main__":
    print(f"Dataset: N={N:,}  D={DIM}  K={K}  queries={N_QUERIES}")
    print("Building HNSW index...")
    hnsw = HNSWIndex(DIM, M=16, ef_construction=200, ef_search=50, metric="l2")
    for v in vecs:
        hnsw.add(v)
    print("Done.\n")

    # ── Threading benchmark (GIL-limited) ─────────────────────────────────────
    print(f"{'Threads':>8}  {'QPS':>10}  {'vs serial':>10}")
    print("─" * 35)

    threading_qps: list[float] = []
    serial_qps: float | None = None

    for n_threads in THREAD_COUNTS:
        slices = np.array_split(queries, n_threads)
        out: list = [None] * n_threads
        threads = [
            threading.Thread(target=_thread_worker, args=(slices[i], out, i))
            for i in range(n_threads)
        ]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - t0
        qps = N_QUERIES / elapsed
        threading_qps.append(qps)
        if serial_qps is None:
            serial_qps = qps
        speedup = qps / serial_qps
        print(f"{n_threads:>8}  {qps:>10,.0f}  {speedup:>9.2f}×")

    # ── Multiprocessing benchmark (GIL-free) ──────────────────────────────────
    print()
    print("Multiprocessing (ProcessPoolExecutor):")
    print(f"{'Workers':>8}  {'build s':>10}  {'query QPS':>12}  {'vs serial':>10}")
    print("─" * 52)

    mp_qps: list[float] = []
    mp_build_s: list[float] = []

    for n_workers in THREAD_COUNTS:
        slices = np.array_split(queries, n_workers)
        mp_ctx = mp.get_context("fork") if sys.platform != "win32" else None
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(vecs,),
            mp_context=mp_ctx,
        ) as pool:
            t_build = time.perf_counter()
            list(pool.map(_worker_warmup, range(n_workers)))
            build_elapsed = time.perf_counter() - t_build

            t_query = time.perf_counter()
            list(pool.map(_worker_search_batch, slices))
            query_elapsed = time.perf_counter() - t_query

        qps = N_QUERIES / query_elapsed
        mp_qps.append(qps)
        mp_build_s.append(build_elapsed)
        speedup = qps / (serial_qps or 1.0)
        print(f"{n_workers:>8}  {build_elapsed:>10.2f}  {qps:>12,.0f}  {speedup:>9.2f}×")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(THREAD_COUNTS, threading_qps, "o-", color="tomato",
            linewidth=2, markersize=8, label="Threading (GIL-limited)")
    ax.plot(THREAD_COUNTS, mp_qps, "s-", color="steelblue",
            linewidth=2, markersize=8, label="Multiprocessing query QPS")

    ideal = [serial_qps * n for n in THREAD_COUNTS]
    ax.plot(THREAD_COUNTS, ideal, "--", color="green",
            linewidth=1.5, alpha=0.6, label="Ideal linear scaling")

    ax.set_xlabel("Parallel workers", fontsize=11)
    ax.set_ylabel("QPS (queries per second)", fontsize=11)
    ax.set_title(f"HNSW Search Throughput vs Parallelism  —  N={N:,}, D={DIM}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(THREAD_COUNTS)
    plt.tight_layout()

    os.makedirs("bench/results", exist_ok=True)
    plt.savefig("bench/results/concurrency_bench.png", dpi=150)
    plt.close()
    print("\nSaved → bench/results/concurrency_bench.png")
    print("Build-time-only (s) per worker count:")
    for workers, build_s in zip(THREAD_COUNTS, mp_build_s):
        print(f"  workers={workers:<2d} build={build_s:.2f}s")
