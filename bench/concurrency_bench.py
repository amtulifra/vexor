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


def _thread_worker(q_slice: np.ndarray, out: list, idx: int) -> None:
    results = []
    for q in q_slice:
        results.extend(hnsw.search(q, k=K))
    out[idx] = results


def _search_batch(q_batch: np.ndarray) -> list:
    """Run in a worker process — no shared state, no GIL contention."""
    local_hnsw = HNSWIndex(DIM, M=16, ef_construction=200, ef_search=50, metric="l2")
    for v in vecs:
        local_hnsw.add(v)
    return [local_hnsw.search(q, k=K) for q in q_batch]


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
    print(f"{'Workers':>8}  {'QPS':>10}  {'vs serial':>10}")
    print("─" * 35)

    mp_qps: list[float] = []

    for n_workers in THREAD_COUNTS:
        slices = np.array_split(queries, n_workers)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_search_batch, slices))
        elapsed = time.perf_counter() - t0
        qps = N_QUERIES / elapsed
        mp_qps.append(qps)
        speedup = qps / (serial_qps or 1.0)
        print(f"{n_workers:>8}  {qps:>10,.0f}  {speedup:>9.2f}×")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(THREAD_COUNTS, threading_qps, "o-", color="tomato",
            linewidth=2, markersize=8, label="Threading (GIL-limited)")
    ax.plot(THREAD_COUNTS, mp_qps, "s-", color="steelblue",
            linewidth=2, markersize=8, label="Multiprocessing (GIL-free)")

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
