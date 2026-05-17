"""
Benchmark snapshot load time with vs without memory mapping.

Usage:
    python bench/mmap_startup_bench.py --n 200000 --dim 128
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import resource

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from vexor.db import VectorDB
from vexor.storage.format import load_index


def _rss_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="mmap snapshot load benchmark")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--path", type=str, default="bench/results/mmap_bench.snap")
    args = parser.parse_args()

    os.makedirs("bench/results", exist_ok=True)

    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((args.n, args.dim)).astype(np.float32)

    db = VectorDB(dim=args.dim, index_type="flat", metric="l2")
    db.add_batch(vecs)
    db.save(args.path)

    t0 = time.perf_counter()
    loaded_copy = load_index(args.path, mmap_vectors=False)
    copy_elapsed = time.perf_counter() - t0
    copy_rss = _rss_mb()
    _ = loaded_copy["vectors"][0, 0]

    t0 = time.perf_counter()
    loaded_mmap = load_index(args.path, mmap_vectors=True)
    mmap_elapsed = time.perf_counter() - t0
    mmap_rss = _rss_mb()
    _ = loaded_mmap["vectors"][0, 0]

    print(f"N={args.n:,} D={args.dim}")
    print(f"Load (copy): {copy_elapsed:.4f}s  RSS~{copy_rss:.1f}MB")
    print(f"Load (mmap): {mmap_elapsed:.4f}s  RSS~{mmap_rss:.1f}MB")
    if mmap_elapsed > 0:
        print(f"Startup speedup: {copy_elapsed / mmap_elapsed:.2f}x")


if __name__ == "__main__":
    main()
