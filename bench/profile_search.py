"""
Profile Vexor search with cProfile.

Usage:
  python bench/profile_search.py --index hnsw --n 50000 --dim 128 --queries 1000 --k 10
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from vexor.db import VectorDB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile Vexor search path")
    p.add_argument("--index", choices=["flat", "hnsw", "ivf", "ivfpq", "lsh"], default="hnsw")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile-out", type=str, default="bench/results/search_profile.prof")
    p.add_argument("--stats-out", type=str, default="bench/results/search_profile.txt")
    return p.parse_args()


def build_db(index_type: str, vecs: np.ndarray) -> VectorDB:
    dim = vecs.shape[1]
    if index_type == "hnsw":
        db = VectorDB(dim=dim, index_type="hnsw", metric="l2", M=16, ef_construction=200, ef_search=80)
    elif index_type == "ivf":
        db = VectorDB(dim=dim, index_type="ivf", metric="l2", nlist=max(8, int(np.sqrt(len(vecs)))))
    elif index_type == "ivfpq":
        m = 8 if dim % 8 == 0 else 4
        db = VectorDB(dim=dim, index_type="ivfpq", metric="l2", nlist=max(8, int(np.sqrt(len(vecs)))), M=m, K=256)
    elif index_type == "lsh":
        db = VectorDB(dim=dim, index_type="lsh", metric="l2", n_tables=10, n_hyperplanes=12)
    else:
        db = VectorDB(dim=dim, index_type="flat", metric="l2")

    if index_type in {"ivf", "ivfpq"}:
        db.train(vecs)
    db.add_batch(vecs)
    return db


def main() -> None:
    args = parse_args()
    os.makedirs("bench/results", exist_ok=True)

    rng = np.random.default_rng(args.seed)
    vecs = rng.standard_normal((args.n, args.dim)).astype(np.float32)
    queries = rng.standard_normal((args.queries, args.dim)).astype(np.float32)

    db = build_db(args.index, vecs)

    prof = cProfile.Profile()
    prof.enable()
    t0 = time.perf_counter()
    for q in queries:
        db.search(q, k=args.k)
    elapsed = time.perf_counter() - t0
    prof.disable()

    prof.dump_stats(args.profile_out)
    with open(args.stats_out, "w", encoding="utf-8") as f:
        stats = pstats.Stats(prof, stream=f).sort_stats("cumtime")
        stats.print_stats(60)

    qps = len(queries) / max(elapsed, 1e-12)
    print(f"index={args.index} n={args.n} d={args.dim} queries={args.queries} k={args.k}")
    print(f"elapsed={elapsed:.3f}s qps={qps:.0f}")
    print(f"profile saved -> {args.profile_out}")
    print(f"stats saved   -> {args.stats_out}")
    print("Tip: run `python -m pstats bench/results/search_profile.prof` for interactive drill-down.")


if __name__ == "__main__":
    main()
