"""
Cross-engine ANN benchmark harness.

Compares Vexor against FAISS, hnswlib, and Annoy when available.

Metrics:
  - Recall@K (vs exact flat ground truth)
  - QPS
  - P50/P95 latency (ms)
  - Build time (s)
  - Build throughput (vectors/s)
  - Approx. RAM delta (MB)

Usage examples:
  python bench/real_benchmark.py --dataset synthetic --n 20000 --dim 128
  python bench/real_benchmark.py --dataset sift --sift-base path/to/sift_base.fvecs --sift-query path/to/sift_query.fvecs
  python bench/real_benchmark.py --dataset glove --glove path/to/glove.txt --dim 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from vexor.db import VectorDB

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    import hnswlib
except Exception:  # pragma: no cover - optional dependency
    hnswlib = None

try:
    from annoy import AnnoyIndex
except Exception:  # pragma: no cover - optional dependency
    AnnoyIndex = None


@dataclass
class Dataset:
    base: np.ndarray
    query: np.ndarray
    name: str


@dataclass
class BenchResult:
    engine: str
    recall_at_k: float
    qps: float
    p50_ms: float
    p95_ms: float
    build_s: float
    build_throughput: float
    ram_delta_mb: float
    notes: str = ""


def _rss_mb() -> float:
    if psutil is None:
        return 0.0
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)


def _latency_percentiles_ms(latencies_s: list[float]) -> tuple[float, float]:
    arr = np.asarray(latencies_s, dtype=np.float64)
    return float(np.percentile(arr, 50) * 1000.0), float(np.percentile(arr, 95) * 1000.0)


def _recall_at_k(results: list[list[int]], gt: list[set[int]], k: int) -> float:
    hits = 0
    total = len(gt) * k
    for got, exp in zip(results, gt):
        hits += len(set(got) & exp)
    return hits / max(total, 1)


def _ground_truth(base: np.ndarray, queries: np.ndarray, k: int) -> list[set[int]]:
    db = VectorDB(dim=base.shape[1], index_type="flat", metric="l2")
    db.add_batch(base)
    return [{vid for vid, _ in db.search(q, k=k)} for q in queries]


def _load_fvecs(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")
    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"Invalid fvecs dimension in {path}")
    vec_size = dim + 1
    if raw.size % vec_size != 0:
        raise ValueError(f"Malformed fvecs file: {path}")
    mat = raw.reshape(-1, vec_size)
    if not np.all(mat[:, 0] == dim):
        raise ValueError(f"Inconsistent vector dimensions in {path}")
    return mat[:, 1:].view(np.float32)


def _load_glove(path: str, dim: int, limit: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != dim + 1:
                continue
            vals = np.asarray(parts[1:], dtype=np.float32)
            vectors.append(vals)
            if len(vectors) >= limit:
                break
    if not vectors:
        raise ValueError(f"No vectors read from GloVe file: {path}")
    return np.vstack(vectors).astype(np.float32)


def load_dataset(args: argparse.Namespace) -> Dataset:
    if args.dataset == "synthetic":
        rng = np.random.default_rng(args.seed)
        base = rng.standard_normal((args.n, args.dim)).astype(np.float32)
        query = rng.standard_normal((args.queries, args.dim)).astype(np.float32)
        return Dataset(base=base, query=query, name=f"synthetic_n{args.n}_d{args.dim}")

    if args.dataset == "sift":
        if not args.sift_base or not args.sift_query:
            raise ValueError("SIFT dataset requires --sift-base and --sift-query.")
        base = _load_fvecs(args.sift_base)
        query = _load_fvecs(args.sift_query)
        base = base[: args.n]
        query = query[: args.queries]
        return Dataset(base=base, query=query, name="sift")

    if args.dataset == "glove":
        if not args.glove:
            raise ValueError("GloVe dataset requires --glove path.")
        all_vecs = _load_glove(args.glove, dim=args.dim, limit=args.n + args.queries)
        if len(all_vecs) <= args.queries:
            raise ValueError("Not enough vectors in glove file for requested n + queries.")
        base = all_vecs[: args.n]
        query = all_vecs[args.n : args.n + args.queries]
        return Dataset(base=base, query=query, name="glove")

    raise ValueError(f"Unknown dataset: {args.dataset}")


def benchmark_vexor_hnsw(data: Dataset, k: int) -> BenchResult:
    before = _rss_mb()
    t0 = time.perf_counter()
    db = VectorDB(dim=data.base.shape[1], index_type="hnsw", metric="l2", M=16, ef_construction=200, ef_search=80)
    db.add_batch(data.base)
    build_s = time.perf_counter() - t0
    after = _rss_mb()

    latencies: list[float] = []
    ids: list[list[int]] = []
    for q in data.query:
        q0 = time.perf_counter()
        hits = db.search(q, k=k)
        latencies.append(time.perf_counter() - q0)
        ids.append([vid for vid, _ in hits])

    p50, p95 = _latency_percentiles_ms(latencies)
    total_s = sum(latencies)
    gt = _ground_truth(data.base, data.query, k)
    return BenchResult(
        engine="vexor_hnsw",
        recall_at_k=_recall_at_k(ids, gt, k),
        qps=len(data.query) / max(total_s, 1e-12),
        p50_ms=p50,
        p95_ms=p95,
        build_s=build_s,
        build_throughput=len(data.base) / max(build_s, 1e-12),
        ram_delta_mb=max(after - before, 0.0),
    )


def benchmark_faiss_flat(data: Dataset, k: int) -> BenchResult | None:
    if faiss is None:
        return None
    before = _rss_mb()
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(data.base.shape[1])
    index.add(data.base)
    build_s = time.perf_counter() - t0
    after = _rss_mb()

    latencies: list[float] = []
    ids: list[list[int]] = []
    for q in data.query:
        q0 = time.perf_counter()
        _, idx = index.search(q.reshape(1, -1), k)
        latencies.append(time.perf_counter() - q0)
        ids.append([int(x) for x in idx[0]])

    p50, p95 = _latency_percentiles_ms(latencies)
    total_s = sum(latencies)
    gt = _ground_truth(data.base, data.query, k)
    return BenchResult(
        engine="faiss_flat",
        recall_at_k=_recall_at_k(ids, gt, k),
        qps=len(data.query) / max(total_s, 1e-12),
        p50_ms=p50,
        p95_ms=p95,
        build_s=build_s,
        build_throughput=len(data.base) / max(build_s, 1e-12),
        ram_delta_mb=max(after - before, 0.0),
    )


def benchmark_hnswlib(data: Dataset, k: int) -> BenchResult | None:
    if hnswlib is None:
        return None
    before = _rss_mb()
    t0 = time.perf_counter()
    index = hnswlib.Index(space="l2", dim=data.base.shape[1])
    index.init_index(max_elements=len(data.base), ef_construction=200, M=16)
    index.add_items(data.base, np.arange(len(data.base)))
    index.set_ef(max(80, k))
    build_s = time.perf_counter() - t0
    after = _rss_mb()

    latencies: list[float] = []
    ids: list[list[int]] = []
    for q in data.query:
        q0 = time.perf_counter()
        labels, _ = index.knn_query(q.reshape(1, -1), k=k)
        latencies.append(time.perf_counter() - q0)
        ids.append([int(x) for x in labels[0]])

    p50, p95 = _latency_percentiles_ms(latencies)
    total_s = sum(latencies)
    gt = _ground_truth(data.base, data.query, k)
    return BenchResult(
        engine="hnswlib",
        recall_at_k=_recall_at_k(ids, gt, k),
        qps=len(data.query) / max(total_s, 1e-12),
        p50_ms=p50,
        p95_ms=p95,
        build_s=build_s,
        build_throughput=len(data.base) / max(build_s, 1e-12),
        ram_delta_mb=max(after - before, 0.0),
    )


def benchmark_annoy(data: Dataset, k: int, trees: int = 20) -> BenchResult | None:
    if AnnoyIndex is None:
        return None
    before = _rss_mb()
    t0 = time.perf_counter()
    index = AnnoyIndex(data.base.shape[1], metric="euclidean")
    for i, v in enumerate(data.base):
        index.add_item(i, v.tolist())
    index.build(trees)
    build_s = time.perf_counter() - t0
    after = _rss_mb()

    latencies: list[float] = []
    ids: list[list[int]] = []
    for q in data.query:
        q0 = time.perf_counter()
        idx = index.get_nns_by_vector(q.tolist(), k)
        latencies.append(time.perf_counter() - q0)
        ids.append([int(x) for x in idx])

    p50, p95 = _latency_percentiles_ms(latencies)
    total_s = sum(latencies)
    gt = _ground_truth(data.base, data.query, k)
    return BenchResult(
        engine="annoy",
        recall_at_k=_recall_at_k(ids, gt, k),
        qps=len(data.query) / max(total_s, 1e-12),
        p50_ms=p50,
        p95_ms=p95,
        build_s=build_s,
        build_throughput=len(data.base) / max(build_s, 1e-12),
        ram_delta_mb=max(after - before, 0.0),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-engine ANN benchmark harness")
    p.add_argument("--dataset", choices=["synthetic", "sift", "glove"], default="synthetic")
    p.add_argument("--n", type=int, default=10000, help="Base vectors")
    p.add_argument("--dim", type=int, default=128, help="Vector dimension")
    p.add_argument("--queries", type=int, default=200, help="Number of query vectors")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sift-base", type=str, default="")
    p.add_argument("--sift-query", type=str, default="")
    p.add_argument("--glove", type=str, default="")
    p.add_argument("--out", type=str, default="bench/results/real_benchmark_results.json")
    return p.parse_args()


def _to_json(rows: list[BenchResult], dataset_name: str, k: int) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "k": k,
        "generated_at": time.time(),
        "results": [
            {
                "engine": r.engine,
                "recall_at_k": r.recall_at_k,
                "qps": r.qps,
                "p50_ms": r.p50_ms,
                "p95_ms": r.p95_ms,
                "build_s": r.build_s,
                "build_throughput": r.build_throughput,
                "ram_delta_mb": r.ram_delta_mb,
                "notes": r.notes,
            }
            for r in rows
        ],
    }


def _print_table(rows: list[BenchResult]) -> None:
    print(
        f"{'engine':14s} {'recall@k':>9s} {'qps':>10s} {'p50ms':>9s} "
        f"{'p95ms':>9s} {'build_s':>9s} {'vec/s':>11s} {'ram_mb':>9s}"
    )
    print("-" * 92)
    for r in rows:
        print(
            f"{r.engine:14s} {r.recall_at_k:9.3f} {r.qps:10.0f} {r.p50_ms:9.3f} "
            f"{r.p95_ms:9.3f} {r.build_s:9.3f} {r.build_throughput:11.0f} {r.ram_delta_mb:9.1f}"
        )


def main() -> None:
    args = parse_args()
    os.makedirs("bench/results", exist_ok=True)
    data = load_dataset(args)
    if data.base.shape[1] != args.dim:
        print(f"[warn] dataset dimension {data.base.shape[1]} overrides --dim={args.dim}")

    runners: list[Callable[[Dataset, int], BenchResult | None]] = [
        benchmark_vexor_hnsw,
        benchmark_faiss_flat,
        benchmark_hnswlib,
        benchmark_annoy,
    ]
    rows: list[BenchResult] = []
    for run in runners:
        try:
            out = run(data, args.k)
        except Exception as exc:
            name = run.__name__.replace("benchmark_", "")
            rows.append(
                BenchResult(
                    engine=name,
                    recall_at_k=0.0,
                    qps=0.0,
                    p50_ms=0.0,
                    p95_ms=0.0,
                    build_s=0.0,
                    build_throughput=0.0,
                    ram_delta_mb=0.0,
                    notes=f"error: {exc}",
                )
            )
            continue
        if out is not None:
            rows.append(out)

    _print_table(rows)
    payload = _to_json(rows, dataset_name=data.name, k=args.k)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results -> {out_path}")


if __name__ == "__main__":
    main()
