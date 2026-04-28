"""
Benchmark 5: Recall@10 vs latency percentiles (p50 / p95 / p99).

QPS is a throughput number — it hides the tail. A system with great p50
latency can still miss SLAs if p99 is 10× higher. This benchmark sweeps
each index's accuracy parameter and records the full per-query latency
distribution, then plots recall vs p50/p95/p99.

Usage:
    python bench/latency_percentile_bench.py
"""

from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vexor.indexes.flat import FlatIndex
from vexor.indexes.hnsw import HNSWIndex
from vexor.indexes.ivf import IVFIndex


N = 5_000
DIM = 64
K = 10
N_QUERIES = 500   # more queries → stable high-percentile estimates

rng = np.random.default_rng(0)
vecs = rng.standard_normal((N, DIM)).astype(np.float32)
queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)

print(f"Dataset: N={N:,}  D={DIM}  K={K}  queries={N_QUERIES}")

# ── Ground truth ───────────────────────────────────────────────────────────────
print("Building ground truth (flat exact)...")
flat = FlatIndex(metric="l2")
for v in vecs:
    flat.add(v)
ground_truth = [{r[0] for r in flat.search(q, k=K)} for q in queries]
print("Done.\n")


def profile(search_fn, label: str) -> dict:
    """Run all queries individually and return recall + latency percentiles."""
    latencies_ms: list[float] = []
    hits = 0
    for q, gt in zip(queries, ground_truth):
        t0 = time.perf_counter()
        res = search_fn(q)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        hits += len({r[0] for r in res} & gt)

    lat = np.array(latencies_ms)
    recall = hits / (N_QUERIES * K)
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    print(f"  {label:38s}  recall={recall:.3f}  "
          f"p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms")
    return dict(recall=recall, p50=p50, p95=p95, p99=p99)


# ── Flat baseline ──────────────────────────────────────────────────────────────
print("── Flat (exact baseline) ────────────────────────────")
flat_prof = profile(lambda q: flat.search(q, k=K), "Flat")

# ── HNSW — ef_search sweep ────────────────────────────────────────────────────
print("\n── HNSW (ef_search sweep) ───────────────────────────")
hnsw = HNSWIndex(DIM, M=16, ef_construction=200, metric="l2")
for v in vecs:
    hnsw.add(v)

hnsw_pts: list[dict] = []
for ef in [10, 20, 40, 80, 150, 300]:
    pt = profile(lambda q, ef=ef: hnsw.search(q, k=K, ef=ef), f"HNSW ef={ef}")
    pt["label"] = f"ef={ef}"
    hnsw_pts.append(pt)

# ── IVF — nprobe sweep ────────────────────────────────────────────────────────
print("\n── IVF (nprobe sweep) ───────────────────────────────")
ivf = IVFIndex(nlist=64, metric="l2", online_updates=False)
ivf.train(vecs)
for v in vecs:
    ivf.add(v)

ivf_pts: list[dict] = []
for nprobe in [1, 4, 8, 16, 32, 64]:
    pt = profile(
        lambda q, np=nprobe: ivf.search(q, k=K, nprobe=np, use_adaptive_nprobe=False),
        f"IVF nprobe={nprobe}",
    )
    pt["label"] = f"nprobe={nprobe}"
    ivf_pts.append(pt)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

PERCENTILES = [("p50", "o-", 2.0), ("p95", "s--", 1.5), ("p99", "^:", 1.5)]

# HNSW curves
hnsw_recalls = [p["recall"] for p in hnsw_pts]
for key, style, lw in PERCENTILES:
    lats = [p[key] for p in hnsw_pts]
    ax.plot(hnsw_recalls, lats, style, color="steelblue", linewidth=lw,
            markersize=7, label=f"HNSW {key}")

# Annotate ef labels at p50 points
for pt in hnsw_pts[::2]:
    ax.annotate(pt["label"], (pt["recall"], pt["p50"]),
                xytext=(4, 5), textcoords="offset points",
                fontsize=7, color="steelblue")

# IVF curves
ivf_recalls = [p["recall"] for p in ivf_pts]
for key, style, lw in PERCENTILES:
    lats = [p[key] for p in ivf_pts]
    ax.plot(ivf_recalls, lats, style, color="darkorange", linewidth=lw,
            markersize=7, label=f"IVF {key}")

for pt in ivf_pts[::2]:
    ax.annotate(pt["label"], (pt["recall"], pt["p50"]),
                xytext=(4, -12), textcoords="offset points",
                fontsize=7, color="darkorange")

# Flat baseline
ax.axhline(flat_prof["p50"], color="black", linestyle="-",
           linewidth=1.2, alpha=0.5, label=f"Flat p50 ({flat_prof['p50']:.1f}ms)")
ax.axhline(flat_prof["p99"], color="black", linestyle=":",
           linewidth=1.2, alpha=0.5, label=f"Flat p99 ({flat_prof['p99']:.1f}ms)")

ax.set_xlabel("Recall@10", fontsize=11)
ax.set_ylabel("Latency per query (ms, log scale)", fontsize=11)
ax.set_yscale("log")
ax.set_xlim(0, 1.05)
ax.set_title(
    f"Recall@10 vs Query Latency Percentiles  —  N={N:,}, D={DIM}, K={K}",
    fontsize=13,
)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()

os.makedirs("bench/results", exist_ok=True)
plt.savefig("bench/results/latency_percentile_bench.png", dpi=150)
plt.close()
print("\nSaved → bench/results/latency_percentile_bench.png")
