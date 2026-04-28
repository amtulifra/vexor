"""
Benchmark 1: Recall@10 vs QPS for all index types.

Usage:
    python bench/recall_qps.py

Sweeps each index's accuracy parameter (ef_search, nprobe) to produce a
recall/speed tradeoff curve. Saves bench/results/recall_qps.png.
"""

from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vexor.indexes.flat import FlatIndex
from vexor.indexes.kdtree import KDTreeIndex
from vexor.indexes.hnsw import HNSWIndex
from vexor.indexes.ivf import IVFIndex
from vexor.indexes.ivfpq import IVFPQIndex


N = 5_000
DIM = 64
N_QUERIES = 200
K = 10

rng = np.random.default_rng(0)
vecs = rng.standard_normal((N, DIM)).astype(np.float32)
queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)

print(f"Dataset: N={N:,}  D={DIM}  queries={N_QUERIES}")
print("Building ground truth (flat exact)...")
flat = FlatIndex(metric="l2")
for v in vecs:
    flat.add(v)
ground_truth = [{r[0] for r in flat.search(q, k=K)} for q in queries]


def measure(search_fn, label: str) -> tuple[float, float]:
    hits = total = 0
    t0 = time.perf_counter()
    for q, gt in zip(queries, ground_truth):
        res = {r[0] for r in search_fn(q)}
        hits += len(res & gt)
        total += K
    elapsed = time.perf_counter() - t0
    recall = hits / max(total, 1)
    qps = N_QUERIES / elapsed
    print(f"  {label:32s}  recall={recall:.3f}  QPS={qps:,.0f}")
    return recall, qps


# ── Flat ──────────────────────────────────────────────────────────────────────
print("\n── Flat (exact baseline) ────────────────────────────")
flat_r, flat_q = measure(lambda q: flat.search(q, k=K), "Flat")

# ── KD-Tree ───────────────────────────────────────────────────────────────────
print("\n── KD-Tree ──────────────────────────────────────────")
kd = KDTreeIndex(metric="l2")
for v in vecs:
    kd.add(v)
kd.build()
kd_r, kd_q = measure(lambda q: kd.search(q, k=K), "KD-Tree (exact, low-dim)")

# ── HNSW — sweep ef_search ────────────────────────────────────────────────────
print("\n── HNSW (ef_search sweep) ───────────────────────────")
hnsw = HNSWIndex(DIM, M=16, ef_construction=200, metric="l2")
for v in vecs:
    hnsw.add(v)
hnsw_points: list[tuple[float, float, str]] = []
for ef in [10, 20, 40, 80, 150, 300]:
    r, q = measure(lambda qv, ef=ef: hnsw.search(qv, k=K, ef=ef), f"HNSW ef={ef}")
    hnsw_points.append((r, q, f"ef={ef}"))

# ── IVF — sweep nprobe ────────────────────────────────────────────────────────
print("\n── IVF (nprobe sweep) ───────────────────────────────")
ivf = IVFIndex(nlist=64, metric="l2", online_updates=False)
ivf.train(vecs)
for v in vecs:
    ivf.add(v)
ivf_points: list[tuple[float, float, str]] = []
for np_val in [1, 4, 8, 16, 32, 64]:
    r, q = measure(
        lambda qv, np_val=np_val: ivf.search(qv, k=K, nprobe=np_val, use_adaptive_nprobe=False),
        f"IVF nprobe={np_val}",
    )
    ivf_points.append((r, q, f"nprobe={np_val}"))

# ── IVFPQ — sweep nprobe (M=16 fixed, shows speed/recall tradeoff) ────────────
print("\n── IVFPQ (nprobe sweep, M=16) ───────────────────────")
ivfpq = IVFPQIndex(nlist=32, nprobe=8, M=16, K=256, metric="l2")
ivfpq.train(vecs)
for v in vecs:
    ivfpq.add(v)
pq_points: list[tuple[float, float, str]] = []
for np_val in [1, 2, 4, 8, 16, 32]:
    r, q = measure(
        lambda qv, np_val=np_val: ivfpq.search(qv, k=K, nprobe=np_val),
        f"IVFPQ nprobe={np_val}",
    )
    pq_points.append((r, q, f"nprobe={np_val}"))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

# Baselines
ax.scatter(flat_q, flat_r, s=120, color="black", zorder=5, marker="D")
ax.annotate("Flat (exact)", (flat_q, flat_r), xytext=(8, 4),
            textcoords="offset points", fontsize=8, color="black")
ax.scatter(kd_q, kd_r, s=120, color="dimgray", zorder=5, marker="D")
ax.annotate("KD-Tree", (kd_q, kd_r), xytext=(8, 4),
            textcoords="offset points", fontsize=8, color="dimgray")

# HNSW curve
hnsw_recalls = [p[0] for p in hnsw_points]
hnsw_qps     = [p[1] for p in hnsw_points]
ax.plot(hnsw_qps, hnsw_recalls, "o-", color="steelblue",
        linewidth=2, markersize=7, label="HNSW (ef_search sweep)")
for r, q, lbl in hnsw_points[::2]:
    ax.annotate(lbl, (q, r), xytext=(5, 4), textcoords="offset points",
                fontsize=7, color="steelblue")

# IVF curve
ivf_recalls = [p[0] for p in ivf_points]
ivf_qps     = [p[1] for p in ivf_points]
ax.plot(ivf_qps, ivf_recalls, "s-", color="darkorange",
        linewidth=2, markersize=7, label="IVF (nprobe sweep)")
for r, q, lbl in ivf_points[::2]:
    ax.annotate(lbl, (q, r), xytext=(5, 4), textcoords="offset points",
                fontsize=7, color="darkorange")

# IVFPQ curve
pq_recalls = [p[0] for p in pq_points]
pq_qps     = [p[1] for p in pq_points]
ax.plot(pq_qps, pq_recalls, "^-", color="mediumseagreen",
        linewidth=2, markersize=7, label="IVFPQ M=16 (nprobe sweep)")
for r, q, lbl in pq_points[::2]:
    ax.annotate(lbl, (q, r), xytext=(5, -12), textcoords="offset points",
                fontsize=7, color="mediumseagreen")

ax.axhline(0.95, color="green", linestyle="--", linewidth=1,
           alpha=0.5, label="Recall@10 = 0.95")
ax.set_xscale("log")
ax.set_xlabel("QPS (queries per second, log scale)", fontsize=11)
ax.set_ylabel("Recall@10", fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_title(f"Recall@10 vs QPS  —  N={N:,}, D={DIM}, K={K}", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()

os.makedirs("bench/results", exist_ok=True)
plt.savefig("bench/results/recall_qps.png", dpi=150)
plt.close()
print("\nSaved → bench/results/recall_qps.png")
