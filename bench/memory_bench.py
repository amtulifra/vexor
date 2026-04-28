"""
Benchmark 3: Memory footprint vs recall for each compression method.

Compares float32 exact storage, HNSW+SQ (int8), and IVFPQ at various M
values. Shows the recall cost of each compression level.

Usage:
    python bench/memory_bench.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vexor.indexes.flat import FlatIndex
from vexor.indexes.hnsw import HNSWIndex
from vexor.indexes.ivfpq import IVFPQIndex


N = 8_000
DIM = 128
K = 10
N_QUERIES = 100

rng = np.random.default_rng(0)
vecs = rng.standard_normal((N, DIM)).astype(np.float32)
queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)

print(f"Dataset: N={N:,}  D={DIM}  K={K}")
print("Building ground truth (flat exact)...")
flat = FlatIndex(metric="l2")
for v in vecs:
    flat.add(v)
ground_truth = [{r[0] for r in flat.search(q, k=K)} for q in queries]


def recall(results_list: list[set], gts: list[set]) -> float:
    hits = sum(len(r & g) for r, g in zip(results_list, gts))
    return hits / (len(gts) * K)


print(f"\n{'Method':<22}  {'Bytes/vec':>10}  {'Compression':>12}  {'Recall@10':>10}")
print("─" * 62)

points: list[tuple[str, int, float, str]] = []

# ── Flat: DIM × 4 bytes ───────────────────────────────────────────────────────
flat_bpv = DIM * 4
flat_rec = recall([{r[0] for r in flat.search(q, k=K)} for q in queries], ground_truth)
label = "Flat (float32)"
print(f"{label:<22}  {flat_bpv:>10,}  {'1×':>12}  {flat_rec:>10.3f}")
points.append((label, flat_bpv, flat_rec, "black"))

# ── HNSW + SQ: DIM × 1 byte (int8 traversal, float32 final rerank) ───────────
hnsw_sq = HNSWIndex(DIM, M=16, ef_construction=200, ef_search=200, metric="l2", sq=True)
for v in vecs:
    hnsw_sq.add(v)
sq_rec = recall([{r[0] for r in hnsw_sq.search(q, k=K)} for q in queries], ground_truth)
sq_bpv = DIM  # int8 traversal vectors
label = "HNSW + SQ (int8)"
print(f"{label:<22}  {sq_bpv:>10,}  {f'{flat_bpv//sq_bpv}×':>12}  {sq_rec:>10.3f}")
points.append((label, sq_bpv, sq_rec, "mediumpurple"))

# ── IVFPQ at various M: M bytes/vector ────────────────────────────────────────
# Use nprobe = nlist // 2 (50% coverage) so the IVF component is not the
# bottleneck — this isolates the recall impact of PQ compression quality.
nlist = 16
nprobe = 16  # 100% cluster coverage — isolates PQ compression error from IVF recall
M_vals = [m for m in [4, 8, 16, 32, 64] if DIM % m == 0]

for m_val in M_vals:
    idx_pq = IVFPQIndex(nlist=nlist, nprobe=nprobe, M=m_val, K=256, metric="l2")
    idx_pq.train(vecs)
    for v in vecs:
        idx_pq.add(v)
    pq_rec = recall([{r[0] for r in idx_pq.search(q, k=K)} for q in queries], ground_truth)
    label = f"IVFPQ M={m_val}"
    compression = f"{flat_bpv // m_val}×"
    print(f"{label:<22}  {m_val:>10,}  {compression:>12}  {pq_rec:>10.3f}")
    points.append((label, m_val, pq_rec, "steelblue"))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

colors = {"black": "black", "mediumpurple": "mediumpurple", "steelblue": "steelblue"}
plotted_labels: set[str] = set()

for label, bpv, rec, color in sorted(points, key=lambda x: x[1]):
    group = color
    lbl_with_group = group if group not in plotted_labels else "_nolegend_"
    group_label = {
        "black": "Flat (exact)",
        "mediumpurple": "HNSW + SQ",
        "steelblue": "IVFPQ (varying M)",
    }.get(color, label)
    ax.scatter(bpv, rec, s=120, color=color, zorder=5,
               label=group_label if color not in plotted_labels else "_nolegend_")
    plotted_labels.add(color)
    offset = (8, 6) if bpv > 100 else (8, -14)
    ax.annotate(label, (bpv, rec), xytext=offset,
                textcoords="offset points", fontsize=8.5)

# Connect IVFPQ points with a line to show the tradeoff curve
pq_pts = sorted([(bpv, rec) for lbl, bpv, rec, c in points if c == "steelblue"],
                key=lambda x: x[0])
if pq_pts:
    ax.plot([p[0] for p in pq_pts], [p[1] for p in pq_pts],
            "-", color="steelblue", linewidth=1.5, alpha=0.5)

ax.axhline(0.90, color="green", linestyle="--", linewidth=1,
           alpha=0.6, label="Recall@10 = 0.90")
ax.set_xscale("log")
ax.set_xlabel("Bytes per vector (log scale)", fontsize=11)
ax.set_ylabel("Recall@10", fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title(f"Memory Footprint vs Recall  —  N={N:,}, D={DIM}, K={K}", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()

os.makedirs("bench/results", exist_ok=True)
plt.savefig("bench/results/memory_bench.png", dpi=150)
plt.close()
print("\nSaved → bench/results/memory_bench.png")
