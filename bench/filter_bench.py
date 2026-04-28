"""
Benchmark 2: In-graph filter recall vs post-filter recall at varying selectivity.

This is the key correctness proof: Vexor's in-graph filtered HNSW maintains
recall as selectivity drops, while naive post-filtering silently returns fewer
and fewer results.

Usage:
    python bench/filter_bench.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vexor.indexes.hnsw import HNSWIndex
from vexor.indexes.flat import FlatIndex


N = 2_000
DIM = 32
K = 10
N_QUERIES = 50
SELECTIVITIES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]

rng = np.random.default_rng(0)
vecs = rng.standard_normal((N, DIM)).astype(np.float32)
queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)

print(f"Dataset: N={N:,}  D={DIM}  queries={N_QUERIES}")
print(f"{'Selectivity':>12}  {'Vexor in-graph':>16}  {'Post-filter':>12}  {'Matching vecs':>14}")
print("─" * 60)

vexor_recalls: list[float] = []
post_recalls: list[float] = []
actual_k_post: list[float] = []

for sel in SELECTIVITIES:
    n_match = max(1, int(N * sel))
    labels = np.array(["match"] * n_match + ["other"] * (N - n_match))
    rng.shuffle(labels)

    idx_vexor = HNSWIndex(DIM, M=16, ef_construction=200, ef_search=100)
    flat = FlatIndex(metric="cosine")

    for i, v in enumerate(vecs):
        meta = {"label": labels[i]}
        idx_vexor.add(v, meta)
        flat.add(v, meta)

    filt = {"label": "match"}
    v_recall = p_recall = 0.0
    avg_post_k = 0.0

    for q in queries:
        exact = {r[0] for r in flat.search(q, k=K, filter=filt)}
        if not exact:
            continue

        # In-graph filter: pass filter into search, adaptive ef scales automatically
        in_graph = {r[0] for r in idx_vexor.search(q, k=K, filter=filt)}

        # Post-filter: search without filter, then discard non-matching results
        raw = [r[0] for r in idx_vexor.search(q, k=N)]  # search for all
        post = {vid for vid in raw if labels[vid] == "match"}

        v_recall += len(in_graph & exact) / len(exact)
        p_recall += len(post & exact) / len(exact)
        avg_post_k += len(post)

    vexor_recalls.append(v_recall / N_QUERIES)
    post_recalls.append(p_recall / N_QUERIES)
    avg_k = avg_post_k / N_QUERIES
    actual_k_post.append(avg_k)
    print(f"{sel:>11.0%}  {vexor_recalls[-1]:>16.3f}  {post_recalls[-1]:>12.3f}  "
          f"{n_match:>8} ({avg_k:.1f} avg returned)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sel_pcts = [s * 100 for s in SELECTIVITIES]

# Left: recall curves
ax = axes[0]
ax.plot(sel_pcts, vexor_recalls, "o-", color="steelblue",
        linewidth=2.5, markersize=8, label="Vexor in-graph filter")
ax.plot(sel_pcts, post_recalls, "s--", color="tomato",
        linewidth=2, markersize=7, label="Post-filter (naive)")
ax.axhline(0.90, color="green", linestyle=":", linewidth=1.2,
           alpha=0.7, label="Recall = 0.90")
ax.set_xlabel("Filter selectivity (%)", fontsize=11)
ax.set_ylabel("Recall@10", fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_title("Recall@10 vs Selectivity", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)

# Right: average results returned by post-filter
ax2 = axes[1]
ax2.bar(range(len(SELECTIVITIES)), actual_k_post, color="tomato", alpha=0.75,
        label="Post-filter avg results returned")
ax2.axhline(K, color="steelblue", linestyle="--", linewidth=1.5,
            label=f"Target k={K}")
ax2.set_xticks(range(len(SELECTIVITIES)))
ax2.set_xticklabels([f"{s:.0%}" for s in SELECTIVITIES], rotation=30, fontsize=9)
ax2.set_xlabel("Selectivity", fontsize=11)
ax2.set_ylabel(f"Avg results returned (target: {K})", fontsize=11)
ax2.set_title("Post-Filter: Results Returned vs Target k", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(axis="y", alpha=0.3)

fig.suptitle(f"In-Graph Filter vs Post-Filter  —  N={N:,}, D={DIM}, K={K}", fontsize=13)
plt.tight_layout()

os.makedirs("bench/results", exist_ok=True)
plt.savefig("bench/results/filter_bench.png", dpi=150)
plt.close()
print("\nSaved → bench/results/filter_bench.png")
