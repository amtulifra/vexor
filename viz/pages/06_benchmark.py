"""
Interactive benchmark dashboard — Recall@10 vs QPS for all index types.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from vexor.indexes.flat import FlatIndex
from vexor.indexes.hnsw import HNSWIndex
from vexor.indexes.ivf import IVFIndex
from vexor.indexes.ivfpq import IVFPQIndex

st.set_page_config(layout="wide")
st.title("06 — Interactive Benchmark: Recall@10 vs QPS")

col1, col2, col3 = st.columns(3)
N = col1.slider("Vectors (N)", 500, 10_000, 2_000, step=500, key="bench_n")
DIM = col2.slider("Dimensions", 8, 128, 32, key="bench_dim")
K_bench = col3.slider("k", 1, 20, 10, key="bench_k")
N_QUERIES = st.slider("Query count", 10, 100, 30, key="bench_nq")

include_hnsw = st.checkbox("HNSW", value=True)
include_ivf = st.checkbox("IVF", value=True)
include_ivfpq = st.checkbox("IVFPQ", value=True)

if st.button("Run benchmark", key="bench_run"):
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((N, DIM)).astype(np.float32)
    queries_b = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)

    flat_b = FlatIndex(metric="l2")
    for v in vecs:
        flat_b.add(v)
    gt = [{r[0] for r in flat_b.search(q, k=K_bench)} for q in queries_b]

    def bench_fn(fn, label):
        hits = total = 0
        t0 = time.perf_counter()
        for q, g in zip(queries_b, gt):
            res = {r[0] for r in fn(q)}
            hits += len(res & g)
            total += K_bench
        elapsed = time.perf_counter() - t0
        return label, hits / max(total, 1), N_QUERIES / elapsed

    points = []

    t0 = time.perf_counter()
    gt_list = [{r[0] for r in flat_b.search(q, k=K_bench)} for q in queries_b]
    flat_qps = N_QUERIES / (time.perf_counter() - t0)
    points.append(("Flat (exact)", 1.0, flat_qps, "gray"))

    if include_hnsw:
        with st.spinner("Building HNSW..."):
            hnsw_b = HNSWIndex(DIM, M=16, ef_construction=200, ef_search=50, metric="l2")
            for v in vecs:
                hnsw_b.add(v)
        for ef in [10, 25, 50, 100]:
            lbl, rec, qps = bench_fn(lambda q, ef=ef: hnsw_b.search(q, k=K_bench, ef=ef), f"HNSW ef={ef}")
            points.append((lbl, rec, qps, "steelblue"))

    if include_ivf and N >= 200:
        with st.spinner("Training IVF..."):
            nlist = min(64, N // 8)
            ivf_b = IVFIndex(nlist=nlist, metric="l2", online_updates=False)
            ivf_b.train(vecs)
            for v in vecs:
                ivf_b.add(v)
        for np_val in [1, 4, 8, 16]:
            lbl, rec, qps = bench_fn(
                lambda q, np_val=np_val: ivf_b.search(q, k=K_bench, nprobe=np_val, use_adaptive_nprobe=False),
                f"IVF nprobe={np_val}")
            points.append((lbl, rec, qps, "orange"))

    if include_ivfpq and N >= 300:
        M_vals = [m for m in [4, 8, 16] if DIM % m == 0]
        for m_val in M_vals:
            with st.spinner(f"Training IVFPQ M={m_val}..."):
                nlist_pq = min(32, N // 10)
                idx_pq = IVFPQIndex(nlist=nlist_pq, nprobe=min(8, nlist_pq), M=m_val, K=min(256, N // 4), metric="l2")
                idx_pq.train(vecs)
                for v in vecs:
                    idx_pq.add(v)
            lbl, rec, qps = bench_fn(lambda q, idx=idx_pq: idx.search(q, k=K_bench), f"IVFPQ M={m_val}")
            points.append((lbl, rec, qps, "green"))

    st.session_state["bench_points"] = points

if "bench_points" in st.session_state:
    points = st.session_state["bench_points"]
    fig = go.Figure()
    color_map = {"gray": "gray", "steelblue": "steelblue", "orange": "darkorange", "green": "seagreen"}
    for label, recall, qps, color in points:
        fig.add_trace(go.Scatter(
            x=[qps], y=[recall], mode="markers+text",
            marker=dict(size=12, color=color),
            text=[label], textposition="top center",
            name=label,
        ))
    fig.add_hline(y=0.95, line_dash="dash", line_color="green", annotation_text="0.95 recall")
    fig.update_layout(
        xaxis_title="QPS (queries per second)",
        yaxis_title="Recall@10",
        yaxis_range=[0, 1.1],
        height=550,
        title="Recall@10 vs QPS — All Index Types",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Results table")
        import pandas as pd
        df = pd.DataFrame([(l, f"{r:.3f}", f"{q:,.0f}") for l, r, q, _ in sorted(points, key=lambda x: -x[2])],
                           columns=["Index", "Recall@10", "QPS"])
        st.dataframe(df, use_container_width=True)
