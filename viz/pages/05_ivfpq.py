"""
IVFPQ visualizer — codebook heatmap + compression/recall tradeoff.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from vexor.indexes.ivfpq import IVFPQIndex
from vexor.indexes.flat import FlatIndex
from vexor.quantization.pq import ProductQuantizer

st.set_page_config(layout="wide")
st.title("05 — IVFPQ: Product Quantization + Compression/Recall Tradeoff")

tab_codebook, tab_tradeoff = st.tabs(["Codebook Viewer", "Compression vs Recall"])


with tab_codebook:
    st.subheader("PQ Codebook Heatmap")
    col1, col2, col3 = st.columns(3)
    M_cb = col1.slider("M (subspaces)", 2, 16, 4, key="cb_M")
    K_cb = col2.slider("K (centroids/subspace)", 8, 64, 16, key="cb_K")
    n_cb = col3.slider("Training vectors", 100, 1000, 300, key="cb_n")

    if st.button("Train PQ codebook", key="cb_train"):
        rng = np.random.default_rng(7)
        dim_cb = M_cb * 4
        vecs_cb = rng.standard_normal((n_cb, dim_cb)).astype(np.float32)
        pq = ProductQuantizer(M=M_cb, K=K_cb)
        pq.train(vecs_cb)
        codes_cb = pq.encode(vecs_cb)
        st.session_state.update({
            "cb_pq": pq, "cb_codes": codes_cb,
            "cb_vecs": vecs_cb, "cb_dim": dim_cb,
        })

    if "cb_pq" in st.session_state:
        pq = st.session_state["cb_pq"]
        codes = st.session_state["cb_codes"]
        vecs_cb = st.session_state["cb_vecs"]

        # Centroid usage heatmap
        usage = np.zeros((M_cb, K_cb), dtype=np.int32)
        for m in range(M_cb):
            for c_id in codes[:, m]:
                usage[m, c_id] += 1

        fig_heat = px.imshow(
            usage,
            labels={"x": "Centroid ID", "y": "Subspace", "color": "# vectors"},
            color_continuous_scale="Blues",
            title="Centroid usage per subspace (overloaded = recall loss risk)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Quantization error
        decoded = pq.decode(codes)
        errors = np.linalg.norm(vecs_cb - decoded, axis=1)
        fig_err = go.Figure(go.Histogram(x=errors, nbinsx=40, marker_color="steelblue"))
        fig_err.update_layout(
            xaxis_title="Reconstruction error (L2)",
            yaxis_title="Count",
            title="PQ reconstruction error distribution",
            height=300,
        )
        st.plotly_chart(fig_err, use_container_width=True)

        # Compression table
        D = st.session_state["cb_dim"]
        float32_bytes = D * 4
        pq_bytes = M_cb
        st.write(f"**Compression:** {float32_bytes} bytes/vector → {pq_bytes} bytes/vector "
                 f"(**{float32_bytes // pq_bytes}× reduction**)")


with tab_tradeoff:
    st.subheader("Recall vs bytes/vector across M values")

    col1, col2 = st.columns(2)
    n_trade = col1.slider("Vectors", 300, 2000, 500, key="tr_n")
    dim_trade = col2.slider("Dimension", 32, 256, 64, key="tr_dim")
    k_trade = st.slider("k", 1, 20, 10, key="tr_k")

    M_values = [4, 8, 16, 32, 48, 64]
    available_M = [m for m in M_values if dim_trade % m == 0]

    if st.button("Run recall vs compression benchmark", key="tr_run"):
        rng = np.random.default_rng(0)
        vecs_t = rng.standard_normal((n_trade, dim_trade)).astype(np.float32)
        queries_t = rng.standard_normal((20, dim_trade)).astype(np.float32)

        flat_t = FlatIndex(metric="l2")
        for v in vecs_t:
            flat_t.add(v)

        recalls, bytes_per_vec = [], []
        progress = st.progress(0)

        for step_i, m_val in enumerate(available_M):
            nlist_t = min(32, n_trade // 10)
            idx_pq = IVFPQIndex(nlist=nlist_t, nprobe=min(8, nlist_t), M=m_val, K=256, metric="l2")
            idx_pq.train(vecs_t)
            for v in vecs_t:
                idx_pq.add(v)

            total = hits = 0
            for q in queries_t:
                exact = {r[0] for r in flat_t.search(q, k=k_trade)}
                approx = {r[0] for r in idx_pq.search(q, k=k_trade)}
                hits += len(exact & approx)
                total += k_trade
            recalls.append(hits / max(total, 1))
            bytes_per_vec.append(m_val)
            progress.progress((step_i + 1) / len(available_M))

        float_bytes = dim_trade * 4
        exact_recall = 1.0
        bytes_per_vec.append(float_bytes)
        recalls.append(exact_recall)

        st.session_state["tr_bytes"] = bytes_per_vec
        st.session_state["tr_recalls"] = recalls
        st.session_state["tr_M_vals"] = available_M + [None]
        st.session_state["tr_dim"] = dim_trade

    if "tr_bytes" in st.session_state:
        bpv = st.session_state["tr_bytes"]
        recs = st.session_state["tr_recalls"]
        m_vals = st.session_state["tr_M_vals"]
        dim_t = st.session_state["tr_dim"]

        fig = go.Figure()
        colors = ["steelblue"] * (len(bpv) - 1) + ["green"]
        labels = [f"IVFPQ M={m}" if m else "Flat (exact)" for m in m_vals]
        fig.add_trace(go.Scatter(
            x=bpv, y=recs, mode="lines+markers+text",
            line=dict(color="steelblue", width=2),
            marker=dict(color=colors, size=10),
            text=labels,
            textposition="top center",
        ))
        fig.update_layout(
            xaxis_title="Bytes per vector",
            yaxis_title="Recall@k",
            yaxis_range=[0, 1.1],
            height=450,
            title=f"IVFPQ compression/recall tradeoff (D={dim_t})",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("**Compression table:**")
        rows = []
        for m, b, r in zip(m_vals, bpv, recs):
            if m:
                rows.append({"M": m, "bytes/vector": b, f"recall@{k_trade}": f"{r:.3f}",
                              "compression": f"{dim_t * 4 // b}×"})
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
