"""
KD-Tree visualizer — space partition view + curse-of-dimensionality demo.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from vexor.indexes.flat import FlatIndex
from vexor.indexes.kdtree import KDTreeIndex

st.set_page_config(layout="wide")
st.title("02 — KD-Tree: Space Partitioning")

tab_viz, tab_curse = st.tabs(["Space Partition View", "Curse of Dimensionality"])

with tab_viz:
    col1, col2, col3 = st.columns(3)
    n_vecs = col1.slider("Vectors", 30, 300, 80, key="kd_n")
    leaf = col2.slider("Leaf size", 2, 30, 10, key="kd_leaf")
    k_search = col3.slider("k", 1, 15, 5, key="kd_k")

    if st.button("Build KD-Tree", key="kd_build"):
        rng = np.random.default_rng(7)
        vecs_2d = rng.standard_normal((n_vecs, 2)).astype(np.float32)
        query_2d = rng.standard_normal(2).astype(np.float32)
        st.session_state["kd_vecs"] = vecs_2d
        st.session_state["kd_query"] = query_2d

    vecs = st.session_state.get("kd_vecs")
    query = st.session_state.get("kd_query")

    if vecs is None:
        st.info("Click 'Build KD-Tree' to begin.")
        st.stop()

    idx = KDTreeIndex(metric="l2")
    idx._LEAF_SIZE = leaf
    for v in vecs:
        idx.add(v)
    idx.build()

    results = idx.search(query, k=k_search)
    result_ids = {r[0] for r in results}

    flat_idx = FlatIndex(metric="l2")
    for v in vecs:
        flat_idx.add(v)
    exact = flat_idx.search(query, k=k_search)
    exact_ids = {r[0] for r in exact}
    recall = len(result_ids & exact_ids) / max(len(exact_ids), 1)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("2D Scatter with KD-Tree Splits")
        colors = []
        for i in range(len(vecs)):
            if i in result_ids:
                colors.append("green")
            else:
                colors.append("steelblue")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vecs[:, 0], y=vecs[:, 1],
            mode="markers",
            marker=dict(color=colors, size=9),
            name="vectors",
        ))
        fig.add_trace(go.Scatter(
            x=[query[0]], y=[query[1]],
            mode="markers",
            marker=dict(symbol="star", size=18, color="red"),
            name="query",
        ))
        fig.update_layout(height=450, title=f"recall@{k_search} = {recall:.2f}")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Green = top-k results from KD-tree search")

    with col_right:
        st.subheader("Statistics")
        st.metric("Recall@k", f"{recall:.3f}")
        st.metric("Index size", f"{len(vecs)} vectors")
        st.metric("Dimensions", "2")
        st.write("KD-tree provides exact results in 2D. Try the Curse of Dimensionality tab to see it fail at high dims.")


with tab_curse:
    st.subheader("Recall vs Dimensionality")
    st.write("Drag the slider to watch recall collapse as dimensions increase.")

    max_dim = st.slider("Max dimension to test", 5, 100, 50, key="curse_max_dim")
    n_test = st.slider("Test vectors", 100, 1000, 200, key="curse_n")
    k_test = st.slider("k", 1, 20, 10, key="curse_k")

    if st.button("Run curse-of-dimensionality benchmark"):
        dims = list(range(2, max_dim + 1, 2))
        recalls = []
        rng = np.random.default_rng(0)

        progress = st.progress(0)
        for step_i, d in enumerate(dims):
            vecs_d = rng.standard_normal((n_test, d)).astype(np.float32)
            queries_d = rng.standard_normal((10, d)).astype(np.float32)

            kd = KDTreeIndex(metric="l2")
            fl = FlatIndex(metric="l2")
            for v in vecs_d:
                kd.add(v)
                fl.add(v)
            kd.build()

            total = hits = 0
            for q in queries_d:
                kd_res = {r[0] for r in kd.search(q, k=k_test)}
                fl_res = {r[0] for r in fl.search(q, k=k_test)}
                hits += len(kd_res & fl_res)
                total += k_test
            recalls.append(hits / total if total > 0 else 0.0)
            progress.progress((step_i + 1) / len(dims))

        st.session_state["curse_dims"] = dims
        st.session_state["curse_recalls"] = recalls

    dims = st.session_state.get("curse_dims")
    recalls = st.session_state.get("curse_recalls")

    if dims and recalls:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dims, y=recalls, mode="lines+markers",
            line=dict(color="steelblue", width=2),
            marker=dict(size=6),
            name="KD-tree recall",
        ))
        fig.add_hline(y=0.95, line_dash="dash", line_color="green", annotation_text="0.95 threshold")
        fig.update_layout(
            xaxis_title="Dimensions",
            yaxis_title="Recall@k",
            yaxis_range=[0, 1.05],
            height=400,
            title="KD-Tree Recall Collapse Above ~20 Dimensions",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each point is the average recall@k over 10 random queries. The tree becomes useless above ~20 dims — this is why HNSW exists.")
    else:
        st.info("Click 'Run curse-of-dimensionality benchmark' to see the chart.")
