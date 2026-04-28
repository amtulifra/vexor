"""
Flat index visualizer — distance heatmap + step-by-step brute-force search.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from vexor.indexes.flat import FlatIndex

st.set_page_config(layout="wide")
st.title("01 — Flat Index: Brute-Force Exact Search")

# --- Controls ---
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
n_vectors = col_ctrl1.slider("Number of vectors", 20, 500, 100)
dim = col_ctrl2.slider("Dimensions", 2, 128, 16)
metric = col_ctrl3.selectbox("Distance metric", ["cosine", "l2", "inner_product"])
k = st.slider("k (results)", 1, 20, 5)

if st.button("Generate random vectors"):
    rng = np.random.default_rng(42)
    st.session_state["flat_vecs"] = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    st.session_state["flat_query"] = rng.standard_normal(dim).astype(np.float32)
    st.session_state["flat_search_step"] = 0
    st.session_state["flat_visits"] = []

vecs: np.ndarray | None = st.session_state.get("flat_vecs")
query: np.ndarray | None = st.session_state.get("flat_query")

if vecs is None:
    st.info("Click 'Generate random vectors' to begin.")
    st.stop()

# Build index
index = FlatIndex(metric=metric)
for v in vecs:
    index.add(v)

results = index.search(query, k=k)
result_ids = {r[0] for r in results}

# --- Layout ---
col_heat, col_scatter = st.columns(2)

# Distance heatmap (capped at 100 vectors to keep it readable)
with col_heat:
    st.subheader("Distance Heatmap")
    cap = min(len(vecs), 100)
    from vexor.distance.kernels import batch_cosine, batch_l2, batch_inner_product
    dist_fn = {"cosine": batch_cosine, "l2": batch_l2, "inner_product": batch_inner_product}[metric]
    heat = np.zeros((cap, cap), dtype=np.float32)
    for i in range(cap):
        heat[i] = dist_fn(vecs[i], vecs[:cap])
    fig_heat = px.imshow(
        heat,
        color_continuous_scale="Viridis",
        labels={"color": "distance"},
        title=f"Pairwise {metric} distances (first {cap} vectors)",
    )
    fig_heat.update_layout(height=420)
    st.plotly_chart(fig_heat, use_container_width=True)

# 2D PCA scatter with search animation
with col_scatter:
    st.subheader("Search Animation (2D PCA)")

    pca = PCA(n_components=2)
    all_pts = np.vstack([vecs, query])
    coords = pca.fit_transform(all_pts)
    vec_coords = coords[:-1]
    query_coord = coords[-1]

    # Step-by-step animation
    step = st.session_state.get("flat_search_step", len(vecs))
    col_prev, col_next, col_all = st.columns(3)
    if col_prev.button("← Prev") and step > 0:
        st.session_state["flat_search_step"] = step - 1
        step -= 1
    if col_next.button("Next →") and step < len(vecs):
        st.session_state["flat_search_step"] = step + 1
        step += 1
    if col_all.button("Show all"):
        st.session_state["flat_search_step"] = len(vecs)
        step = len(vecs)

    visited = set(range(step))

    colors = []
    for i in range(len(vecs)):
        if i in result_ids and i in visited:
            colors.append("green")
        elif i in visited:
            colors.append("steelblue")
        else:
            colors.append("lightgray")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vec_coords[:, 0], y=vec_coords[:, 1],
        mode="markers",
        marker=dict(color=colors, size=8, line=dict(width=1, color="white")),
        name="vectors",
    ))
    fig.add_trace(go.Scatter(
        x=[query_coord[0]], y=[query_coord[1]],
        mode="markers",
        marker=dict(symbol="star", size=18, color="red"),
        name="query",
    ))
    fig.update_layout(
        height=420,
        title=f"Visited {step}/{len(vecs)} | Top-{k} = green",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# Stats
st.divider()
st.subheader("Search Results")
cols = st.columns(len(results) or 1)
for idx, (vec_id, dist) in enumerate(results):
    cols[idx].metric(f"Rank {idx+1}", f"ID {vec_id}", f"dist = {dist:.4f}")

from vexor.distance.kernels import batch_cosine, batch_l2, batch_inner_product
dist_fn2 = {"cosine": batch_cosine, "l2": batch_l2, "inner_product": batch_inner_product}[metric]
all_dists = dist_fn2(query, vecs)
st.caption(f"Exact recall@{k} = 1.0 (flat index is always exact) | {n_vectors}×{dim} = {n_vectors*dim:,} ops")
