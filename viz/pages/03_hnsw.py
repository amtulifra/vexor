"""
HNSW visualizer — live graph build, step-by-step search, and filter demo.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from sklearn.decomposition import PCA
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from vexor.indexes.hnsw import HNSWIndex
from vexor.indexes.flat import FlatIndex

st.set_page_config(layout="wide")
st.title("03 — HNSW: Hierarchical Navigable Small World")

tab_build, tab_search, tab_filter = st.tabs(["Build Step-by-Step", "Search Step-by-Step", "Filter Correctness Demo"])


def _build_plotly_graph(index: HNSWIndex, layer: int, pca_coords: np.ndarray,
                         highlight_path: list[int] | None = None,
                         result_ids: set[int] | None = None,
                         query_coord: np.ndarray | None = None):
    G = nx.Graph()
    node_ids = [nid for nid in index._vectors if nid not in index._deleted]
    for nid in node_ids:
        if layer in index._layers[nid]:
            G.add_node(nid)
            for nb in index._layers[nid][layer]:
                if nb not in index._deleted:
                    G.add_edge(nid, nb)

    if not G.nodes:
        return go.Figure()

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pca_coords[u]
        x1, y1 = pca_coords[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    path_set = set(highlight_path or [])
    result_set = result_ids or set()

    node_colors, node_sizes = [], []
    for nid in G.nodes():
        if nid in result_set:
            node_colors.append("green")
            node_sizes.append(14)
        elif nid in path_set:
            node_colors.append("red")
            node_sizes.append(12)
        elif nid == index._entry_point:
            node_colors.append("gold")
            node_sizes.append(14)
        else:
            max_l = index._max_layer.get(nid, 0)
            node_colors.append("cornflowerblue" if max_l >= layer else "lightsteelblue")
            node_sizes.append(9)

    node_x = [pca_coords[nid][0] for nid in G.nodes()]
    node_y = [pca_coords[nid][1] for nid in G.nodes()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                              line=dict(color="#888", width=1), hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(color=node_colors, size=node_sizes, line=dict(width=1, color="white")),
        text=[str(nid) for nid in G.nodes()],
        hoverinfo="text", name="nodes",
    ))
    if query_coord is not None:
        fig.add_trace(go.Scatter(
            x=[query_coord[0]], y=[query_coord[1]], mode="markers",
            marker=dict(symbol="star", size=18, color="crimson"),
            name="query",
        ))
    fig.update_layout(height=500, showlegend=False,
                       title=f"Layer {layer} — {len(G.nodes())} nodes, {len(G.edges())} edges")
    return fig


with tab_build:
    col1, col2, col3, col4 = st.columns(4)
    M = col1.slider("M (connections)", 4, 32, 8, key="hb_M")
    ef_c = col2.slider("ef_construction", 20, 400, 100, key="hb_efc")
    n_to_build = col3.slider("Vectors to insert", 10, 200, 50, key="hb_n")
    dim_build = col4.slider("Dim", 2, 128, 16, key="hb_dim")

    if st.button("Initialize fresh index", key="hb_init"):
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((n_to_build, dim_build)).astype(np.float32)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vecs)
        st.session_state["hb_vecs"] = vecs
        st.session_state["hb_coords"] = coords
        st.session_state["hb_index"] = HNSWIndex(dim_build, M=M, ef_construction=ef_c)
        st.session_state["hb_step"] = 0

    idx: HNSWIndex | None = st.session_state.get("hb_index")
    vecs = st.session_state.get("hb_vecs")
    coords = st.session_state.get("hb_coords")
    step = st.session_state.get("hb_step", 0)

    if idx is None:
        st.info("Click 'Initialize fresh index' to begin.")
    else:
        col_ins, col_all = st.columns(2)
        if col_ins.button("Insert next vector →") and step < len(vecs):
            idx.add(vecs[step])
            st.session_state["hb_step"] = step + 1
        if col_all.button("Insert all remaining"):
            while step < len(vecs):
                idx.add(vecs[step])
                step += 1
            st.session_state["hb_step"] = step

        st.write(f"Inserted: **{step}** / {len(vecs)} vectors | Layers: 0–{idx._top_layer}")
        selected_layer = st.selectbox("View layer", list(range(idx._top_layer + 1)), index=0, key="hb_layer")
        if step > 0:
            fig = _build_plotly_graph(idx, selected_layer, coords)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Gold = entry point | Blue = nodes present at this layer | Edges = M connections")


with tab_search:
    st.subheader("Step-by-step greedy search")
    col1, col2, col3 = st.columns(3)
    n_s = col1.slider("Vectors", 30, 300, 80, key="hs_n")
    ef_s = col2.slider("ef_search", 10, 200, 50, key="hs_ef")
    k_s = col3.slider("k", 1, 20, 5, key="hs_k")

    if st.button("Build & search", key="hs_go"):
        rng = np.random.default_rng(99)
        vecs_s = rng.standard_normal((n_s, 32)).astype(np.float32)
        query_s = rng.standard_normal(32).astype(np.float32)
        pca_s = PCA(n_components=2)
        all_pts = np.vstack([vecs_s, query_s])
        coords_s = pca_s.fit_transform(all_pts)
        idx_s = HNSWIndex(32, M=8, ef_construction=100, ef_search=ef_s)
        for v in vecs_s:
            idx_s.add(v)
        results_s = idx_s.search(query_s, k=k_s)
        st.session_state.update({
            "hs_vecs": vecs_s, "hs_query": query_s,
            "hs_coords": coords_s, "hs_idx": idx_s,
            "hs_results": results_s,
        })

    if "hs_idx" in st.session_state:
        idx_s = st.session_state["hs_idx"]
        coords_s = st.session_state["hs_coords"]
        query_s = st.session_state["hs_query"]
        results_s = st.session_state["hs_results"]
        result_ids_s = {r[0] for r in results_s}
        query_coord_s = coords_s[-1]

        layer_view = st.selectbox("View layer", list(range(idx_s._top_layer + 1)), key="hs_layer")
        fig_s = _build_plotly_graph(idx_s, layer_view, coords_s[:-1],
                                     result_ids=result_ids_s,
                                     query_coord=query_coord_s)
        st.plotly_chart(fig_s, use_container_width=True)
        st.write("**Top results:**")
        for rank, (vid, dist) in enumerate(results_s):
            st.write(f"  Rank {rank+1}: ID {vid} | dist = {dist:.4f}")


with tab_filter:
    st.subheader("In-graph filtering vs post-filtering recall comparison")
    st.write("Drag the selectivity slider left to watch post-filter recall collapse while Vexor's in-graph filter holds.")

    n_filter = st.slider("Number of vectors", 200, 2000, 500, key="hf_n")
    selectivity_pct = st.slider("Filter selectivity (%)", 1, 100, 50, key="hf_sel")
    k_filter = st.slider("k", 1, 20, 10, key="hf_k")

    if st.button("Run comparison", key="hf_run"):
        rng = np.random.default_rng(0)
        vecs_f = rng.standard_normal((n_filter, 32)).astype(np.float32)
        selectivity = selectivity_pct / 100.0
        n_matching = max(1, int(n_filter * selectivity))
        labels = np.array(["match"] * n_matching + ["other"] * (n_filter - n_matching))
        rng.shuffle(labels)

        idx_vexor = HNSWIndex(32, M=16, ef_construction=200, ef_search=100)
        idx_post = HNSWIndex(32, M=16, ef_construction=200, ef_search=100)
        flat_f = FlatIndex(metric="cosine")

        for i, v in enumerate(vecs_f):
            meta = {"label": labels[i]}
            idx_vexor.add(v, meta)
            idx_post.add(v, meta)
            flat_f.add(v, meta)

        filter_f = {"label": "match"}
        n_queries = 20
        queries_f = rng.standard_normal((n_queries, 32)).astype(np.float32)

        vexor_recall = post_recall = 0.0
        for q in queries_f:
            exact = {r[0] for r in flat_f.search(q, k=k_filter, filter=filter_f)}
            in_graph = {r[0] for r in idx_vexor.search(q, k=k_filter, filter=filter_f)}
            post = {r[0] for r in idx_post.search(q, k=k_filter)}
            post = {r for r in post if labels[r] == "match"}
            post = set(list(post)[:k_filter])
            if exact:
                vexor_recall += len(in_graph & exact) / len(exact)
                post_recall += len(post & exact) / len(exact)

        vexor_recall /= n_queries
        post_recall /= n_queries
        st.session_state["hf_vexor"] = vexor_recall
        st.session_state["hf_post"] = post_recall

    col_v, col_p = st.columns(2)
    vexor_r = st.session_state.get("hf_vexor")
    post_r = st.session_state.get("hf_post")

    if vexor_r is not None:
        col_v.metric("Vexor in-graph filter recall@k", f"{vexor_r:.3f}",
                      delta=f"+{vexor_r - post_r:.3f} vs post-filter" if post_r is not None else None)
        col_p.metric("Post-filter recall@k", f"{post_r:.3f}")
        if post_r < 0.5:
            st.warning(f"Post-filter recall has collapsed to {post_r:.1%} at {selectivity_pct}% selectivity. "
                       "Vexor's in-graph filter maintains recall by scaling ef adaptively.")
    else:
        st.info("Click 'Run comparison' to see the results.")
