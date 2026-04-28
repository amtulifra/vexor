"""
IVF visualizer — Voronoi cluster diagram + centroid drift animation.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from vexor.indexes.ivf import IVFIndex
from vexor.indexes.flat import FlatIndex

st.set_page_config(layout="wide")
st.title("04 — IVF: Inverted File Index with Online K-Means")

tab_voronoi, tab_drift = st.tabs(["Voronoi Cluster View", "Centroid Drift Demo"])


def _voronoi_scatter(vecs_2d, centroids_2d, labels, query_2d, probed_ids, k_result_ids):
    n_clusters = len(centroids_2d)
    cluster_colors = [
        f"hsl({int(360 * i / n_clusters)}, 60%, 65%)" for i in range(n_clusters)
    ]

    fig = go.Figure()
    for c_id in range(n_clusters):
        mask = labels == c_id
        color = cluster_colors[c_id]
        opacity = 0.9 if c_id in probed_ids else 0.3
        fig.add_trace(go.Scatter(
            x=vecs_2d[mask, 0], y=vecs_2d[mask, 1],
            mode="markers",
            marker=dict(color=color, size=7, opacity=opacity),
            name=f"Cluster {c_id}" if c_id in probed_ids else None,
            showlegend=c_id in probed_ids,
        ))

    for c_id, (cx, cy) in enumerate(centroids_2d):
        color = cluster_colors[c_id]
        size = 16 if c_id in probed_ids else 10
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker=dict(symbol="diamond", size=size, color=color,
                        line=dict(width=2, color="black")),
            showlegend=False,
        ))

    for vid in k_result_ids:
        fig.add_trace(go.Scatter(
            x=[vecs_2d[vid, 0]], y=[vecs_2d[vid, 1]], mode="markers",
            marker=dict(symbol="circle-open", size=16, color="green",
                        line=dict(width=3, color="green")),
            showlegend=False,
        ))

    if query_2d is not None:
        fig.add_trace(go.Scatter(
            x=[query_2d[0]], y=[query_2d[1]], mode="markers",
            marker=dict(symbol="star", size=20, color="red"),
            name="query",
        ))

    fig.update_layout(height=500, title="Voronoi clusters (blue outline = probed)")
    return fig


with tab_voronoi:
    col1, col2, col3 = st.columns(3)
    nlist = col1.slider("nlist (clusters)", 4, 32, 8, key="ivf_nlist")
    nprobe = col2.slider("nprobe", 1, 16, 3, key="ivf_nprobe")
    n_vecs = col3.slider("Vectors", 100, 1000, 300, key="ivf_n")
    k_ivf = st.slider("k", 1, 20, 5, key="ivf_k")

    if st.button("Build IVF index", key="ivf_build"):
        rng = np.random.default_rng(1)
        vecs = rng.standard_normal((n_vecs, 16)).astype(np.float32)
        query = rng.standard_normal(16).astype(np.float32)

        idx_ivf = IVFIndex(nlist=nlist, nprobe=nprobe, metric="l2", online_updates=False)
        idx_ivf.train(vecs)
        for v in vecs:
            idx_ivf.add(v)

        pca = PCA(n_components=2)
        all_pts = np.vstack([vecs, idx_ivf._centroids, query])
        coords = pca.fit_transform(all_pts)

        st.session_state.update({
            "ivf_vecs": vecs,
            "ivf_query": query,
            "ivf_idx": idx_ivf,
            "ivf_coords": coords[:n_vecs],
            "ivf_centroid_coords": coords[n_vecs:n_vecs + nlist],
            "ivf_query_coord": coords[-1],
        })

    if "ivf_idx" in st.session_state:
        idx_ivf = st.session_state["ivf_idx"]
        vecs = st.session_state["ivf_vecs"]
        query = st.session_state["ivf_query"]
        coords = st.session_state["ivf_coords"]
        centroid_coords = st.session_state["ivf_centroid_coords"]
        query_coord = st.session_state["ivf_query_coord"]

        results = idx_ivf.search(query, k=k_ivf, nprobe=nprobe)
        result_ids = {r[0] for r in results}

        centroid_dists = np.array([
            float(np.linalg.norm(query - c)) for c in idx_ivf._centroids
        ])
        probed_ids = set(np.argsort(centroid_dists)[:nprobe])

        labels = np.array([idx_ivf._nearest_centroid(v) for v in vecs])

        fig = _voronoi_scatter(coords, centroid_coords, labels, query_coord, probed_ids, result_ids)
        st.plotly_chart(fig, use_container_width=True)

        flat = FlatIndex(metric="l2")
        for v in vecs:
            flat.add(v)
        exact_ids = {r[0] for r in flat.search(query, k=k_ivf)}
        recall = len(result_ids & exact_ids) / max(len(exact_ids), 1)
        st.metric("Recall@k", f"{recall:.3f}")
        st.caption(f"Probing {nprobe}/{nlist} clusters. Increase nprobe for better recall.")


with tab_drift:
    st.subheader("Static vs online-updated centroids under streaming data")
    st.write("Add more vectors and watch the online-updated centroids follow the data while frozen ones drift out of alignment.")

    col1, col2 = st.columns(2)
    n_init = col1.slider("Initial training vectors", 100, 500, 200, key="drift_init")
    nlist_d = col2.slider("nlist", 4, 16, 6, key="drift_nlist")

    if st.button("Initialize drift demo", key="drift_init_btn"):
        rng = np.random.default_rng(5)
        init_vecs = rng.standard_normal((n_init, 8)).astype(np.float32)

        static_idx = IVFIndex(nlist=nlist_d, metric="l2", online_updates=False)
        static_idx.train(init_vecs)
        for v in init_vecs:
            static_idx.add(v)

        online_idx = IVFIndex(nlist=nlist_d, metric="l2", online_updates=True)
        online_idx.train(init_vecs)
        for v in init_vecs:
            online_idx.add(v)

        st.session_state.update({
            "drift_static": static_idx,
            "drift_online": online_idx,
            "drift_vecs": list(init_vecs),
            "drift_rng": rng,
            "drift_recalls_static": [],
            "drift_recalls_online": [],
            "drift_steps": [],
            "drift_step_n": 0,
        })

    if "drift_static" in st.session_state:
        static_idx = st.session_state["drift_static"]
        online_idx = st.session_state["drift_online"]
        drift_vecs = st.session_state["drift_vecs"]
        rng_d = st.session_state["drift_rng"]
        recalls_s = st.session_state["drift_recalls_static"]
        recalls_o = st.session_state["drift_recalls_online"]
        steps_d = st.session_state["drift_steps"]
        step_n = st.session_state["drift_step_n"]

        batch = st.slider("Vectors to add", 10, 200, 50, key="drift_batch")
        if st.button("Add more vectors (shifted distribution)", key="drift_add"):
            shift = rng_d.standard_normal(8).astype(np.float32) * 3
            new_vecs = rng_d.standard_normal((batch, 8)).astype(np.float32) + shift
            for v in new_vecs:
                static_idx.add(v)
                online_idx.add(v)
                drift_vecs.append(v)

            all_v = np.stack(drift_vecs)
            flat_d = FlatIndex(metric="l2")
            for v in all_v:
                flat_d.add(v)

            q_test = rng_d.standard_normal((10, 8)).astype(np.float32)
            rs = ro = 0.0
            for q in q_test:
                exact = {r[0] for r in flat_d.search(q, k=10)}
                sr = {r[0] for r in static_idx.search(q, k=10)}
                or_ = {r[0] for r in online_idx.search(q, k=10)}
                if exact:
                    rs += len(sr & exact) / len(exact)
                    ro += len(or_ & exact) / len(exact)
            recalls_s.append(rs / len(q_test))
            recalls_o.append(ro / len(q_test))
            step_n += batch
            steps_d.append(len(drift_vecs))
            st.session_state.update({
                "drift_step_n": step_n,
                "drift_recalls_static": recalls_s,
                "drift_recalls_online": recalls_o,
                "drift_steps": steps_d,
            })

        if steps_d:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=steps_d, y=recalls_s, mode="lines+markers",
                                      line=dict(color="tomato", width=2), name="Static centroids"))
            fig.add_trace(go.Scatter(x=steps_d, y=recalls_o, mode="lines+markers",
                                      line=dict(color="steelblue", width=2), name="Online-updated centroids"))
            fig.update_layout(
                xaxis_title="Total vectors in index",
                yaxis_title="Recall@10",
                yaxis_range=[0, 1.05],
                height=400,
                title="Recall under streaming data: static vs online centroids",
            )
            st.plotly_chart(fig, use_container_width=True)
