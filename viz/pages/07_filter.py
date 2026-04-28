"""
Filter correctness demo — in-graph vs post-filter recall at varying selectivity.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from vexor.indexes.hnsw import HNSWIndex
from vexor.indexes.flat import FlatIndex

st.set_page_config(layout="wide")
st.title("07 — Metadata Filter: In-Graph vs Post-Filter")

st.markdown("""
This demo shows the key correctness advantage of Vexor's in-graph filtered search.

**Post-filter** (naive): search vector space → filter results → may return <k results at low selectivity.

**In-graph filter** (Vexor): filter during traversal + adaptive ef scaling → maintains recall at any selectivity.
""")

col1, col2, col3 = st.columns(3)
N_filter = col1.slider("Vectors", 300, 3000, 1000, key="f_n")
K_f = col2.slider("k", 1, 20, 10, key="f_k")
N_queries_f = col3.slider("Queries per point", 10, 50, 20, key="f_nq")
M_f = st.slider("HNSW M", 4, 32, 16, key="f_M")

SELECTIVITIES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]

if st.button("Run full sweep", key="f_run"):
    rng = np.random.default_rng(0)
    vecs_f = rng.standard_normal((N_filter, 32)).astype(np.float32)
    queries_f = rng.standard_normal((N_queries_f, 32)).astype(np.float32)

    vexor_recalls, post_recalls = [], []
    progress = st.progress(0)

    for step_i, sel in enumerate(SELECTIVITIES):
        n_match = max(1, int(N_filter * sel))
        labels = np.array(["match"] * n_match + ["other"] * (N_filter - n_match))
        rng.shuffle(labels)
        filt = {"label": "match"}

        idx_vexor = HNSWIndex(32, M=M_f, ef_construction=200, ef_search=100)
        idx_post = HNSWIndex(32, M=M_f, ef_construction=200, ef_search=100)
        flat_f = FlatIndex(metric="cosine")
        for i, v in enumerate(vecs_f):
            meta = {"label": labels[i]}
            idx_vexor.add(v, meta)
            idx_post.add(v, meta)
            flat_f.add(v, meta)

        v_r = p_r = 0.0
        for q in queries_f:
            exact = {r[0] for r in flat_f.search(q, k=K_f, filter=filt)}
            in_graph = {r[0] for r in idx_vexor.search(q, k=K_f, filter=filt)}
            post = {r[0] for r in idx_post.search(q, k=K_f)}
            post = set(list({r for r in post if labels[r] == "match"})[:K_f])
            if exact:
                v_r += len(in_graph & exact) / len(exact)
                p_r += len(post & exact) / len(exact)

        vexor_recalls.append(v_r / N_queries_f)
        post_recalls.append(p_r / N_queries_f)
        progress.progress((step_i + 1) / len(SELECTIVITIES))

    st.session_state["f_vexor"] = vexor_recalls
    st.session_state["f_post"] = post_recalls

v_recs = st.session_state.get("f_vexor")
p_recs = st.session_state.get("f_post")

if v_recs:
    sel_pcts = [s * 100 for s in SELECTIVITIES]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sel_pcts, y=v_recs, mode="lines+markers",
        line=dict(color="steelblue", width=3), marker=dict(size=8),
        name="Vexor in-graph filter",
    ))
    fig.add_trace(go.Scatter(
        x=sel_pcts, y=p_recs, mode="lines+markers",
        line=dict(color="tomato", width=3, dash="dash"), marker=dict(size=8),
        name="Post-filter (naive)",
    ))
    fig.add_hline(y=0.90, line_dash="dot", line_color="green",
                  annotation_text="0.90 recall threshold", annotation_position="bottom right")

    fig.update_layout(
        xaxis_title="Filter selectivity (%)",
        yaxis_title=f"Recall@{K_f}",
        yaxis_range=[0, 1.05],
        height=500,
        title="In-Graph Filtering vs Post-Filtering — Recall at Low Selectivity",
        xaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    collapse_sel = next((s for s, r in zip(SELECTIVITIES, p_recs) if r < 0.5), None)
    if collapse_sel:
        col_l.warning(f"Post-filter recall drops below 0.50 at {collapse_sel:.0%} selectivity")
    else:
        col_l.success("Post-filter recall stays above 0.50 across all tested selectivities")

    min_vexor = min(v_recs)
    col_r.metric("Minimum Vexor recall across all selectivities", f"{min_vexor:.3f}")
else:
    st.info("Click 'Run full sweep' to generate the comparison chart.")
