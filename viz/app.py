"""
Vexor live algorithm dashboard.

Run with:  streamlit run viz/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Vexor — Vector DB Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Vexor — Vector Database Visualizer")
st.markdown(
    """
    A live, interactive dashboard for every algorithm inside Vexor.

    Each page lets you build, step through, and benchmark a different index type.
    Use the sidebar to navigate.

    | Page | Algorithm | Key insight |
    |------|-----------|-------------|
    | 01 Flat | Brute-force exact search | Distance heatmap + step-by-step scan |
    | 02 KD-Tree | Space partitioning | Curse-of-dimensionality demo |
    | 03 HNSW | Hierarchical navigable small world | Live graph build + in-graph filter demo |
    | 04 IVF | Inverted file / k-means | Voronoi diagram + centroid drift |
    | 05 IVFPQ | Product quantization | Codebook grid + compression/recall tradeoff |
    | 06 Benchmark | All indexes | Recall vs QPS + memory footprint |
    | 07 Filter | In-graph vs post-filter | Recall collapse at low selectivity |
    """
)

st.info("Select a page from the sidebar to begin.")
