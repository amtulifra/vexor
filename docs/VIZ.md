# Vexor Visualization Guide

## Running the Dashboard

```bash
streamlit run viz/app.py
```

The dashboard opens at `http://localhost:8501`.

## Pages

| Page | Key Visualization |
|------|------------------|
| 01 Flat | N×N distance heatmap + step-by-step brute-force scan animation |
| 02 KD-Tree | 2D space partition with split lines + curse-of-dimensionality recall curve |
| 03 HNSW | Live layer graph, node-by-node insert, greedy search trace, filter demo |
| 04 IVF | Voronoi cluster diagram + centroid drift animation (static vs online) |
| 05 IVFPQ | Codebook usage heatmap + compression/recall tradeoff curve |
| 06 Benchmark | Interactive recall@10 vs QPS for all indexes |
| 07 Filter | Recall collapse demo: in-graph vs post-filter across selectivities |

## The Thesis Chart (Page 07)

The filter selectivity demo on page 07 (and the HNSW page 03) is the core proof
of Vexor's correctness advantage. Drag the selectivity slider from 100% → 1%:

- Post-filter recall collapses below 10% selectivity (returns <2 results for k=10)
- Vexor in-graph filter maintains recall ≥ 0.90 at 1% selectivity via adaptive ef

## Hook System

To instrument your own code with the visualizer:

```python
from vexor.hooks.streamlit_hook import StreamlitHook
from vexor.indexes.hnsw import HNSWIndex

hook = StreamlitHook(session_key="my_events")
idx = HNSWIndex(dim=128, M=16, hook=hook)
```

Events accumulate in `st.session_state["my_events"]` as a list of dicts.
Read them to drive any Streamlit visualization.

In production, swap `StreamlitHook` for `NoopHook` (or omit the hook argument
entirely) — all methods are inherited no-ops with zero overhead.
