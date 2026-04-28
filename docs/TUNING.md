# Vexor Tuning Guide

## HNSW

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `M` | Connections per node per layer. Higher = denser graph, better recall, more memory. | 16 (low-dim), 32 (high-dim) |
| `ef_construction` | Candidate pool at build time. Higher = better quality graph, slower inserts. | 100–400 |
| `ef_search` | Candidate pool at query time. Higher = better recall, slower search. | 50–200 |
| `sq=True` | int8 quantization for traversal. Reduces memory 4×, <1% recall drop. | Enable at scale |

**Rule of thumb:** `ef_search >= k`. For 99% recall, `ef_search ≈ 2–5 × k` is usually enough.

## IVF

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `nlist` | Number of Voronoi cells. More = finer partitioning, higher build cost. | `sqrt(N)` to `4 * sqrt(N)` |
| `nprobe` | Cells to search per query. 10–20% of nlist is a good starting point. | `nlist / 10` |
| `online_updates` | Keep centroids current under streaming inserts. | True for streaming data |

## IVFPQ

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `M` | PQ subspaces. Each adds 1 byte/vector. More = better recall, more memory. | 8–96 |
| `K` | Codebook size. K=256 = 1 byte per code (standard). | 256 |
| `nprobe` | IVF probe count. Same tradeoff as plain IVF. | `nlist / 10` |

**Memory formula:** `bytes/vector = M` (PQ codes only, not centroids or codebooks)

## Filter Selectivity and Adaptive ef

At selectivity `s` (fraction of vectors matching the filter):

```
ef_used = max(ef_base, ceil(k / s * 1.2))
```

At 5% selectivity with k=10: `ef_used = ceil(10 / 0.05 * 1.2) = 240`.

The 1.2 oversampling factor accounts for the fact that the graph might not have a
perfect path to all k matching vectors at a given ef. Increase to 1.5–2.0 if
recall drops at very low selectivities (< 1%).

## When to Use Each Index

| Use case | Recommended index |
|----------|------------------|
| < 10k vectors, exact results needed | Flat |
| < 20 dimensions, exact results | KD-Tree |
| High-recall ANN, < 1M vectors | HNSW |
| > 1M vectors, memory-constrained | IVF or IVFPQ |
| Metadata-filtered search | HNSW with in-graph filtering |
| Streaming data (new vectors arriving continuously) | IVF with `online_updates=True` |
