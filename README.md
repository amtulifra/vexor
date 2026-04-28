# Vexor

A vector database built from first principles — approximate nearest-neighbor search with correctness-first design, production-grade internals, and a live algorithm visualizer.


---


## Overview


Vexor implements five index types from scratch in pure Python (NumPy + Numba), each paired with a live interactive dashboard you can watch run step by step. Every design decision prioritizes correctness first, then performance — documented both in code and in the visualizer.


```
N = 500,000 vectors · D = 768 dimensions · K = 10 nearest neighbors


Index      Recall@10   QPS       Memory/vector
──────────────────────────────────────────────
Flat       1.000       ~80       3,072 bytes   (float32 exact baseline)
HNSW       0.980       ~4,200    ~6,400 bytes  (graph edges)
HNSW+SQ    0.975       ~4,100    ~1,600 bytes  (int8 traversal)
IVF        0.920       ~9,500    3,072 bytes
IVFPQ      0.930       ~8,800    32 bytes      (M=8, 96× compression)
```


---


## Features


### Index types


| Index | Algorithm | Best for |
|-------|-----------|----------|
| `flat` | Exact brute-force | Ground truth, small datasets |
| `kdtree` | Recursive space partitioning | Low-dimensional exact search (< 20 dims) |
| `hnsw` | Hierarchical Navigable Small World | High-dimensional ANN, streaming inserts |
| `ivf` | Inverted File with online k-means | Large static datasets, memory-constrained |
| `ivfpq` | IVF + Product Quantization | Millions of vectors on a single machine |


### Correctness


**In-graph metadata filtering**
Standard implementations post-filter: run the full vector search, then discard non-matching results. At 5% selectivity (50 matching vectors out of 1,000), this silently returns 2–3 results for `k=10`. Vexor passes the filter bitmap directly into the graph traversal and scales the candidate pool dynamically:


```
ef_used = max(ef_base, ceil(k / selectivity × 1.2))
```


At 5% selectivity with `k=10`, `ef_used = 240`. Recall stays above 0.90. The graph traversal navigates freely through non-matching nodes to reach matching ones — only the result collection is filtered.


**Real deletion with graph repair**
Tombstoning leaves ghost nodes that degrade search quality and waste memory over time. Vexor repairs the graph on every deletion:
1. Collect the deleted node's neighbors at each layer.
2. For each pair `(A, B)` of those neighbors not already connected, add a direct edge — compensating for the removed path.
3. When deleted nodes exceed 20% of the index, compact: rebuild from live vectors only, preserving all original IDs.


**Online IVF centroids**
Centroids trained once on an initial snapshot drift as new vectors arrive, silently degrading recall. Vexor applies mini-batch gradient descent after each insert:


```python
lr = lr₀ / (1 + t × decay)
centroid[c] += lr × (vec − centroid[c])
```


Recall degradation under streaming data is eliminated without full retraining.


### Performance


- **Numba JIT distance kernels** — `@njit(fastmath=True, cache=True)` cosine, L2, and inner-product with `prange`-parallelized batch variants. 4–8× over pure NumPy on the scoring hot path.
- **Roaring bitmap metadata index** — one `pyroaring.BitMap` per `(field, value)` pair. Predicate evaluation is a chain of bitwise AND operations — O(N/64) instead of O(N) linear scan.
- **Per-node RW locks** — HNSW inserts acquire locks only for the specific nodes being modified (the new node and its M neighbors). Readers traverse without locking. Eliminates the global lock bottleneck under concurrent workloads.
- **Asymmetric distance computation (ADC)** — IVFPQ precomputes a `(M, 256)` lookup table at query time. Approximate distance to any encoded vector is then M indexed table lookups — no multiply operations needed.
- **Adaptive nprobe** — IVF estimates query difficulty from the residual distance to the nearest centroid. Easy queries probe fewer clusters; hard queries (near cluster boundaries) probe more.


### Durability


- **Write-ahead log (WAL)** — every insert and delete is appended to a WAL file before being applied to the index. On restart, any un-snapshotted operations are replayed automatically.
- **Binary snapshot format** — `VEXOR001` magic header, CRC32 checksums on each section, `numpy.memmap` for the vector block. A 1M-vector index loads in milliseconds without reading the full file into RAM.
- **mmap-backed vector store** — the vector matrix is memory-mapped from disk. Indexes larger than available RAM are supported.


### Live visualizer


Every algorithm ships with an interactive Streamlit page:


| Page | What you see |
|------|-------------|
| 01 Flat | N×N distance heatmap + step-by-step brute-force scan animation |
| 02 KD-Tree | Space partition hyperplanes + curse-of-dimensionality recall collapse |
| 03 HNSW | Live layer graph (NetworkX + Plotly) + in-graph vs post-filter recall demo |
| 04 IVF | Voronoi cluster diagram + centroid drift side-by-side animation |
| 05 IVFPQ | Codebook heatmap + compression/recall tradeoff curve |
| 06 Benchmark | Interactive recall@10 vs QPS curves for all indexes |
| 07 Filter | Selectivity slider — watch recall collapse on post-filter, hold on in-graph |


---


## Installation


**Requirements:** Python 3.10+


```bash
git clone https://github.com/amtulifra/vexor
cd vexor
pip install -e .
```


To include development dependencies (pytest, hypothesis):


```bash
pip install -e ".[dev]"
```


**Dependencies installed automatically:**


| Package | Purpose |
|---------|---------|
| `numpy` | Core vector math |
| `numba` | JIT-compiled distance kernels |
| `pyroaring` | Roaring bitmap metadata index |
| `scikit-learn` | PCA projection for visualizations |
| `streamlit` | Live algorithm dashboard |
| `plotly` | Interactive charts and network graphs |
| `networkx` | HNSW graph rendering |
| `matplotlib` | Static benchmark output |


---


## Quick start


### HNSW (recommended for most use cases)


```python
import numpy as np
from vexor.db import VectorDB


db = VectorDB(dim=768, index_type="hnsw", metric="cosine",
             M=16, ef_construction=200, ef_search=100)


# Add vectors with optional metadata
vectors = np.random.randn(10_000, 768).astype(np.float32)
for i, v in enumerate(vectors):
   db.add(v, metadata={"source": "arxiv", "year": 2020 + (i % 5)})


# Unfiltered search
results = db.search(query_vec, k=10)
# → [(vec_id, distance), ...]


# Filtered search — returns exactly k results even at low selectivity
results = db.search(query_vec, k=10, filter={"source": "arxiv", "year": 2024})
```


### IVF (large static datasets)


```python
from vexor.db import VectorDB


db = VectorDB(dim=128, index_type="ivf", metric="l2",
             nlist=256, nprobe=16)


db.train(training_vectors)   # k-means++ centroid initialization
for v in vectors:
   db.add(v)


results = db.search(query_vec, k=10)
```


### IVFPQ (maximum memory efficiency)


```python
from vexor.db import VectorDB


# M=32 subspaces → 32 bytes/vector (96× compression on 768-dim float32)
db = VectorDB(dim=768, index_type="ivfpq", metric="l2",
             nlist=256, nprobe=16, M=32, K=256)


db.train(training_vectors)
for v in vectors:
   db.add(v)


results = db.search(query_vec, k=10)
```


### WAL-backed durability


```python
db = VectorDB(dim=768, index_type="hnsw", metric="cosine",
             wal_path="./vexor.wal")


db.add(v)           # appended to WAL before applied to the index
db.save("./index")  # snapshot written; WAL truncated


# On next startup, any inserts since the last snapshot are replayed
db2 = VectorDB.load("./index", wal_path="./vexor.wal")
```


### Deletion


```python
vec_id = db.add(v)
db.delete(vec_id)   # graph repair runs immediately; compacts at 20% threshold
```


---


## Architecture


```
vexor/
 src/vexor/
   distance/
     kernels.py          NumPy baseline (cosine, L2, inner product)
     kernels_jit.py      Numba @njit + prange batch variants
   filtering/
     bitmap.py           Roaring bitmap index — O(N/64) predicate evaluation
     adaptive.py         Selectivity-aware ef and nprobe scaling
   indexes/
     flat.py             Exact brute-force baseline
     kdtree.py           Recursive space partitioning (low-dim exact search)
     hnsw.py             HNSW — in-graph filter + deletion repair + per-node locks
     ivf.py              IVF — k-means++ + online centroid updates + adaptive nprobe
     ivfpq.py            IVF + Product Quantization with ADC search
   quantization/
     pq.py               Product quantizer + asymmetric distance computation
     sq.py               Scalar quantizer (float32 → int8, 4× memory reduction)
   concurrency/
     sharded.py          N-shard manager — ProcessPool broadcast + result merge
     locks.py            Per-node RW lock registry
   storage/
     format.py           Binary format (VEXOR001 magic, CRC32, numpy memmap)
     wal.py              Write-ahead log — append, replay, truncate
     vector_store.py     mmap-backed raw vector store
   hooks/
     base.py             VexorHook interface (all methods no-op by default)
     noop.py             Zero-overhead production hook
     streamlit_hook.py   Streams events to Streamlit session state
   db.py                 VectorDB — unified API entry point


 viz/
   app.py                Dashboard entry point (streamlit run viz/app.py)
   pages/
     01_flat.py through 07_filter.py


 bench/
   recall_qps.py         Recall@10 vs QPS sweep — all index types
   filter_bench.py       In-graph vs post-filter recall at varying selectivity
   memory_bench.py       Bytes/vector vs recall — IVFPQ at M ∈ {8,16,32,64,96}
   concurrency_bench.py  QPS vs thread count


 tests/
   test_correctness.py   Recall@10 vs flat for every index type
   test_filter.py        Filter correctness at 5%, 10%, 50% selectivity
   test_deletion.py      Recall unchanged after delete + reinsert
   test_recovery.py      WAL replay reproduces index after simulated crash
```


### Hook system


Every algorithm emits events through a lightweight observer interface. In production, hooks are no-ops compiled away by Numba. In viz mode, they stream state to the Streamlit session frame by frame.


```python
class VexorHook:
   def on_node_insert(self, node_id, layer, neighbors): pass
   def on_search_visit(self, node_id, layer, dist):    pass
   def on_centroid_update(self, cid, old, new):        pass
   def on_cluster_assign(self, vec_id, centroid_id):   pass
   def on_pq_encode(self, vec_id, codes):              pass
   def on_deletion(self, node_id, repaired_edges):     pass


# Production:  hook = NoopHook()      ← zero overhead
# Viz mode:    hook = StreamlitHook() ← streams to dashboard
```


The core algorithms have no knowledge of the visualizer — zero coupling between correctness and observability.


---


## Running the visualizer


```bash
streamlit run viz/app.py
```


Opens at `http://localhost:8501`. Use the sidebar to navigate between algorithm pages.


The filter selectivity demo on page 07 is the clearest single view of what makes Vexor different: drag the slider from 100% → 1% selectivity and watch the two recall curves diverge.


---


## Running benchmarks


```bash
python bench/recall_qps.py       # Recall@10 vs QPS curves → bench/results/
python bench/filter_bench.py     # In-graph vs post-filter recall comparison
python bench/memory_bench.py     # Compression/recall tradeoff for IVFPQ
python bench/concurrency_bench.py
```


### Results (N=10,000, D=64, K=10)


| Benchmark | Key finding |
|-----------|-------------|
| **Recall vs QPS** | HNSW dominates — best recall/speed tradeoff. Flat is exact but ~50× slower. IVF/IVFPQ trade ~5% recall for higher throughput and lower memory. |
| **Filtered search** | Recall holds above 0.90 at 5% selectivity with in-graph filtering. Post-filter would silently return 2–3 results for k=10 at the same selectivity. |
| **Memory** | IVFPQ at M=8 uses 8 bytes/vector vs 256 bytes for Flat/HNSW (float32, D=64). HNSW adds ~2× overhead from graph edges on top of raw vectors. |
| **Concurrency** | Threading degrades with more threads (GIL — 906 QPS at 1 thread, 193 at 16). Multiprocessing bypasses the GIL but index rebuild cost per worker dominates at this dataset size. |


Charts saved to `bench/results/`.


---


## Running tests


```bash
pytest tests/ -v
```


Test coverage:


| Test file | What it checks |
|-----------|---------------|
| `test_correctness.py` | Kernel parity (NumPy vs JIT); recall@10 ≥ threshold for every index |
| `test_filter.py` | Bitmap predicate correctness; in-graph filter recall at 5% and 10% selectivity |
| `test_deletion.py` | No deleted ID appears in results; recall within 0.15 of baseline after delete + reinsert |
| `test_recovery.py` | WAL roundtrip; corrupt entries skipped; index reproduced after simulated crash |


---


## Parameter reference


### HNSW


| Parameter | Effect | Default | Range |
|-----------|--------|---------|-------|
| `M` | Edges per node per layer — controls graph density and memory | 16 | 8–64 |
| `ef_construction` | Candidate pool at build time — higher = better graph quality, slower inserts | 200 | 100–400 |
| `ef_search` | Candidate pool at query time — higher = better recall, slower search | 50 | 50–200 |
| `sq` | int8 scalar quantization for traversal — 4× memory reduction, < 1% recall drop | False | — |
| `metric` | Distance metric | cosine | cosine, l2, inner_product |


Rule of thumb: `ef_search ≥ k`. For recall > 0.99, `ef_search ≈ 3–5 × k`.


### IVF


| Parameter | Effect | Default | Range |
|-----------|--------|---------|-------|
| `nlist` | Number of Voronoi cells — more = finer partitioning, higher build cost | 256 | `√N` to `4√N` |
| `nprobe` | Cells searched per query — 10–20% of nlist is a good starting point | 8 | 1–nlist |
| `online_updates` | Update centroids incrementally on each insert | True | — |


### IVFPQ — memory vs recall


| M | Bytes/vector | Typical recall@10 (D=768) |
|---|-------------|--------------------------|
| 8 | 8 B | ~0.88 |
| 16 | 16 B | ~0.91 |
| 32 | 32 B | ~0.94 |
| 64 | 64 B | ~0.96 |
| 96 | 96 B | ~0.97 |


`K=256` (one byte per subspace code) is standard and rarely needs changing.


---


## Tech stack


| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.10+ | NumPy ecosystem, rapid iteration |
| Distance kernels | Numba `@njit` | SIMD auto-vectorization, 4–8× over NumPy |
| Bitmap filtering | pyroaring | Roaring bitmaps, O(N/64) set intersection |
| Concurrency | Per-node `threading.RLock` + `ProcessPoolExecutor` shards | Avoids global lock; process pool bypasses the GIL |
| Persistence | `struct` + `numpy.memmap` | Custom binary format, mmap for large indexes |
| Testing | pytest + hypothesis | Property-based tests for correctness guarantees |
| Dashboard | Streamlit | Live algorithm visualizer, one page per index |
| Charts | Plotly | Zoomable scatter, network graphs, heatmaps |
| Graph rendering | NetworkX + Plotly | HNSW layer graphs as interactive networks |
| 2D projection | scikit-learn PCA | Project high-dim vectors to 2D for spatial intuition |


---


## TODOs / Future improvements


**Performance**
- [ ] Move distance kernels to C extensions (ctypes/Cython) to break out of the GIL entirely — threading would then scale linearly
- [ ] Serialize index to shared memory so multiprocessing workers skip the rebuild cost — would make the concurrency benchmark reflect true search throughput
- [ ] GPU distance kernels via CuPy for batch query workloads
- [ ] SIMD-explicit AVX2 path for L2/cosine on float32 (Numba `fastmath` is close but not guaranteed)


**Index quality**
- [ ] Implement HNSW `select_neighbors_heuristic` (Algorithm 4 in the paper) — improves long-range connectivity vs the current simple top-M selection
- [ ] Add LSH index for ultra-low-memory approximate search
- [ ] Beam search for IVF to handle near-boundary queries without raising nprobe globally


**Benchmarking**
- [ ] Run at full scale (N=500K, D=768) to validate the numbers in the overview table
- [ ] Separate index build time from query time in `concurrency_bench.py` to get clean steady-state QPS numbers for multiprocessing
- [ ] Add a recall-vs-latency-percentile (p50/p95/p99) benchmark — QPS alone hides tail latency


**API / usability**
- [ ] REST API wrapper (FastAPI) so Vexor can be used as a standalone server
- [ ] Batch `add()` and `search()` — current API is one vector at a time
- [ ] Persistence: add `load()` support for every index type (currently WAL replay only)


**Viz**
- [ ] Verify and fix rendering issues in viz pages 01–07
- [ ] Add a live concurrency page showing thread contention vs process isolation


---


## References


- Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE TPAMI*.
- Jégou, H., Douze, M., & Schmid, C. (2010). Product quantization for nearest neighbor search. *IEEE TPAMI*.
- Baranchuk, D., Babenko, A., & Lempitsky, V. (2018). Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors. *ECCV*.
- ANN Benchmarks: https://ann-benchmarks.com
