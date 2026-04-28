# Vexor Design Notes

## Architecture

Vexor is organized as a stack of layers:

```
VectorDB (db.py)          ← unified API entry point
    ├── indexes/           ← FlatIndex, KDTreeIndex, HNSWIndex, IVFIndex, IVFPQIndex
    ├── filtering/         ← BitmapIndex (pyroaring), adaptive ef/nprobe
    ├── quantization/      ← ProductQuantizer, ScalarQuantizer
    ├── distance/          ← NumPy + Numba JIT kernels
    ├── concurrency/       ← NodeLockRegistry, ShardedIndex
    ├── storage/           ← VectorStore, binary format, WAL
    └── hooks/             ← VexorHook, NoopHook, StreamlitHook
```

## Key Design Decisions

### In-Graph Metadata Filtering (HNSW)

Standard vector DB implementations post-filter: search, then discard non-matching results.
At low selectivity (e.g. 5% of vectors match a predicate), this can return fewer than k
results for k=10, which is a correctness bug.

Vexor's fix: pass the filter bitmap into `_search_layer`. Skip expanding any node whose
ID is not set in the bitmap. Dynamically scale `ef` based on filter selectivity so the
candidate pool remains large enough to find k valid results:

```
ef_used = max(ef_base, int(k / selectivity * 1.2))
```

### Online IVF Centroids

Static centroids trained once on an initial dataset drift as new vectors arrive.
Vexor uses mini-batch gradient descent to update the nearest centroid after each insert:

```
lr = lr_0 / (1 + t * decay)
centroid[c] += lr * (vec - centroid[c])
```

The learning rate decays over time to prevent instability with large indexes.

### Deletion with Graph Repair

Tombstoning (marking deleted nodes and skipping them at search time) leaves ghost nodes
in the graph that degrade traversal and waste memory. Vexor's repair algorithm:

1. For each layer where node X appears, collect X's neighbors A, B, C...
2. For each pair (A, B) not already connected: add a direct edge, compensating for
   the path that ran through X.
3. When deleted nodes exceed 20% of the index, compact: rebuild from non-deleted vectors.

### Per-Node Locking

Global index locks serialize all inserts under multi-threaded workloads. Vexor's
per-node RW lock registry acquires locks only for the specific nodes being modified
(the new node and its M neighbors). Readers hold no locks — graph traversal is
read-only and safe for concurrent access.

### WAL + Crash Recovery

Before any insert is applied to the index, it is written to a WAL file with a CRC32
checksum. On startup, if a WAL file exists, replay it against the last snapshot.
After a successful save (which writes the full index), truncate the WAL.

## Visualization Philosophy

Every algorithm ships with a live visualizer. The visualizer is the development tool —
not a demo. The hook interface is the bridge:

- Production: `VexorHook` — all methods are no-ops (zero overhead)
- Viz mode: `StreamlitHook` — appends events to `st.session_state`

Core algorithms never know they are being visualized.
