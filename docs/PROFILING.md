# Vexor Profiling & Optimization Workflow

This document captures a repeatable method to identify and fix performance bottlenecks.

## 1) Baseline benchmark first

Run a representative benchmark before profiling:

```bash
python bench/real_benchmark.py --dataset synthetic --n 20000 --dim 128 --queries 200
```

Record:
- Recall@K
- QPS
- P50/P95 latency
- Build time
- RAM delta

Keep this JSON as your baseline artifact.

## 2) cProfile search path

```bash
python bench/profile_search.py --index hnsw --n 50000 --dim 128 --queries 1000 --k 10
```

Outputs:
- `bench/results/search_profile.prof`
- `bench/results/search_profile.txt`

Focus on cumulative time in:
- distance kernels
- candidate heap updates
- graph traversal loops
- filter checks

## 3) py-spy flamegraph (recommended)

Install:

```bash
pip install py-spy
```

Capture:

```bash
py-spy record -o bench/results/flamegraph.svg -- python bench/profile_search.py --index hnsw
```

Use the flamegraph to validate where wall-time is really spent.

## 4) Memory hotspot checks

Use:
- benchmark RAM delta from `real_benchmark.py`
- `tracemalloc` snapshots for allocation-heavy paths
- memmap load benchmark:

```bash
python bench/mmap_startup_bench.py --n 200000 --dim 128
```

## 5) Optimization log template

For each optimization, document:
- Hypothesis
- Change
- Before/after metrics
- Trade-offs

Example:

```
Hypothesis: HNSW search spends too much time in Python distance dispatch.
Change: switched to JIT distance kernels in traversal hot loop.
Result: QPS +32%, P95 -24% on synthetic N=100k D=128.
Trade-off: increased JIT warmup on first query.
```

## 6) What to include in writeup

- Methodology and dataset details
- Hardware and runtime environment
- Bottlenecks found (with evidence)
- Fixes applied and impact
- Remaining bottlenecks / next steps

This level of rigor is usually more important than raw speed numbers in interviews.
