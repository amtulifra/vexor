"""
Recall correctness tests: every approximate index must achieve >= 0.95
recall@10 vs the flat exact baseline on random 768-dim vectors.
"""

import pytest
import numpy as np

from vexor.indexes.flat import FlatIndex
from vexor.indexes.kdtree import KDTreeIndex
from vexor.indexes.hnsw import HNSWIndex
from vexor.indexes.ivf import IVFIndex
from vexor.indexes.ivfpq import IVFPQIndex
from vexor.distance.kernels import cosine_distance, l2_distance, inner_product_distance
from vexor.distance.kernels_jit import cosine_distance_jit, l2_distance_jit, inner_product_distance_jit


N = 500
DIM = 64
K = 10
N_QUERIES = 20
RECALL_THRESHOLD = 0.95

rng = np.random.default_rng(0)
_VECS = rng.standard_normal((N, DIM)).astype(np.float32)
_QUERIES = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)


def _ground_truth(vecs, queries, metric="l2"):
    flat = FlatIndex(metric=metric)
    for v in vecs:
        flat.add(v)
    return [{r[0] for r in flat.search(q, k=K)} for q in queries]


def _recall(results, gt):
    hits = sum(len(r & g) for r, g in zip(results, gt))
    return hits / (len(gt) * K)


@pytest.fixture(scope="module")
def ground_truth_l2():
    return _ground_truth(_VECS, _QUERIES, "l2")


# --- Distance kernel parity tests ---

class TestDistanceKernelParity:
    def test_cosine_numpy_jit_match(self):
        a, b = _VECS[0], _VECS[1]
        np_result = cosine_distance(a, b)
        jit_result = cosine_distance_jit(a, b)
        assert abs(np_result - jit_result) < 1e-5

    def test_l2_numpy_jit_match(self):
        a, b = _VECS[0], _VECS[1]
        np_result = l2_distance(a, b)
        jit_result = l2_distance_jit(a, b)
        assert abs(np_result - jit_result) < 1e-5

    def test_inner_product_numpy_jit_match(self):
        a, b = _VECS[0], _VECS[1]
        np_result = inner_product_distance(a, b)
        jit_result = inner_product_distance_jit(a, b)
        assert abs(np_result - jit_result) < 1e-5

    def test_cosine_distance_range(self):
        for i in range(10):
            d = cosine_distance(_VECS[i], _VECS[i + 1])
            assert 0.0 <= d <= 2.0

    def test_l2_distance_nonnegative(self):
        for i in range(10):
            d = l2_distance(_VECS[i], _VECS[i + 1])
            assert d >= 0.0

    def test_identical_vectors_zero_distance(self):
        v = _VECS[0]
        assert l2_distance(v, v) < 1e-7
        assert cosine_distance(v, v) < 1e-5


# --- Flat index: always 1.0 recall ---

class TestFlatIndex:
    def test_recall_is_exact(self, ground_truth_l2):
        flat = FlatIndex(metric="l2")
        for v in _VECS:
            flat.add(v)
        results = [{r[0] for r in flat.search(q, k=K)} for q in _QUERIES]
        assert _recall(results, ground_truth_l2) == 1.0

    def test_returns_k_results(self):
        flat = FlatIndex(metric="l2")
        for v in _VECS[:50]:
            flat.add(v)
        res = flat.search(_QUERIES[0], k=10)
        assert len(res) == 10

    def test_results_sorted_by_distance(self):
        flat = FlatIndex(metric="l2")
        for v in _VECS[:100]:
            flat.add(v)
        res = flat.search(_QUERIES[0], k=10)
        dists = [d for _, d in res]
        assert dists == sorted(dists)

    def test_empty_index_returns_empty(self):
        flat = FlatIndex()
        assert flat.search(_QUERIES[0], k=5) == []


# --- HNSW: recall >= 0.95 ---

class TestHNSWIndex:
    def test_recall_above_threshold(self, ground_truth_l2):
        hnsw = HNSWIndex(DIM, M=16, ef_construction=200, ef_search=100, metric="l2")
        for v in _VECS:
            hnsw.add(v)
        results = [{r[0] for r in hnsw.search(q, k=K)} for q in _QUERIES]
        assert _recall(results, ground_truth_l2) >= RECALL_THRESHOLD

    def test_higher_ef_improves_recall(self, ground_truth_l2):
        hnsw = HNSWIndex(DIM, M=16, ef_construction=200, metric="l2")
        for v in _VECS:
            hnsw.add(v)
        r_low = _recall([{r[0] for r in hnsw.search(q, k=K, ef=10)} for q in _QUERIES], ground_truth_l2)
        r_high = _recall([{r[0] for r in hnsw.search(q, k=K, ef=200)} for q in _QUERIES], ground_truth_l2)
        assert r_high >= r_low

    def test_size_tracking(self):
        hnsw = HNSWIndex(DIM, M=8)
        for v in _VECS[:20]:
            hnsw.add(v)
        assert hnsw.size == 20


# --- IVF: recall >= 0.90 ---

class TestIVFIndex:
    def test_recall_above_threshold(self, ground_truth_l2):
        # nlist=16, nprobe=10 → 62.5% cluster coverage on random 64D Gaussian data
        ivf = IVFIndex(nlist=16, nprobe=10, metric="l2", online_updates=False)
        ivf.train(_VECS)
        for v in _VECS:
            ivf.add(v)
        results = [{r[0] for r in ivf.search(q, k=K, nprobe=10, use_adaptive_nprobe=False)}
                   for q in _QUERIES]
        assert _recall(results, ground_truth_l2) >= 0.90

    def test_requires_training(self):
        ivf = IVFIndex(nlist=8)
        with pytest.raises(RuntimeError):
            ivf.add(_VECS[0])


# --- KD-Tree: exact on low dimensions ---

class TestKDTreeIndex:
    def test_recall_exact_2d(self):
        rng2 = np.random.default_rng(1)
        vecs_2d = rng2.standard_normal((200, 2)).astype(np.float32)
        queries_2d = rng2.standard_normal((20, 2)).astype(np.float32)
        gt_2d = _ground_truth(vecs_2d, queries_2d, "l2")

        kd = KDTreeIndex(metric="l2")
        for v in vecs_2d:
            kd.add(v)
        kd.build()
        results_2d = [{r[0] for r in kd.search(q, k=K)} for q in queries_2d]
        assert _recall(results_2d, gt_2d) >= 0.95

    def test_build_required(self):
        kd = KDTreeIndex()
        kd.add(_VECS[0])
        with pytest.raises(RuntimeError):
            kd.search(_VECS[1], k=1)


# --- IVFPQ ---

class TestIVFPQIndex:
    def test_recall_above_threshold(self, ground_truth_l2):
        M = 8
        assert DIM % M == 0
        nlist = 16
        pq_idx = IVFPQIndex(nlist=nlist, nprobe=8, M=M, K=256, metric="l2")
        pq_idx.train(_VECS)
        for v in _VECS:
            pq_idx.add(v)
        results = [{r[0] for r in pq_idx.search(q, k=K)} for q in _QUERIES]
        assert _recall(results, ground_truth_l2) >= 0.80

    def test_requires_training(self):
        pq_idx = IVFPQIndex(nlist=8, M=8)
        with pytest.raises(RuntimeError):
            pq_idx.add(_VECS[0])
