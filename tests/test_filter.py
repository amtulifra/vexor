"""
Filter correctness tests.

Vexor's in-graph filter must maintain recall >= 0.90 at 5% selectivity,
where naive post-filtering fails (returns <k results).
"""

import pytest
import numpy as np

from vexor.indexes.hnsw import HNSWIndex
from vexor.indexes.flat import FlatIndex
from vexor.filtering.bitmap import BitmapIndex


N = 1_000
DIM = 32
K = 10
N_QUERIES = 30

rng = np.random.default_rng(42)
_VECS = rng.standard_normal((N, DIM)).astype(np.float32)
_QUERIES = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)


def _build_labeled_index(selectivity: float):
    n_match = max(1, int(N * selectivity))
    labels = np.array(["match"] * n_match + ["other"] * (N - n_match))
    rng.shuffle(labels)

    hnsw = HNSWIndex(DIM, M=16, ef_construction=200, ef_search=100)
    flat = FlatIndex(metric="cosine")
    for i, v in enumerate(_VECS):
        meta = {"label": labels[i]}
        hnsw.add(v, meta)
        flat.add(v, meta)
    return hnsw, flat, labels


class TestBitmapIndex:
    def test_empty_filter_returns_all(self):
        bm = BitmapIndex()
        for i in range(100):
            bm.add(i, {"x": i % 5})
        result = bm.query(None)
        assert len(result) == 100

    def test_single_predicate(self):
        bm = BitmapIndex()
        for i in range(100):
            bm.add(i, {"color": "red" if i < 30 else "blue"})
        reds = bm.query({"color": "red"})
        assert len(reds) == 30
        assert all(i < 30 for i in reds)

    def test_and_predicate(self):
        bm = BitmapIndex()
        for i in range(100):
            bm.add(i, {"color": "red" if i % 2 == 0 else "blue", "size": "big" if i < 50 else "small"})
        result = bm.query({"color": "red", "size": "big"})
        expected = {i for i in range(100) if i % 2 == 0 and i < 50}
        assert set(result) == expected

    def test_remove(self):
        bm = BitmapIndex()
        for i in range(10):
            bm.add(i, {"x": "a"})
        bm.remove(5, {"x": "a"})
        result = bm.query({"x": "a"})
        assert 5 not in result
        assert len(result) == 9


class TestInGraphFilter:
    def test_100_percent_selectivity_matches_unfiltered(self):
        hnsw, flat, _ = _build_labeled_index(1.0)
        filt = {"label": "match"}
        for q in _QUERIES[:5]:
            res_filtered = {r[0] for r in hnsw.search(q, k=K, filter=filt)}
            res_unfiltered = {r[0] for r in hnsw.search(q, k=K)}
            assert len(res_filtered) == len(res_unfiltered)

    def test_recall_at_5_percent_selectivity(self):
        hnsw, flat, labels = _build_labeled_index(0.05)
        filt = {"label": "match"}
        hits = total = 0
        for q in _QUERIES:
            exact = {r[0] for r in flat.search(q, k=K, filter=filt)}
            approx = {r[0] for r in hnsw.search(q, k=K, filter=filt)}
            hits += len(exact & approx)
            total += max(len(exact), 1)
        recall = hits / total
        assert recall >= 0.75, f"Recall at 5% selectivity = {recall:.3f} (expected >= 0.75)"

    def test_recall_at_10_percent_selectivity(self):
        hnsw, flat, labels = _build_labeled_index(0.10)
        filt = {"label": "match"}
        hits = total = 0
        for q in _QUERIES:
            exact = {r[0] for r in flat.search(q, k=K, filter=filt)}
            approx = {r[0] for r in hnsw.search(q, k=K, filter=filt)}
            hits += len(exact & approx)
            total += max(len(exact), 1)
        recall = hits / total
        assert recall >= 0.80, f"Recall at 10% selectivity = {recall:.3f} (expected >= 0.80)"

    def test_filter_results_only_contain_matching_ids(self):
        hnsw, _, labels = _build_labeled_index(0.20)
        filt = {"label": "match"}
        for q in _QUERIES[:10]:
            results = hnsw.search(q, k=K, filter=filt)
            for vid, _ in results:
                assert labels[vid] == "match", f"Non-matching vector {vid} in filtered results"
