"""
Deletion tests: recall must remain stable after deleting and reinserting vectors.
"""

import pytest
import numpy as np

from vexor.indexes.hnsw import HNSWIndex
from vexor.indexes.flat import FlatIndex


N = 500
DIM = 32
K = 10
N_QUERIES = 20
DELETE_FRACTION = 0.30

rng = np.random.default_rng(7)
_VECS = rng.standard_normal((N, DIM)).astype(np.float32)
_QUERIES = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)


def _recall(results, ground_truth):
    hits = sum(len(r & g) for r, g in zip(results, ground_truth))
    return hits / (len(ground_truth) * K)


class TestHNSWDeletion:
    def test_deleted_ids_not_in_results(self):
        hnsw = HNSWIndex(DIM, M=16, ef_construction=200, ef_search=100)
        for v in _VECS:
            hnsw.add(v)

        n_delete = int(N * DELETE_FRACTION)
        to_delete = list(range(n_delete))
        for vid in to_delete:
            hnsw.delete(vid)

        deleted_set = set(to_delete)
        for q in _QUERIES:
            results = hnsw.search(q, k=K)
            for vid, _ in results:
                assert vid not in deleted_set, f"Deleted vector {vid} appeared in search results"

    def test_recall_after_deletion_and_reinsert(self):
        hnsw = HNSWIndex(DIM, M=16, ef_construction=200, ef_search=100)
        flat = FlatIndex(metric="l2")

        for v in _VECS:
            hnsw.add(v)
            flat.add(v)

        gt_before = [{r[0] for r in flat.search(q, k=K)} for q in _QUERIES]
        recall_before = _recall([{r[0] for r in hnsw.search(q, k=K)} for q in _QUERIES], gt_before)

        n_delete = int(N * DELETE_FRACTION)
        to_delete = list(range(n_delete))
        for vid in to_delete:
            hnsw.delete(vid)

        for vid in to_delete:
            new_id = hnsw.add(_VECS[vid])

        gt_after = [{r[0] for r in flat.search(q, k=K)} for q in _QUERIES]
        recall_after = _recall([{r[0] for r in hnsw.search(q, k=K)} for q in _QUERIES], gt_after)

        assert recall_after >= recall_before - 0.15, (
            f"Recall dropped too much after delete+reinsert: {recall_before:.3f} → {recall_after:.3f}"
        )

    def test_active_size_decrements_on_delete(self):
        hnsw = HNSWIndex(DIM, M=8)
        for v in _VECS[:50]:
            hnsw.add(v)

        assert hnsw.active_size == 50
        hnsw.delete(0)
        assert hnsw.active_size == 49
        hnsw.delete(1)
        assert hnsw.active_size == 48

    def test_double_delete_is_safe(self):
        hnsw = HNSWIndex(DIM, M=8)
        for v in _VECS[:20]:
            hnsw.add(v)
        hnsw.delete(0)
        hnsw.delete(0)
        assert hnsw.active_size == 19
