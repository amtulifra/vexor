"""Tests for benchmark harness utilities."""

from __future__ import annotations

import argparse
import numpy as np

from bench.real_benchmark import _load_fvecs, _recall_at_k, load_dataset


def test_recall_at_k_helper():
    got = [[1, 2, 3], [4, 5, 6]]
    gt = [{1, 9, 10}, {4, 8, 7}]
    r = _recall_at_k(got, gt, k=3)
    assert abs(r - (2 / 6)) < 1e-12


def test_load_dataset_synthetic():
    args = argparse.Namespace(
        dataset="synthetic",
        n=100,
        dim=16,
        queries=10,
        seed=42,
        sift_base="",
        sift_query="",
        glove="",
    )
    d = load_dataset(args)
    assert d.base.shape == (100, 16)
    assert d.query.shape == (10, 16)


def test_load_fvecs_roundtrip(tmp_path):
    # fvecs format: [dim(int32), values(float32 interpreted as int32 bytes)] repeated
    dim = 4
    vecs = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    rows = []
    for v in vecs:
        row = np.empty(dim + 1, dtype=np.int32)
        row[0] = dim
        row[1:] = v.view(np.int32)
        rows.append(row)
    data = np.concatenate(rows).astype(np.int32)
    path = tmp_path / "toy.fvecs"
    data.tofile(path)

    loaded = _load_fvecs(str(path))
    assert loaded.shape == vecs.shape
    np.testing.assert_allclose(loaded, vecs)
