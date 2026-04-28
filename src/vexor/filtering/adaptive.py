"""Selectivity-aware parameter scaling for HNSW ef and IVF nprobe."""

from __future__ import annotations
from pyroaring import BitMap


_OVERSAMPLE_FACTOR = 1.2
_MIN_SELECTIVITY = 1e-6


def adaptive_ef(k: int, ef_base: int, filter_bitmap: BitMap, total_vectors: int) -> int:
    """
    Scale ef_search based on filter selectivity.

    When only a small fraction of vectors match the filter, the graph
    traversal needs a larger candidate pool to guarantee k valid results.
    """
    if total_vectors == 0:
        return ef_base
    selectivity = len(filter_bitmap) / total_vectors
    selectivity = max(selectivity, _MIN_SELECTIVITY)
    return max(ef_base, int(k / selectivity * _OVERSAMPLE_FACTOR))


def adaptive_nprobe(
    base_nprobe: int,
    nlist: int,
    nearest_centroid_dist: float,
    mean_centroid_dist: float,
) -> int:
    """
    Scale nprobe based on query difficulty.

    A query deep inside a cluster (small residual) needs fewer probes.
    A query between clusters (large residual) needs more.
    """
    if mean_centroid_dist == 0.0:
        return base_nprobe
    scale = nearest_centroid_dist / mean_centroid_dist
    return min(nlist, int(base_nprobe * scale * 1.5))
