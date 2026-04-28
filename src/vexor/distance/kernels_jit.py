"""
Numba JIT distance kernels with SIMD auto-vectorization.

These are the production hot-path implementations. The NumPy variants in
kernels.py serve as correctness references.
"""

from __future__ import annotations
import math
import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True)
def cosine_distance_jit(a: np.ndarray, b: np.ndarray) -> float:
    dot = na = nb = 0.0
    for i in range(a.shape[0]):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return 1.0
    return 1.0 - dot / denom


@njit(fastmath=True, cache=True)
def l2_distance_jit(a: np.ndarray, b: np.ndarray) -> float:
    acc = 0.0
    for i in range(a.shape[0]):
        diff = a[i] - b[i]
        acc += diff * diff
    return acc


@njit(fastmath=True, cache=True)
def inner_product_distance_jit(a: np.ndarray, b: np.ndarray) -> float:
    dot = 0.0
    for i in range(a.shape[0]):
        dot += a[i] * b[i]
    return 1.0 - dot


@njit(fastmath=True, cache=True, parallel=True)
def batch_cosine_jit(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    out = np.empty(n, dtype=np.float32)
    qnorm = 0.0
    for d in range(query.shape[0]):
        qnorm += query[d] * query[d]
    qnorm = math.sqrt(qnorm)

    for i in prange(n):
        dot = rnorm = 0.0
        for d in range(query.shape[0]):
            dot += query[d] * matrix[i, d]
            rnorm += matrix[i, d] * matrix[i, d]
        denom = qnorm * math.sqrt(rnorm)
        out[i] = 1.0 - dot / denom if denom != 0.0 else 1.0
    return out


@njit(fastmath=True, cache=True, parallel=True)
def batch_l2_jit(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        acc = 0.0
        for d in range(query.shape[0]):
            diff = query[d] - matrix[i, d]
            acc += diff * diff
        out[i] = acc
    return out


@njit(fastmath=True, cache=True, parallel=True)
def batch_inner_product_jit(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        dot = 0.0
        for d in range(query.shape[0]):
            dot += query[d] * matrix[i, d]
        out[i] = 1.0 - dot
    return out
