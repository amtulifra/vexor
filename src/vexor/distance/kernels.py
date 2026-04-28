"""Pure NumPy distance kernels — correctness reference implementations."""

from __future__ import annotations
import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.dot(diff, diff))


def inner_product_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def batch_cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine distance from query to every row in matrix. Shape: (N,)."""
    dots = matrix @ query
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * query_norm
    denom = np.where(denom == 0.0, 1e-10, denom)
    return (1.0 - dots / denom).astype(np.float32)


def batch_l2(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Squared L2 distance from query to every row in matrix. Shape: (N,)."""
    diff = matrix - query
    return np.einsum("ij,ij->i", diff, diff).astype(np.float32)


def batch_inner_product(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Inner-product distance from query to every row in matrix. Shape: (N,)."""
    return (1.0 - matrix @ query).astype(np.float32)
