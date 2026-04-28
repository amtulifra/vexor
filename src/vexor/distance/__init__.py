from vexor.distance.kernels import cosine_distance, l2_distance, inner_product_distance
from vexor.distance.kernels import batch_cosine, batch_l2, batch_inner_product
from vexor.distance.kernels_jit import (
    cosine_distance_jit,
    l2_distance_jit,
    inner_product_distance_jit,
    batch_cosine_jit,
    batch_l2_jit,
    batch_inner_product_jit,
)

METRICS = {"cosine", "l2", "inner_product"}

__all__ = [
    "cosine_distance", "l2_distance", "inner_product_distance",
    "batch_cosine", "batch_l2", "batch_inner_product",
    "cosine_distance_jit", "l2_distance_jit", "inner_product_distance_jit",
    "batch_cosine_jit", "batch_l2_jit", "batch_inner_product_jit",
    "METRICS",
]
