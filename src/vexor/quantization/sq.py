"""Scalar quantization: compress float32 vectors to int8 (4× memory reduction)."""

from __future__ import annotations
import numpy as np


class ScalarQuantizer:
    """Per-dimension min/max quantization from float32 to int8."""

    def __init__(self) -> None:
        self._mins: np.ndarray | None = None
        self._scales: np.ndarray | None = None
        self._is_trained = False

    def train(self, vectors: np.ndarray) -> None:
        self._mins = vectors.min(axis=0).astype(np.float32)
        maxs = vectors.max(axis=0).astype(np.float32)
        ranges = maxs - self._mins
        ranges = np.where(ranges == 0, 1.0, ranges)
        self._scales = ranges / 254.0
        self._is_trained = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Call train() first.")
        scaled = (vectors - self._mins) / self._scales
        return np.clip(scaled, -127, 127).astype(np.int8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Call train() first.")
        return codes.astype(np.float32) * self._scales + self._mins

    def approximate_distance(self, query_code: np.ndarray, db_codes: np.ndarray) -> np.ndarray:
        """Squared L2 in int8 space. Shape: (N,)."""
        diff = db_codes.astype(np.int16) - query_code.astype(np.int16)
        return np.einsum("ij,ij->i", diff, diff).astype(np.float32)
