"""Document embedding helpers."""

from __future__ import annotations
import numpy as np


def mean_pool_embeddings(token_vectors: list[np.ndarray]) -> np.ndarray:
    """Pool token vectors by mean. Returns zeros when no vectors exist."""
    if not token_vectors:
        return np.array([], dtype=float)
    return np.mean(np.stack(token_vectors, axis=0), axis=0)
