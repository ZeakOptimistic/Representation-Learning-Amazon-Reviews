"""Document embedding helpers."""

from __future__ import annotations
import numpy as np


def mean_pool_embeddings(
    token_vectors: list[np.ndarray],
    vector_size: int | None = None,
) -> np.ndarray:
    """Pool token vectors by mean and keep output shape stable when empty."""
    if not token_vectors:
        if vector_size is None:
            return np.array([], dtype=np.float32)
        return np.zeros(vector_size, dtype=np.float32)

    pooled = np.mean(np.stack(token_vectors, axis=0), axis=0)
    return pooled.astype(np.float32, copy=False)


def mean_pool_from_tokens(
    tokens: list[str],
    keyed_vectors,
    vector_size: int,
) -> np.ndarray:
    """Map tokens to vectors and mean-pool known tokens only."""
    vectors = [keyed_vectors[token] for token in tokens if token in keyed_vectors]
    return mean_pool_embeddings(vectors, vector_size=vector_size)
