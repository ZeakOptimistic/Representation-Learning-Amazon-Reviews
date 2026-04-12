"""Document embedding helpers.

Provides mean-pooling utilities that convert per-token vectors into a
single fixed-size document vector.  Only tokens present in the vocabulary
contribute to the pool; OOV tokens are silently skipped so that zero-vectors
are returned only when *no* in-vocabulary token is found.
"""

from __future__ import annotations

import numpy as np
from gensim.models.keyedvectors import KeyedVectors


def mean_pool_embeddings(
    token_vectors: list[np.ndarray],
    vector_size: int | None = None,
) -> np.ndarray:
    """Pool a list of token vectors by arithmetic mean.

    Keeps output shape stable so callers never have to check for empty returns.

    Parameters
    ----------
    token_vectors:
        List of 1-D float arrays, all with the same length.
    vector_size:
        Expected embedding dimension.  Required when *token_vectors* is empty
        so a correctly-shaped zero vector can be returned.

    Returns
    -------
    np.ndarray, dtype=float32
        Mean-pooled vector of shape ``(vector_size,)``, or a zero vector
        when *token_vectors* is empty.
    """
    if not token_vectors:
        if vector_size is None:
            return np.array([], dtype=np.float32)
        return np.zeros(vector_size, dtype=np.float32)

    pooled = np.mean(np.stack(token_vectors, axis=0), axis=0)
    return pooled.astype(np.float32, copy=False)


def mean_pool_from_tokens(
    tokens: list[str],
    keyed_vectors: KeyedVectors,
    vector_size: int,
) -> np.ndarray:
    """Map tokens to vectors and mean-pool only the in-vocabulary tokens.

    OOV tokens are skipped without warning.  When *all* tokens are OOV (or
    *tokens* is empty) a zero vector of shape ``(vector_size,)`` is returned.

    Parameters
    ----------
    tokens:
        List of string tokens from a single document.
    keyed_vectors:
        Gensim ``KeyedVectors`` instance (e.g. ``model.wv``).
    vector_size:
        Embedding dimension used to construct zero vectors for empty docs.

    Returns
    -------
    np.ndarray, dtype=float32
        Document vector of shape ``(vector_size,)``.
    """
    vectors = [keyed_vectors[token] for token in tokens if token in keyed_vectors]
    return mean_pool_embeddings(vectors, vector_size=vector_size)
