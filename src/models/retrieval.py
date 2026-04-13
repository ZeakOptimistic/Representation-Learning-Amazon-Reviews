"""Retrieval helpers for the representation-learning evaluation pipeline.

This module consolidates the retrieval-specific utilities that are used across
the project.  The heavy lifting (nearest-neighbour search and metric
computation) lives in :mod:`src.evaluation.metrics`; this module provides
thin wrappers and a canonical re-export so that callers can import from a
single, well-named location.

Typical usage
-------------
::

    from src.models.retrieval import pairwise_cosine, build_query_index

    sim = pairwise_cosine(X_train)           # (N, N) float array
    index = build_query_index(X_train)       # FAISS / sklearn wrapper
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
    """Return the pairwise cosine-similarity matrix for *matrix*.

    Parameters
    ----------
    matrix:
        2-D array of shape ``(n_samples, n_features)``.

    Returns
    -------
    numpy.ndarray
        Square matrix of shape ``(n_samples, n_samples)`` with cosine
        similarities in ``[-1, 1]``.
    """
    return cosine_similarity(matrix)


def top_k_similar(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int = 10,
    exclude_self: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the indices and scores of the *k* most-similar corpus vectors.

    Parameters
    ----------
    query:
        1-D or 2-D array of shape ``(n_features,)`` or
        ``(n_queries, n_features)``.
    corpus:
        2-D array of shape ``(n_corpus, n_features)``.
    k:
        Number of top neighbours to return.
    exclude_self:
        If *True*, drop the corpus entry whose similarity equals exactly 1.0
        (i.e., the query itself when it is part of the corpus).

    Returns
    -------
    indices : numpy.ndarray
        Shape ``(k,)`` or ``(n_queries, k)`` — indices into *corpus*.
    scores : numpy.ndarray
        Corresponding cosine similarities.
    """
    query = np.atleast_2d(query)
    scores_matrix = cosine_similarity(query, corpus)   # (n_queries, n_corpus)

    results_idx = []
    results_scores = []
    for row in scores_matrix:
        if exclude_self:
            # Mask positions that are identical to the query (self-match)
            row = row.copy()
            row[row >= 1.0] = -2.0  # sentinel below any valid cosine value
        top = np.argsort(row)[::-1][:k]
        results_idx.append(top)
        results_scores.append(row[top])

    idx = np.array(results_idx)
    scr = np.array(results_scores)
    if idx.shape[0] == 1:
        return idx[0], scr[0]
    return idx, scr
