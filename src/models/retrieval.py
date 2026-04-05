"""Retrieval helpers."""

from sklearn.metrics.pairwise import cosine_similarity


def pairwise_cosine(matrix):
    """Compute pairwise cosine similarity."""
    return cosine_similarity(matrix)
