"""LSA (Latent Semantic Analysis) feature utilities."""

from __future__ import annotations

from sklearn.decomposition import TruncatedSVD


def build_lsa(
    n_components: int = 300,
    n_iter: int = 7,
    random_state: int = 42,
) -> TruncatedSVD:
    """Return a configured Truncated SVD transformer for LSA.

    Parameters
    ----------
    n_components:
        Number of latent dimensions to keep.  Should match Word2Vec
        ``vector_size`` (default 300) for a fair representation comparison.
    n_iter:
        Number of power-iteration passes in the randomised SVD algorithm.
        Increasing from sklearn's default of 5 to 7 gives more accurate
        singular vectors on large, sparse TF-IDF matrices with only a small
        runtime cost.
    random_state:
        Seed for the randomised SVD algorithm.

    Notes on parallelism (Intel Core Ultra 9 285H, 16 cores)
    ---------------------------------------------------------
    ``TruncatedSVD`` has no ``n_jobs`` parameter.  However, the heavy work
    (randomised matrix multiplications) is done by NumPy, which delegates to
    MKL/OpenBLAS — both of which automatically use all available CPU cores.
    No extra flags are needed; parallelism is free here.

    Returns
    -------
    TruncatedSVD
        Unfitted scikit-learn transformer ready for ``fit_transform``.
    """
    return TruncatedSVD(
        n_components=n_components,
        n_iter=n_iter,
        random_state=random_state,
    )
