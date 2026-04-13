"""TF-IDF feature utilities."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(
    max_features: int = 100_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Return a configured TF-IDF vectorizer.

    Parameters
    ----------
    max_features:
        Vocabulary size cap.  Set to ``None`` to keep the full vocabulary.
    ngram_range:
        Range of n-gram sizes to include. ``(1, 2)`` means unigrams and bigrams.
    min_df:
        Minimum document frequency for a term to be kept.
    max_df:
        Maximum document frequency ratio for a term to be kept.
        Terms that appear in more than this fraction of documents are dropped.
    sublinear_tf:
        If ``True`` apply ``1 + log(tf)`` in place of raw term frequency.
        This dampens the outsized influence of terms that repeat many times
        in a single long review and consistently improves downstream
        classification on review-length text.  Defaults to ``True``.

    Notes on parallelism (Intel Core Ultra 9 285H, 16 cores)
    ---------------------------------------------------------
    ``TfidfVectorizer`` itself is single-threaded during ``fit_transform``.
    Parallelism kicks in at the *consumer* level:

    * ``LogisticRegression`` / ``LinearSVC`` inside a ``Pipeline`` will
      inherit the joblib thread pool — call
      ``joblib.parallel_config(n_jobs=12)`` in the notebook before the
      ``fit`` call, or pass ``n_jobs`` directly to the classifier.
    * Sparse-matrix operations (e.g. ``cosine_similarity`` on a
      100 k-feature matrix) are multi-threaded via BLAS/MKL automatically.

    Returns
    -------
    TfidfVectorizer
        Unfitted scikit-learn vectorizer ready for ``fit_transform``.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=sublinear_tf,
        dtype=np.float32,   # halves memory (vs float64) with no quality loss
    )
