"""Evaluation metric helpers.

Provides three families of metrics used to compare representations:

1. **Classification** — macro-F1 and accuracy via a linear probe.
2. **Retrieval** — Recall@k and MRR@k using cosine similarity.
3. **Clustering** — NMI and ARI against ground-truth category / sentiment.

All functions return plain dicts so results are trivially serialisable and
can be assembled into a unified DataFrame in the calling notebook or script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classification_summary(y_true, y_pred) -> dict:
    """Return accuracy and macro-F1 for a single classification run.

    Parameters
    ----------
    y_true:
        Ground-truth labels (array-like of strings or ints).
    y_pred:
        Predicted labels of the same shape.

    Returns
    -------
    dict
        ``{"accuracy": float, "macro_f1": float}``
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieval_metrics(
    X_query: np.ndarray,
    X_db: np.ndarray,
    query_category: pd.Series,
    query_sentiment: pd.Series,
    db_category: pd.Series,
    db_sentiment: pd.Series,
    k: int = 10,
    query_limit: int = 1000,
    batch_size: int = 128,
) -> dict:
    """Compute Recall@k and MRR@k for category+sentiment joint relevance.

    A retrieved document is considered a *positive* match when it shares both
    the same ``main_category`` and ``sentiment`` as the query.  This is the
    Task C definition from the master plan.

    The function batches cosine similarity computation to avoid OOM on large
    databases, and pre-converts Pandas Series to NumPy arrays before the inner
    loop to avoid the per-row ``.iloc`` overhead.

    Parameters
    ----------
    X_query:
        Dense or sparse matrix of shape ``(n_queries, d)`` — the query vectors.
    X_db:
        Dense or sparse matrix of shape ``(n_db, d)`` — the retrieval database.
    query_category, query_sentiment:
        Label Series aligned with rows of *X_query*.
    db_category, db_sentiment:
        Label Series aligned with rows of *X_db*.
    k:
        Number of top candidates to retrieve per query.
    query_limit:
        Maximum number of query rows to evaluate (randomly takes the first
        *query_limit* rows to keep runtime manageable).
    batch_size:
        Number of query vectors to process per cosine-similarity batch.

    Returns
    -------
    dict
        ``{"recall_at_k": float, "mrr_at_k": float, "queries": int, "k": int}``
    """
    n_queries = min(len(query_category), query_limit)
    recall_hits = 0
    reciprocal_sum = 0.0

    # Pre-convert to numpy arrays — avoids per-row pandas .iloc overhead
    # inside the inner loop, which is the hot path.
    qc_arr = query_category.to_numpy()
    qs_arr = query_sentiment.to_numpy()
    db_c_arr = db_category.to_numpy()
    db_s_arr = db_sentiment.to_numpy()

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        sims = cosine_similarity(X_query[start:end], X_db)
        topk = np.argpartition(sims, -k, axis=1)[:, -k:]

        for i in range(end - start):
            q_idx = start + i
            candidates = topk[i]
            ranked = candidates[np.argsort(sims[i, candidates])[::-1]]

            positives = (db_c_arr[ranked] == qc_arr[q_idx]) & (
                db_s_arr[ranked] == qs_arr[q_idx]
            )

            if positives.any():
                recall_hits += 1
                first_rank = int(np.where(positives)[0][0]) + 1
                reciprocal_sum += 1.0 / first_rank

    return {
        f"recall_at_{k}": float(recall_hits / n_queries),
        f"mrr_at_{k}": float(reciprocal_sum / n_queries),
        "queries": int(n_queries),
        "k": k,
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def clustering_scores(
    X: np.ndarray,
    category_labels,
    sentiment_labels,
    n_clusters: int = 4,
    seed: int = 42,
) -> dict:
    """Cluster *X* with MiniBatchKMeans and score against ground-truth labels.

    Vectors are L2-normalised before clustering so that Euclidean distance
    in the KMeans objective is equivalent to cosine distance.  This is
    essential for mean-pooled Word2Vec and LSA embeddings where magnitude
    variation would otherwise dominate the partitioning.

    Parameters
    ----------
    X:
        Dense matrix of shape ``(n_samples, d)``.
    category_labels:
        Array-like of category strings aligned with rows of *X*.
    sentiment_labels:
        Array-like of sentiment strings aligned with rows of *X*.
    n_clusters:
        Number of clusters — should match the number of product categories
        (default 4).
    seed:
        Random seed for KMeans initialisation.

    Returns
    -------
    dict
        ``{"nmi_category", "ari_category", "nmi_sentiment", "ari_sentiment"}``
        All values are floats.
    """
    from sklearn.cluster import MiniBatchKMeans  # local import: optional dep

    X_norm = normalize(X)
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init="auto",
        batch_size=4096,
    )
    pred = km.fit_predict(X_norm)

    return {
        "nmi_category": float(normalized_mutual_info_score(category_labels, pred)),
        "ari_category": float(adjusted_rand_score(category_labels, pred)),
        "nmi_sentiment": float(normalized_mutual_info_score(sentiment_labels, pred)),
        "ari_sentiment": float(adjusted_rand_score(sentiment_labels, pred)),
    }
