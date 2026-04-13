"""Latent-space analysis utilities.

This module provides reusable helpers that back the analysis performed
interactively in ``notebooks/06_latent_space_analysis.ipynb``.  Extracting
the core logic here makes it easy to re-run experiments from the command line
or to call individual functions from a report-generation script.

Main entry points
-----------------
* :func:`neighbourhood_purity`  — category purity of kNN neighbourhoods
* :func:`centroid_similarity`   — pairwise cosine similarity of class centroids
* :func:`isolation_forest_outliers` — anomaly scores + top-k outlier rows
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Neighbourhood purity
# ---------------------------------------------------------------------------

def neighbourhood_purity(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    query_limit: Optional[int] = 1000,
    seed: int = 42,
) -> tuple[float, pd.DataFrame]:
    """Compute nearest-neighbour category purity for each class.

    For each sampled query point the fraction of its *k* nearest neighbours
    (excluding itself) that share its label is called the *neighbourhood
    purity*.

    Parameters
    ----------
    X:
        L2-normalised document embeddings, shape ``(n_samples, n_features)``.
    labels:
        1-D array of class labels, length ``n_samples``.
    k:
        Number of nearest neighbours.
    query_limit:
        Max number of query points *per class*.  ``None`` means all.
    seed:
        Random seed for reproducible sampling.

    Returns
    -------
    mean_purity : float
        Average purity across all query points.
    per_class_df : pandas.DataFrame
        DataFrame with columns ``["class", "mean", "std", "n"]``.
    """
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    X_norm = normalize(X, norm="l2", axis=1)

    all_purities: list[float] = []
    rows = []

    for cls in unique_labels:
        mask = labels == cls
        idx_cls = np.where(mask)[0]
        if query_limit and len(idx_cls) > query_limit:
            idx_cls = rng.choice(idx_cls, size=query_limit, replace=False)

        purities = []
        for qi in idx_cls:
            q = X_norm[qi : qi + 1]                      # (1, d)
            sims = cosine_similarity(q, X_norm)[0]       # (n_samples,)
            sims[qi] = -2.0                               # exclude self
            top_k_idx = np.argsort(sims)[::-1][:k]
            purity = (labels[top_k_idx] == cls).mean()
            purities.append(purity)

        cls_purities = np.array(purities)
        all_purities.extend(purities)
        rows.append(
            {
                "class": cls,
                "mean": float(cls_purities.mean()),
                "std": float(cls_purities.std()),
                "n": len(purities),
            }
        )

    per_class_df = pd.DataFrame(rows).set_index("class")
    mean_purity = float(np.mean(all_purities))
    return mean_purity, per_class_df


# ---------------------------------------------------------------------------
# Centroid similarity
# ---------------------------------------------------------------------------

def centroid_similarity(
    X: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Compute pairwise cosine similarity between class centroids.

    Parameters
    ----------
    X:
        Document embeddings, shape ``(n_samples, n_features)``.
    labels:
        1-D array of class labels, length ``n_samples``.

    Returns
    -------
    pandas.DataFrame
        Square DataFrame indexed/columned by unique label values.
    """
    unique_labels = np.unique(labels)
    centroids = np.stack([X[labels == cls].mean(axis=0) for cls in unique_labels])
    sim_matrix = cosine_similarity(centroids)
    return pd.DataFrame(sim_matrix, index=unique_labels, columns=unique_labels)


def most_similar_pair(sim_df: pd.DataFrame) -> tuple[str, str, float]:
    """Return the pair of classes with the highest off-diagonal cosine sim.

    Parameters
    ----------
    sim_df:
        Square similarity DataFrame (output of :func:`centroid_similarity`).

    Returns
    -------
    class_a, class_b, similarity : str, str, float
    """
    # ── correct approach: mask diagonal with -inf, then argmax ──────────────
    values = sim_df.values.copy()
    np.fill_diagonal(values, -1.0)          # excludes self-sim of 1.0
    idx = np.unravel_index(np.argmax(values), values.shape)
    return (
        str(sim_df.index[idx[0]]),
        str(sim_df.columns[idx[1]]),
        float(values[idx]),
    )


# ---------------------------------------------------------------------------
# Isolation-Forest outlier detection
# ---------------------------------------------------------------------------

def isolation_forest_outliers(
    X: np.ndarray,
    df: pd.DataFrame,
    top_k: int = 20,
    contamination: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Detect the most anomalous rows using an Isolation Forest.

    Parameters
    ----------
    X:
        Document embeddings, shape ``(n_samples, n_features)``.
    df:
        DataFrame aligned with *X* (same row order).  Must contain at most
        ``len(X)`` rows.
    top_k:
        Number of top outliers to return.
    contamination:
        Expected fraction of anomalous points (sklearn default 0.1).
    seed:
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Rows from *df* corresponding to the *top_k* most anomalous points,
        with an extra ``anomaly_score`` column (more negative = more anomalous).
    """
    from sklearn.ensemble import IsolationForest  # lazy import (optional dep)

    iso = IsolationForest(contamination=contamination, random_state=seed)
    scores = iso.fit(X).score_samples(X)    # negative; lower = more anomalous

    top_idx = np.argsort(scores)[:top_k]
    result = df.iloc[top_idx].copy()
    result["anomaly_score"] = scores[top_idx]
    return result


# ---------------------------------------------------------------------------
# CLI entry point (generates a placeholder marker for DVC compat)
# ---------------------------------------------------------------------------

def main() -> None:
    from pathlib import Path

    out_dir = Path("reports/figures/generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "README_LATENT_SPACE.md"
    marker.write_text(
        "# Latent-space outputs\n\n"
        "Figures and CSV summaries are generated by\n"
        "`notebooks/06_latent_space_analysis.ipynb`.\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote marker to {marker}")


if __name__ == "__main__":
    main()
