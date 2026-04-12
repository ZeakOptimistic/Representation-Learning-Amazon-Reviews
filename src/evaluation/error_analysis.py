"""Error-analysis helpers.

Utilities for inspecting misclassified examples after a linear probe
evaluation.  These are used by Phase 5 to build the failure-case gallery
required by the master plan.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def top_errors(labels) -> Counter:
    """Count the most common labels in an error slice.

    Parameters
    ----------
    labels:
        Iterable of label values (strings or ints).

    Returns
    -------
    Counter
        ``{label: count}`` sorted by count descending.
    """
    return Counter(labels)


def misclassified_examples(
    df: pd.DataFrame,
    y_true,
    y_pred,
    text_column: str = "cleaned_text",
    n: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a sample of misclassified rows with true and predicted labels.

    Useful for building the failure-case gallery required by Phase 5 of the
    master plan.  Rows are sampled randomly from the full error set so the
    gallery is not dominated by a single error type.

    Parameters
    ----------
    df:
        The DataFrame whose rows correspond to *y_true* and *y_pred*.
    y_true:
        Ground-truth labels (array-like, same length as ``len(df)``).
    y_pred:
        Predicted labels (array-like, same length as ``len(df)``).
    text_column:
        Column in *df* containing the review text to include in the output.
    n:
        Maximum number of error rows to return.
    seed:
        Random seed for sampling.

    Returns
    -------
    pd.DataFrame
        Columns: ``[text_column, "true_label", "predicted_label"]``.
        Length ≤ *n*.  Empty if there are no errors.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    error_mask = y_true_arr != y_pred_arr

    if not error_mask.any():
        return pd.DataFrame(columns=[text_column, "true_label", "predicted_label"])

    error_indices = np.where(error_mask)[0]
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(
        error_indices, size=min(n, len(error_indices)), replace=False
    )
    sample_idx.sort()

    result = df.iloc[sample_idx][[text_column]].copy()
    result["true_label"] = y_true_arr[sample_idx]
    result["predicted_label"] = y_pred_arr[sample_idx]
    return result.reset_index(drop=True)


def confusion_pairs(y_true, y_pred, top_n: int = 10) -> pd.DataFrame:
    """Return the most common (true_label, predicted_label) confusion pairs.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    top_n:
        Number of most-frequent confusion pairs to return.

    Returns
    -------
    pd.DataFrame
        Columns: ``["true_label", "predicted_label", "count"]``
        sorted by *count* descending.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    error_mask = y_true_arr != y_pred_arr

    pairs = list(zip(y_true_arr[error_mask], y_pred_arr[error_mask]))
    counts = Counter(pairs)

    rows = [
        {"true_label": t, "predicted_label": p, "count": c}
        for (t, p), c in counts.most_common(top_n)
    ]
    return pd.DataFrame(rows)
