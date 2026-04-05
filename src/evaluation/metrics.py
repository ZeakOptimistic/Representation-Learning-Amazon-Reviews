"""Metric helpers."""

from sklearn.metrics import accuracy_score, f1_score


def classification_summary(y_true, y_pred) -> dict:
    """Return a compact summary for classification runs."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
