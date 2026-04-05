"""Linear probe utilities."""

from sklearn.linear_model import LogisticRegression


def build_logistic_probe() -> LogisticRegression:
    """Return a default linear classifier."""
    return LogisticRegression(max_iter=2000, n_jobs=None)
