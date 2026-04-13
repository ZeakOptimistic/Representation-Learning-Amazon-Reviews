"""Linear probe utilities.

Provides factory functions for light, frozen-feature classifiers used to
evaluate the downstream utility of each representation.  Two probes are
available:

- **Logistic Regression** — fast, well-calibrated; the primary probe.
- **Linear SVM** — harder-margin alternative; useful when logistic regression
  diverges on a particular feature space.

Both probes wrap the classifier in a ``Pipeline`` with
``StandardScaler(with_mean=False)`` so they work correctly for both dense
arrays (Word2Vec, LSA) and sparse matrices (TF-IDF) without requiring the
caller to pre-scale features.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_logistic_probe(C: float = 1.0, max_iter: int = 2000) -> Pipeline:
    """Return a logistic regression linear probe with standard scaling.

    Uses the ``lbfgs`` solver (sklearn default for small-to-medium datasets),
    which handles multi-class natively via softmax.  For very large feature
    spaces (>50 k features, 1 M+ samples) consider switching to
    ``solver='saga'`` and enabling ``n_jobs=-1`` to use all 16 cores.

    Parameters
    ----------
    C:
        Inverse of regularisation strength.  Smaller values mean stronger
        regularisation.  Default ``1.0`` (sklearn default).
    max_iter:
        Maximum iterations for the L-BFGS solver.  2000 is sufficient for
        most representation-size/dataset combinations here.

    Returns
    -------
    Pipeline
        A fitted-ready pipeline: ``StandardScaler -> LogisticRegression``.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(C=C, max_iter=max_iter)),
        ]
    )


def build_linear_svm_probe(C: float = 0.1, max_iter: int = 2000) -> Pipeline:
    """Return a linear SVM probe with standard scaling.

    Use this as a fallback when logistic regression does not converge or
    when you need a harder-margin decision boundary.

    ``LinearSVC`` is generally faster than ``SVC(kernel='linear')`` and scales
    well to large feature spaces (TF-IDF with 50k-100k features).

    Parameters
    ----------
    C:
        Regularisation parameter.  Smaller values mean stronger regularisation.
        Default ``0.1`` -- slightly tighter than LogisticRegression's ``1.0``
        because SVM margins are on a different scale.
    max_iter:
        Maximum iterations for the dual coordinate descent solver.

    Returns
    -------
    Pipeline
        A fitted-ready pipeline: ``StandardScaler -> LinearSVC``.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LinearSVC(C=C, max_iter=max_iter, dual="auto")),
        ]
    )
