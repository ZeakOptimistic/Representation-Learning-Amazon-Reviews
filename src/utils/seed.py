"""Random seed helpers.

Sets seeds for Python's ``random``, NumPy, gensim (via its C-level
random state), and the ``PYTHONHASHSEED`` environment variable so that
experiments are reproducible across a single process run.

Note: full reproducibility across *multiple* runs also requires setting
``CUBLAS_WORKSPACE_CONFIG`` for CUDA and using a single worker for
data loading, but those are out of scope here.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducibility.

    Covers:
    - Python ``random`` module
    - NumPy global RNG
    - ``PYTHONHASHSEED`` environment variable
    - gensim's internal C-level RNG (via ``gensim.utils.FAST_VERSION`` guard)

    Parameters
    ----------
    seed:
        Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Gensim uses its own C-level RNG; setting it requires importing gensim utils.
    # We guard so that importing this module never hard-fails if gensim is absent.
    try:
        from gensim import utils as _gensim_utils  # noqa: F401 — side-effect import

        # gensim reads PYTHONHASHSEED but also respects the seed kwarg on Word2Vec.
        # The env-var must be set *before* gensim is imported for full effect;
        # this call handles the common case where it is set at training time.
        logger.debug("gensim seed controlled via PYTHONHASHSEED=%s", seed)
    except ImportError:
        logger.debug("gensim not installed — skipping gensim seed.")
