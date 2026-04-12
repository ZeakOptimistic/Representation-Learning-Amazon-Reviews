"""Text cleaning utilities for Amazon review text.

Pre-compiles all regex patterns at module load time so they are not
re-compiled on every call when processing millions of rows.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Pre-compiled patterns
# ---------------------------------------------------------------------------
_HTML_TAG = re.compile(r"<[^>]+>")
_NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lightweight text cleaning for Amazon review content.

    Steps applied in order:
    1. Lowercase
    2. Strip HTML tags
    3. Remove non-alphanumeric characters (keep spaces)
    4. Collapse repeated whitespace

    Stopwords are intentionally kept so that TF-IDF ``min_df``/``max_df``
    and Word2Vec ``min_count`` can handle vocabulary control on their own terms.

    Parameters
    ----------
    text:
        Raw review string. Non-string values return an empty string.

    Returns
    -------
    str
        Cleaned, lowercased, whitespace-normalised text.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _HTML_TAG.sub(" ", text)
    text = _NON_ALPHANUM.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text
