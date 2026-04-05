"""Simple text helpers."""

import re


def collapse_whitespace(text: str) -> str:
    """Normalize repeated whitespace."""
    return re.sub(r"\s+", " ", text).strip()
