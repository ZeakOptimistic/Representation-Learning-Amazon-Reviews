"""Error-analysis helpers."""

from collections import Counter


def top_errors(labels) -> Counter:
    """Count the most common labels in an error slice."""
    return Counter(labels)
