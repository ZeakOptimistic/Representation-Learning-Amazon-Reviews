"""Helpers for reading Amazon Reviews 2023 files."""

from pathlib import Path
from typing import Iterator
import json


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one JSON object at a time from a JSONL file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
