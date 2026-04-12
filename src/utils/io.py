"""Filesystem and serialisation helpers."""

from __future__ import annotations

import json
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and all parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: object, path: str | Path, *, indent: int = 2) -> Path:
    """Serialise *data* to a JSON file with UTF-8 encoding.

    Parameters
    ----------
    data:
        Any JSON-serialisable object.
    path:
        Destination file path. Parent directories are created automatically.
    indent:
        Pretty-print indentation level (default 2).

    Returns
    -------
    Path
        The resolved path of the written file.
    """
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    return path


def load_json(path: str | Path) -> object:
    """Load and parse a JSON file with UTF-8 encoding.

    Parameters
    ----------
    path:
        Source file path.

    Returns
    -------
    object
        Parsed JSON value (dict, list, scalar, etc.).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))
