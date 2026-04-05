"""Dataset preparation entrypoint.

This file is intentionally lightweight at project start.
Expand it after the sampling plan is frozen.
"""

from pathlib import Path


def main() -> None:
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    placeholder = out_dir / "README_PREPARED_DATASET.md"
    placeholder.write_text(
        "TODO: implement balanced sampling, joins, and split generation.\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote placeholder to {placeholder}")


if __name__ == "__main__":
    main()
