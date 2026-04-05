"""Placeholder entrypoint for baseline feature generation."""

from pathlib import Path


def main() -> None:
    out_dir = Path("artifacts/vectors")
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "README_BASELINES.md"
    marker.write_text(
        "TODO: implement TF-IDF and LSA vector generation.\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote placeholder to {marker}")


if __name__ == "__main__":
    main()
