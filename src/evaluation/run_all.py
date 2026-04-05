"""Placeholder entrypoint for evaluation runs."""

from pathlib import Path


def main() -> None:
    out_dir = Path("artifacts/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "README_METRICS.md"
    marker.write_text(
        "TODO: implement classification, retrieval, and clustering metrics.\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote placeholder to {marker}")


if __name__ == "__main__":
    main()
