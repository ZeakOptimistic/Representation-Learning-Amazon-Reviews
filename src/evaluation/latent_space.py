"""Placeholder entrypoint for latent-space outputs."""

from pathlib import Path


def main() -> None:
    out_dir = Path("reports/figures/generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "README_LATENT_SPACE.md"
    marker.write_text(
        "TODO: implement UMAP/T-SNE exports and latent-space summaries.\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote placeholder to {marker}")


if __name__ == "__main__":
    main()
