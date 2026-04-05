"""Placeholder entrypoint for Word2Vec training."""

from pathlib import Path


def main() -> None:
    out_dir = Path("artifacts/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "README_WORD2VEC.md"
    marker.write_text(
        "TODO: implement Word2Vec training and saved vectors.\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote placeholder to {marker}")


if __name__ == "__main__":
    main()
