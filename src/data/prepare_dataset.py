"""Dataset preparation entrypoint.

Reads the interim parquet produced by the Colab data extraction notebook
(``data/interim/amazon_reviews_4cat_2m.parquet``), enforces the column
schema expected by all downstream scripts, and writes the three processed
split files::

    data/processed/train.parquet
    data/processed/val.parquet
    data/processed/test.parquet

This script is called by DVC's ``prepare_data`` stage::

    python -m src.data.prepare_dataset

The split is performed by ``parent_asin`` group so that all reviews for a
given product go into exactly one split (no leakage).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_CWD = Path.cwd().resolve()
_CANDIDATES = [_CWD, _CWD.parent]
PROJECT_ROOT = next(
    (p for p in _CANDIDATES if (p / "src").exists() and (p / "configs").exists()),
    _CWD,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import split_by_asin  # noqa: E402
from src.data.text_processing import clean_text  # noqa: E402
from src.utils.io import ensure_dir, save_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Columns that every downstream script expects.
REQUIRED_OUTPUT_COLUMNS = [
    "rating",
    "text",
    "parent_asin",
    "main_category",
    "sentiment",
    "cleaned_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=PROJECT_ROOT / "params.yaml")
    parser.add_argument(
        "--interim",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "amazon_reviews_4cat_2m.parquet",
        help="Path to the interim combined parquet file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Output directory for train/val/test parquet files.",
    )
    return parser.parse_args()


def _map_sentiment(rating: float) -> str:
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def main() -> None:
    args = parse_args()

    with args.params.open("r", encoding="utf-8") as fh:
        params = yaml.safe_load(fh) or {}

    train_ratio = params.get("data", {}).get("train_ratio", 0.70)
    val_ratio = params.get("data", {}).get("val_ratio", 0.15)
    test_ratio = params.get("data", {}).get("test_ratio", 0.15)
    seed = params.get("project", {}).get("random_seed", 42)

    # ------------------------------------------------------------------
    # Load interim data
    # ------------------------------------------------------------------
    logger.info("Loading interim data from %s …", args.interim)
    import pandas as pd

    df = pd.read_parquet(args.interim)

    # Normalise column names from Colab extraction script.
    if "category" in df.columns and "main_category" not in df.columns:
        df = df.rename(columns={"category": "main_category"})

    logger.info("Loaded %d rows with columns: %s", len(df), list(df.columns))

    # ------------------------------------------------------------------
    # Ensure sentiment column
    # ------------------------------------------------------------------
    if "sentiment" not in df.columns:
        if "rating" not in df.columns:
            raise ValueError("Neither 'sentiment' nor 'rating' column found.")
        df["sentiment"] = df["rating"].apply(_map_sentiment)
        logger.info("Derived sentiment from rating column.")

    # ------------------------------------------------------------------
    # Build cleaned_text column (idempotent if already present)
    # ------------------------------------------------------------------
    if "cleaned_text" not in df.columns:
        raw_col = "text" if "text" in df.columns else df.columns[0]
        logger.info("Cleaning text column '%s' …", raw_col)
        df["cleaned_text"] = df[raw_col].apply(clean_text)
    else:
        logger.info("cleaned_text column already present, skipping re-cleaning.")

    # Drop rows with empty cleaned text.
    before = len(df)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    logger.info("Dropped %d empty-text rows (%.2f%%).", before - len(df), 100 * (before - len(df)) / before)

    # ------------------------------------------------------------------
    # Split by parent_asin
    # ------------------------------------------------------------------
    logger.info(
        "Splitting by parent_asin (train=%.0f%% val=%.0f%% test=%.0f%%) …",
        train_ratio * 100,
        val_ratio * 100,
        test_ratio * 100,
    )
    train, val, test = split_by_asin(
        df,
        test_size=test_ratio,
        val_size=val_ratio,
        random_state=seed,
    )

    # ------------------------------------------------------------------
    # Enforce output schema: keep only known columns that exist.
    # ------------------------------------------------------------------
    out_cols = [c for c in REQUIRED_OUTPUT_COLUMNS if c in df.columns]
    train = train[out_cols].reset_index(drop=True)
    val = val[out_cols].reset_index(drop=True)
    test = test[out_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = ensure_dir(args.out_dir)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    summary = {
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "columns": out_cols,
        "seed": seed,
    }
    save_json(summary, out_dir / "split_summary.json")

    logger.info("Saved splits → train=%d val=%d test=%d", len(train), len(val), len(test))
    logger.info("Output directory: %s", out_dir)


if __name__ == "__main__":
    main()
