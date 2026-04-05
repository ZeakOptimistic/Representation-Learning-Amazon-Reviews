# `data/`

This folder stores dataset assets by processing stage.

## Layout

- `raw/`: original downloaded files, never edited by hand
- `interim/`: cleaned but not final tables
- `processed/`: final analysis-ready splits
- `external/`: supporting files from outside the main dataset

## Rules

1. Raw data is immutable.
2. Do not commit large raw data into Git.
3. Every processed dataset must have:
   - a creation script
   - a seed
   - a row count
   - a schema summary
4. Splits must be reproducible.

## Required artifacts for this project

- balanced review subset manifest
- train / val / test split files
- data dictionary
- class balance summary
