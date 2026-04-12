"""Baseline feature generation entrypoint.

Fits TF-IDF and LSA on the training split, transforms all three splits,
and saves the resulting matrices to ``artifacts/vectors/``.

This script is called by DVC's ``build_baselines`` stage::

    python -m src.features.build_all

Saved artefacts
---------------
- ``artifacts/vectors/tfidf_train.npz``  — sparse CSR matrix
- ``artifacts/vectors/tfidf_val.npz``
- ``artifacts/vectors/tfidf_test.npz``
- ``artifacts/vectors/lsa_train.npy``    — dense float32 array
- ``artifacts/vectors/lsa_val.npy``
- ``artifacts/vectors/lsa_test.npy``
- ``artifacts/vectors/baselines_summary.json``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

_CWD = Path.cwd().resolve()
_CANDIDATES = [_CWD, _CWD.parent]
PROJECT_ROOT = next(
    (p for p in _CANDIDATES if (p / "src").exists() and (p / "configs").exists()),
    _CWD,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.lsa_features import build_lsa  # noqa: E402
from src.features.tfidf_features import build_vectorizer  # noqa: E402
from src.utils.io import ensure_dir, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=PROJECT_ROOT / "params.yaml")
    parser.add_argument(
        "--train", type=Path, default=PROJECT_ROOT / "data" / "processed" / "train.parquet"
    )
    parser.add_argument(
        "--val", type=Path, default=PROJECT_ROOT / "data" / "processed" / "val.parquet"
    )
    parser.add_argument(
        "--test", type=Path, default=PROJECT_ROOT / "data" / "processed" / "test.parquet"
    )
    return parser.parse_args()


def _read_parquet_text(path: Path, text_column: str = "cleaned_text") -> "pd.Series":
    import pandas as pd  # local import — pandas is heavy

    df = pd.read_parquet(path)
    col = text_column if text_column in df.columns else "text"
    return df[col].fillna("")


def main() -> None:
    args = parse_args()

    with args.params.open("r", encoding="utf-8") as fh:
        params = yaml.safe_load(fh) or {}

    tfidf_max_features = params.get("features", {}).get("tfidf_max_features", 100_000)
    lsa_components = params.get("features", {}).get("lsa_components", 300)
    text_column = params.get("data", {}).get("text_column", "cleaned_text")
    seed = params.get("project", {}).get("random_seed", 42)

    vectors_dir = ensure_dir(PROJECT_ROOT / "artifacts" / "vectors")

    print(f"Loading text splits (column='{text_column}') …")
    train_text = _read_parquet_text(args.train, text_column)
    val_text = _read_parquet_text(args.val, text_column)
    test_text = _read_parquet_text(args.test, text_column)
    print(f"  train={len(train_text):,}  val={len(val_text):,}  test={len(test_text):,}")

    # ------------------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------------------
    print(f"Fitting TF-IDF (max_features={tfidf_max_features}) …")
    tfidf = build_vectorizer(max_features=tfidf_max_features)
    X_tr_tfidf = tfidf.fit_transform(train_text)
    X_v_tfidf = tfidf.transform(val_text)
    X_te_tfidf = tfidf.transform(test_text)

    print(f"  TF-IDF shape: {X_tr_tfidf.shape}")

    sp.save_npz(vectors_dir / "tfidf_train.npz", X_tr_tfidf)
    sp.save_npz(vectors_dir / "tfidf_val.npz", X_v_tfidf)
    sp.save_npz(vectors_dir / "tfidf_test.npz", X_te_tfidf)
    print("  Saved tfidf_{{train,val,test}}.npz")

    # ------------------------------------------------------------------
    # LSA
    # ------------------------------------------------------------------
    print(f"Fitting LSA (n_components={lsa_components}) …")
    lsa = build_lsa(n_components=lsa_components, random_state=seed)
    X_tr_lsa = lsa.fit_transform(X_tr_tfidf).astype(np.float32)
    X_v_lsa = lsa.transform(X_v_tfidf).astype(np.float32)
    X_te_lsa = lsa.transform(X_te_tfidf).astype(np.float32)

    np.save(vectors_dir / "lsa_train.npy", X_tr_lsa)
    np.save(vectors_dir / "lsa_val.npy", X_v_lsa)
    np.save(vectors_dir / "lsa_test.npy", X_te_lsa)
    print("  Saved lsa_{{train,val,test}}.npy")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    explained = float(lsa.explained_variance_ratio_.sum())
    summary = {
        "tfidf": {
            "max_features": tfidf_max_features,
            "actual_features": X_tr_tfidf.shape[1],
            "shapes": {
                "train": list(X_tr_tfidf.shape),
                "val": list(X_v_tfidf.shape),
                "test": list(X_te_tfidf.shape),
            },
        },
        "lsa": {
            "n_components": lsa_components,
            "explained_variance_ratio_sum": round(explained, 4),
            "shapes": {
                "train": list(X_tr_lsa.shape),
                "val": list(X_v_lsa.shape),
                "test": list(X_te_lsa.shape),
            },
        },
    }
    save_json(summary, vectors_dir / "baselines_summary.json")
    print(f"\nBaseline features saved to {vectors_dir}")
    print(f"LSA explained variance (first {lsa_components} components): {explained:.3f}")


if __name__ == "__main__":
    main()
