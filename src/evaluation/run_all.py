"""Evaluation pipeline entrypoint.

Loads saved representation vectors, runs all metric blocks (classification,
retrieval, clustering), and writes the unified CSV to ``artifacts/metrics/``
and ``reports/tables/``.

This script is called by DVC's ``evaluate`` stage::

    python -m src.evaluation.run_all

It reads the run configuration from the Word2Vec summary JSON so that
Phase 4 always aligns to the exact sub-sampling settings used in Phase 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so the script works whether called from repo root
# or from src/ or tests/.
# ---------------------------------------------------------------------------
_CWD = Path.cwd().resolve()
_CANDIDATES = [_CWD, _CWD.parent]
PROJECT_ROOT = next(
    (p for p in _CANDIDATES if (p / "src").exists() and (p / "configs").exists()),
    _CWD,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    classification_summary,
    clustering_scores,
    retrieval_metrics,
)
from src.features.lsa_features import build_lsa  # noqa: E402
from src.features.tfidf_features import build_vectorizer  # noqa: E402
from src.models.linear_probe import build_logistic_probe  # noqa: E402
from src.models.train_word2vec import (  # noqa: E402
    apply_row_cap,
    build_config,
    enforce_category_scope,
    load_split,
    _read_yaml,
)
from src.utils.io import ensure_dir, load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=PROJECT_ROOT / "params.yaml")
    parser.add_argument(
        "--w2v-config", type=Path, default=PROJECT_ROOT / "configs" / "word2vec.yaml"
    )
    parser.add_argument(
        "--w2v-summary",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "metrics"
        / "word2vec_skipgram_summary.json",
    )
    parser.add_argument(
        "--retrieval-k", type=int, default=10, help="Top-k for retrieval metrics."
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=1000,
        help="Max retrieval queries to evaluate.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of KMeans clusters for clustering metrics.",
    )
    return parser.parse_args()


def _load_vectors(tag: str, vectors_dir: Path) -> np.ndarray:
    path = vectors_dir / f"word2vec_skipgram_{tag}_vectors.npy"
    return np.load(path, mmap_mode="r")


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Config — mirror Phase 3 sub-sampling so comparisons are fair.
    # ------------------------------------------------------------------
    params = _read_yaml(args.params)
    exp_cfg = _read_yaml(args.w2v_config)
    w2v_summary = load_json(args.w2v_summary)

    cli_stub = SimpleNamespace(
        max_train_rows=w2v_summary["config"].get("max_train_rows"),
        max_vector_rows=w2v_summary["config"].get("max_vector_rows"),
    )
    cfg = build_config(params=params, exp=exp_cfg, cli=cli_stub)

    retrieval_k = args.retrieval_k
    query_limit = args.query_limit
    n_clusters = args.n_clusters

    # ------------------------------------------------------------------
    # Load text splits (same logic as Phase 3)
    # ------------------------------------------------------------------
    processed = PROJECT_ROOT / "data" / "processed"
    train_df, train_col = load_split(processed / "train.parquet", cfg.text_column)
    val_df, val_col = load_split(processed / "val.parquet", cfg.text_column)
    test_df, test_col = load_split(processed / "test.parquet", cfg.text_column)

    train_df = apply_row_cap(
        enforce_category_scope(train_df, cfg=cfg, split_name="train"),
        max_rows=cfg.max_train_rows,
        seed=cfg.seed,
    )
    val_df = apply_row_cap(
        enforce_category_scope(val_df, cfg=cfg, split_name="val"),
        max_rows=cfg.max_vector_rows,
        seed=cfg.seed,
    )
    test_df = apply_row_cap(
        enforce_category_scope(test_df, cfg=cfg, split_name="test"),
        max_rows=cfg.max_vector_rows,
        seed=cfg.seed,
    )

    print(f"Splits: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    # ------------------------------------------------------------------
    # Build TF-IDF / LSA features
    # ------------------------------------------------------------------
    tfidf_cap = params.get("features", {}).get("tfidf_max_features", 100_000)
    lsa_components = params.get("features", {}).get("lsa_components", 300)

    tfidf = build_vectorizer(max_features=tfidf_cap)
    lsa = build_lsa(n_components=lsa_components)

    X_tr_tfidf = tfidf.fit_transform(train_df[train_col])
    X_v_tfidf = tfidf.transform(val_df[val_col])
    X_te_tfidf = tfidf.transform(test_df[test_col])

    X_tr_lsa = lsa.fit_transform(X_tr_tfidf)
    X_v_lsa = lsa.transform(X_v_tfidf)
    X_te_lsa = lsa.transform(X_te_tfidf)

    # ------------------------------------------------------------------
    # Load Word2Vec vectors
    # ------------------------------------------------------------------
    vectors_dir = PROJECT_ROOT / "artifacts" / "vectors"
    X_tr_w2v = _load_vectors("train", vectors_dir)
    X_v_w2v = _load_vectors("val", vectors_dir)
    X_te_w2v = _load_vectors("test", vectors_dir)

    assert X_tr_w2v.shape[0] == len(train_df), "Train vector/label mismatch."
    assert X_v_w2v.shape[0] == len(val_df), "Val vector/label mismatch."
    assert X_te_w2v.shape[0] == len(test_df), "Test vector/label mismatch."

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    y_tr_sent = train_df["sentiment"].astype(str)
    y_v_sent = val_df["sentiment"].astype(str)
    y_te_sent = test_df["sentiment"].astype(str)
    y_tr_cat = train_df["main_category"].astype(str)
    y_v_cat = val_df["main_category"].astype(str)
    y_te_cat = test_df["main_category"].astype(str)

    repr_data = {
        "tfidf": (X_tr_tfidf, X_v_tfidf, X_te_tfidf),
        "lsa": (X_tr_lsa, X_v_lsa, X_te_lsa),
        "word2vec": (X_tr_w2v, X_v_w2v, X_te_w2v),
    }

    # ------------------------------------------------------------------
    # 1. Classification
    # ------------------------------------------------------------------
    print("Running classification probes …")
    cls_rows: list[dict] = []
    for rep_name, (Xtr, Xv, Xte) in repr_data.items():
        sent_clf = build_logistic_probe()
        sent_clf.fit(Xtr, y_tr_sent)
        cat_clf = build_logistic_probe()
        cat_clf.fit(Xtr, y_tr_cat)

        for split, Xe, y_sent, y_cat in [
            ("val", Xv, y_v_sent, y_v_cat),
            ("test", Xte, y_te_sent, y_te_cat),
        ]:
            cls_rows.append(
                {
                    "representation": rep_name,
                    "task": "sentiment",
                    "split": split,
                    **classification_summary(y_sent, sent_clf.predict(Xe)),
                }
            )
            cls_rows.append(
                {
                    "representation": rep_name,
                    "task": "category",
                    "split": split,
                    **classification_summary(y_cat, cat_clf.predict(Xe)),
                }
            )

    classification_df = pd.DataFrame(cls_rows).sort_values(
        ["task", "split", "macro_f1"], ascending=[True, True, False]
    )

    # ------------------------------------------------------------------
    # 2. Retrieval
    # ------------------------------------------------------------------
    print(f"Running retrieval metrics (k={retrieval_k}, limit={query_limit}) …")
    retr_rows: list[dict] = []
    for rep_name, (Xtr, Xv, _) in repr_data.items():
        metrics = retrieval_metrics(
            X_query=Xv,
            X_db=Xtr,
            query_category=y_v_cat,
            query_sentiment=y_v_sent,
            db_category=y_tr_cat,
            db_sentiment=y_tr_sent,
            k=retrieval_k,
            query_limit=query_limit,
        )
        retr_rows.append({"representation": rep_name, **metrics})

    retrieval_df = pd.DataFrame(retr_rows).sort_values(
        f"mrr_at_{retrieval_k}", ascending=False
    )

    # ------------------------------------------------------------------
    # 3. Clustering
    # ------------------------------------------------------------------
    print(f"Running clustering metrics (n_clusters={n_clusters}) …")
    cluster_rows: list[dict] = []
    for rep_name, (_, Xv, _) in repr_data.items():
        scores = clustering_scores(
            Xv, y_v_cat, y_v_sent, n_clusters=n_clusters, seed=cfg.seed
        )
        cluster_rows.append({"representation": rep_name, **scores})

    clustering_df = pd.DataFrame(cluster_rows).sort_values(
        "nmi_category", ascending=False
    )

    # ------------------------------------------------------------------
    # 4. Unified table
    # ------------------------------------------------------------------
    recall_col = f"recall_at_{retrieval_k}"
    mrr_col = f"mrr_at_{retrieval_k}"
    test_cls = classification_df[classification_df["split"] == "test"]
    sent_test = (
        test_cls[test_cls["task"] == "sentiment"][["representation", "macro_f1", "accuracy"]]
        .rename(columns={"macro_f1": "sentiment_test_macro_f1", "accuracy": "sentiment_test_accuracy"})
    )
    cat_test = (
        test_cls[test_cls["task"] == "category"][["representation", "macro_f1", "accuracy"]]
        .rename(columns={"macro_f1": "category_test_macro_f1", "accuracy": "category_test_accuracy"})
    )
    unified = (
        sent_test.merge(cat_test, on="representation", how="inner")
        .merge(retrieval_df[["representation", recall_col, mrr_col]], on="representation", how="left")
        .merge(clustering_df, on="representation", how="left")
        .sort_values("category_test_macro_f1", ascending=False)
    )

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    metrics_dir = ensure_dir(PROJECT_ROOT / "artifacts" / "metrics")
    tables_dir = ensure_dir(PROJECT_ROOT / "reports" / "tables")

    classification_df.to_csv(metrics_dir / "phase4_classification_results.csv", index=False)
    retrieval_df.to_csv(metrics_dir / "phase4_retrieval_results.csv", index=False)
    clustering_df.to_csv(metrics_dir / "phase4_clustering_results.csv", index=False)
    unified.to_csv(metrics_dir / "phase4_unified_results.csv", index=False)

    classification_df.to_csv(tables_dir / "tbl_02_classification_results.csv", index=False)
    retrieval_df.to_csv(tables_dir / "tbl_03_retrieval_results.csv", index=False)
    clustering_df.to_csv(tables_dir / "tbl_04_clustering_results.csv", index=False)
    unified.to_csv(tables_dir / "tbl_05_phase4_unified_results.csv", index=False)

    run_meta = {
        "retrieval_k": retrieval_k,
        "query_limit": query_limit,
        "n_clusters": n_clusters,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }
    save_json(run_meta, metrics_dir / "phase4_run_config.json")

    print("Phase 4 outputs saved:")
    print(f"  {metrics_dir}/phase4_unified_results.csv")
    print("\nBest results summary:")
    for _, row in unified.iterrows():
        print(
            f"  {row['representation']:10s}  "
            f"sent={row['sentiment_test_macro_f1']:.4f}  "
            f"cat={row['category_test_macro_f1']:.4f}  "
            f"{mrr_col}={row[mrr_col]:.4f}"
        )


if __name__ == "__main__":
    main()
