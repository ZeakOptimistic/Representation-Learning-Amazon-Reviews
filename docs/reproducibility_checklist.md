# Reproducibility Checklist

This checklist ensures that every step of the pipeline can be rerun to obtain the reported results.

## Environment Setup

- [ ] Python 3.10+ installed
- [ ] Create virtual environment: `python -m venv .venv` and activate (`source .venv/bin/activate` or `.venv\Scripts\activate` on Windows)
- [ ] Upgrade pip: `pip install -U pip`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Install project package in editable mode: `pip install -e .`
- [ ] Verify installation: `python -c "import src"` should succeed

## Data Pipeline

All data files should be placed under `data/raw/` if not using automated download. The notebooks handle extraction and processing.

- [ ] Run `notebooks/00_project_setup.ipynb` (optional) – checks environment, creates required directories.
- [ ] Run `notebooks/01_data_audit_and_sampling.ipynb` – downloads Amazon Reviews 2023, performs reservoir sampling, creates balanced subset, and writes `data/processed/train.parquet`, `val.parquet`, `test.parquet`. Verify that each split has ~500k reviews per category.

## Feature Generation

- [ ] Run `notebooks/02_baseline_tfidf_lsa.ipynb` – fits TF-IDF on train, transforms all splits, fits LSA, and saves:
  - `artifacts/vectors/tfidf_train_vectors.npy`
  - `artifacts/vectors/tfidf_val_vectors.npy`
  - `artifacts/vectors/tfidf_test_vectors.npy`
  - `artifacts/vectors/lsa_train_vectors.npy`
  - `artifacts/vectors/lsa_val_vectors.npy`
  - `artifacts/vectors/lsa_test_vectors.npy`
  - `reports/tables/tbl_02_classification_results.csv` (baseline classification metrics)
  - `reports/tables/tbl_03_retrieval_results.csv` (baseline retrieval metrics)
  - `reports/tables/tbl_04_clustering_results.csv` (baseline clustering NMI/ARI)

- [ ] Run `notebooks/03_train_word2vec.ipynb` – trains Word2Vec skip-gram model, creates document vectors, and saves:
  - `artifacts/models/word2vec_skipgram.model`
  - `artifacts/vectors/word2vec_skipgram_train_vectors.npy`
  - `artifacts/vectors/word2vec_skipgram_val_vectors.npy`
  - `artifacts/vectors/word2vec_skipgram_test_vectors.npy`
  - `artifacts/metrics/word2vec_skipgram_summary.json` (training summary)
  - `reports/tables/tbl_02_classification_results.csv` (updated with Word2Vec)
  - Additional Word2Vec-specific metrics

## Controlled Evaluation

- [ ] Run `notebooks/04_controlled_evaluation.ipynb` – loads all representations and computes unified metrics:
  - `reports/tables/tbl_05_phase4_unified_results.csv` (combined classification, retrieval, clustering metrics)

## Detailed Analysis

- [ ] Run `notebooks/05_retrieval_and_clustering.ipynb` – performs nearest-neighbor sanity checks and extended clustering:
  - Qualitative neighbor tables (printed or saved)
  - Cluster composition plots
  - `reports/tables/tbl_05_word2vec_cluster_summary.csv` (top terms per cluster)
- [ ] Run `notebooks/06_latent_space_analysis.ipynb` – latent space interpretation:
  - `reports/tables/tbl_06_word2vec_umap.csv` (2D coordinates)
  - `reports/tables/tbl_06_word2vec_purity.csv` (neighborhood purity per review)
  - `reports/tables/tbl_06_word2vec_outliers.csv` (top 20 outliers)
  - `reports/tables/tbl_06_word2vec_category_centroid_sim.csv` (centroid similarity matrix)
  - `reports/tables/tbl_06_word2vec_insights.txt` (key insights)

## Expected Outputs

All results tables should be present under `reports/tables/`:
- `tbl_02_classification_results.csv`
- `tbl_03_retrieval_results.csv`
- `tbl_04_clustering_results.csv`
- `tbl_05_phase4_unified_results.csv`
- `tbl_05_word2vec_cluster_summary.csv`
- `tbl_06_word2vec_*.csv` and `tbl_06_word2vec_insights.txt`

Figures may be saved under `reports/figures/` by notebooks.

## Seeds and Configuration

- The random seed is fixed in `params.yaml` (`seed: 42`) and propagated through `build_config`.
- All data splits use `parent_asin` grouping to prevent leakage.
- Hyperparameters for Word2Vec are in `configs/word2vec.yaml` (e.g., `embedding_dim=300`, `window=10`, `negative_samples=5`).

By following this checklist and running notebooks sequentially, another researcher should be able to reproduce the entire workflow and obtain the same numbers (modulo nondeterminism in hardware or library versions, which should be minimal with fixed seeds).
