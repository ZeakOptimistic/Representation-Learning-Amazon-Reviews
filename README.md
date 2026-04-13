<div align="center">
  <h1>Representation Learning on Amazon Reviews 2023</h1>
  <p><strong>Comparing classical sparse representations (TF-IDF/LSA) with learned dense embeddings (Word2Vec) on a large-scale real-world dataset.</strong></p>
</div>

---

## 📌 Project Overview

This repository houses a full-scale data mining and representation learning project built on the massive **[Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/)**. 

By processing a dedicated subset of **2.0 million reviews** across 4 categories (Electronics, Home & Kitchen, Beauty, Sports), this project moves beyond simple string matching or rudimentary NLP. We challenge classical methodologies against learned dense embeddings to evaluate exactly *when* and *why* context-aware geometry outperforms sparse frequencies.

### Core Hypotheses
* **Hypothesis A (Sparsity vs. Density for Classification):** Word2Vec's dense embeddings will outperform TF-IDF in category classification by naturally capturing semantic synonyms (e.g., "fast" and "quick") that count-based sparse methods miss or fragment.
* **Hypothesis B (Latent Space Geometry for Clustering):** We expect the dense latent space, when visualized via UMAP, to display distinct "semantic neighborhoods" where products with similar utility (e.g., all "Gaming Keyboards") cluster together despite vastly differing brand names or exact lexical tokens.

---

## 📊 Selected Results

Key performance metrics on the held-out test set:

| Representation | Sentiment F1 | Category F1 | Retrieval MRR@10 | Clustering NMI (category) |
|----------------|--------------|-------------|------------------|---------------------------|
| TF-IDF | 0.577 | 0.672 | 0.0171 | 0.0034 |
| Word2Vec | 0.595 | 0.694 | 0.0249 | 0.0023 |
| LSA | 0.554 | 0.649 | 0.0128 | 0.0011 |

Latent space analysis (Word2Vec) reveals:
- **Neighborhood purity**: 55.0% of a review's 10 nearest neighbors share its main category.
- **Category variation**: Purity ranges from 46.9% (Home & Kitchen) to 59.1% (Beauty & Personal Care).
- **Outliers**: 35% of top 20 anomalies come from Beauty & Personal Care.
- **Semantic overlap**: The most similar distinct category centroids are Home & Kitchen and Sports & Outdoors (cosine ≈ 0.997), indicating shared review language.
- **Interpretive gap**: Neighborhood purity (0.550) exceeds global category classification F1 (0.694), suggesting local neighborhoods are purer than the overall classifier decision boundaries.
- **UMAP visualization**: Overlapping clusters with no single category dominating; Electronics forms more distinct islands.

See `reports/tables/` for detailed outputs.

---

## 🗺️ Knowledge & Execution Roadmap

We approach the problem systematically. Each phase of the pipeline is mapped to specific, reproducible Jupyter Notebooks (`notebooks/`) and backed by modular Python source code (`src/`).

| Phase | Knowledge Objective | Key Deliverable | Assoc. Notebooks |
| :--- | :--- | :--- | :--- |
| **I. Data Foundations** | Handling large-scale data ingestion, sampling bias, and executing strict zero-leakage `parent_asin` cross-validation splits. | Balanced Subset parquet logs (2.0M reviews). | `00`, `01` |
| **II. The Baseline Era** | Establishing a strong performance floor using classical sparse NLP (Scikit-Learn TF-IDF, Truncated SVD). | Baseline Performance Tables & Failure analysis. | `02` |
| **III. Embedding Space** | Training dense semantic vectors from scratch to map lexical tokens to continuous space. | Custom Trained Word2Vec Model & Document pooled embeddings. | `03`, `04` |
| **IV. The "Deep Dive"** | Mathematically visualizing and interpreting the latent geometry of the customer feedback. | UMAP Clusters, Neighborhood Analysis. | `05`, `06`, `07` |

---

## 🏗️ Repository Architecture

This repository strictly adheres to the **Cookiecutter Data Science** standard to ensure complete reproducibility and professional modularization. 

```text
rep-learning-amazon-reviews/
├── .github/          # CI/CD and Pull Request templates
├── artifacts/        # Heavy generated vectors and trained models (git-ignored)
├── configs/          # Configuration files for experiments
├── data/             # raw/, interim/, and processed/ datasets (git-ignored)
├── docs/             # Project master plans, progress documents, and reports
├── notebooks/        # Numbered narrative analysis notebooks (00 -> 07)
├── pyproject.toml    # Modern python manifest to make `src/` installable
├── requirements.txt  # Project-level dependencies
├── src/              # Core reusable Python modules
│   ├── data/         -> Ingestion, streaming, splits
│   ├── features/     -> TF-IDF / LSA wrappers
│   ├── models/       -> Word2Vec training, Classifiers
│   └── evaluation/   -> Metrics, Latent Space mappings
└── tests/            # Unit testing suite (pytest)
```

> **Note**: Notebooks are used strictly for narration, plotting, and orchestration. All heavy algorithmic lifting is done inside the importable `src/` directory.

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and install the dependencies into a virtual environment. Because we use `pyproject.toml`, installing the requirements will also install `src/` locally.

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows run: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .             # Installs the src package
```

### 2. Data Processing Pipeline

Run the notebooks in order to generate features and models:

1. `notebooks/00_project_setup.ipynb` (optional – verifies environment and directory structure)
2. `notebooks/01_data_audit_and_sampling.ipynb` – download, sample, and split the Amazon Reviews 2023 dataset.
3. `notebooks/02_baseline_tfidf_lsa.ipynb` – generate TF-IDF and LSA feature matrices.
4. `notebooks/03_train_word2vec.ipynb` – train Word2Vec skip-gram embeddings and produce document vectors.
5. `notebooks/04_controlled_evaluation.ipynb` – evaluate classification (sentiment, category), retrieval, and clustering metrics on the test set.
6. `notebooks/05_retrieval_and_clustering.ipynb` – deep dive into nearest-neighbor examples and cluster summaries.
7. `notebooks/06_latent_space_analysis.ipynb` – latent space interpretation via UMAP, neighborhood purity, outlier detection, and semantic direction analysis.

All outputs are saved under `reports/tables/` and `reports/figures/`.

---

## 📚 Citations & References

- **Dataset**: Hou, Y., et al. "Amazon Reviews 2023." *Under review*. [Project Page](https://amazon-reviews-2023.github.io/).
- **Methodology**: Mikolov, T., et al. "Efficient Estimation of Word Representations in Vector Space." *arXiv preprint arXiv:1301.3781 (2013)*.