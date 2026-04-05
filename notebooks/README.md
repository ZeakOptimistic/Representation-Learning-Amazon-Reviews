# `notebooks/`

This is where narrative analysis happens.

## Notebook policy

- Each notebook has one clear purpose
- Notebooks are numbered in execution order
- Heavy reusable logic belongs in `src/`
- Clear outputs are saved to `reports/` or `artifacts/`
- Clear all unnecessary cell outputs before commit

## Notebook sequence

- `00_scope_and_hypotheses.ipynb`
- `01_data_audit_and_sampling.ipynb`
- `02_baseline_tfidf_lsa.ipynb`
- `03_train_word2vec.ipynb`
- `04_linear_probe_eval.ipynb`
- `05_retrieval_and_clustering.ipynb`
- `06_latent_space_analysis.ipynb`
- `07_report_figure_generation.ipynb`

## Golden rule

The notebook should explain **what** you are doing and **why**.  
The scripts in `src/` should do the heavy lifting.
