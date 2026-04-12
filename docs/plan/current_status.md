# Current Project Status

Last updated: 2026-04-12

## Executive summary
- Notebook 3 (Word2Vec training) is complete in full-run mode.
- Notebook 4 (controlled evaluation) is complete and exported final tables.
- Phase 4 currently supports a mixed conclusion: TF-IDF is strongest on classification, Word2Vec is strongest on retrieval, and clustering evidence remains weak.
- Strict category scope is enforced: Electronics, Home_and_Kitchen, Beauty_and_Personal_Care, Sports_and_Outdoors.

## Phase checklist
- [x] Phase 0: Scope freeze
- [x] Phase 1: Data readiness
- [x] Phase 2: Baseline features (TF-IDF, LSA)
- [x] Phase 3: Learned embeddings (Word2Vec)
- [x] Phase 4: Controlled evaluation
- [ ] Phase 5: Interpretation (UMAP, neighborhood analysis, failure analysis)
- [ ] Phase 6: Delivery (report, slides, reproducibility package)

## Verified artifacts

### Phase 3 (full mode)
- `artifacts/metrics/word2vec_skipgram_summary.json`
  - `max_train_rows: null`
  - `max_vector_rows: null`
- `artifacts/models/word2vec_skipgram.model`
- `artifacts/models/word2vec_skipgram.kv`
- `artifacts/vectors/word2vec_skipgram_train_vectors.npy`
- `artifacts/vectors/word2vec_skipgram_val_vectors.npy`
- `artifacts/vectors/word2vec_skipgram_test_vectors.npy`

### Phase 4 (final evaluation tables)
- `artifacts/metrics/phase4_classification_results.csv`
- `artifacts/metrics/phase4_retrieval_results.csv`
- `artifacts/metrics/phase4_clustering_results.csv`
- `artifacts/metrics/phase4_unified_results.csv`
- `reports/tables/tbl_02_classification_results.csv`
- `reports/tables/tbl_03_retrieval_results.csv`
- `reports/tables/tbl_04_clustering_results.csv`
- `reports/tables/tbl_05_phase4_unified_results.csv`

## Key Phase 4 snapshot
- Best category classification: TF-IDF (`category_test_macro_f1 = 0.7387`)
- Best sentiment classification: TF-IDF (`sentiment_test_macro_f1 = 0.6664`)
- Best retrieval: Word2Vec (`mrr_at_10 = 0.6866`)
- Best clustering (NMI-category): TF-IDF (`nmi_category = 0.0034`)
- Interpretation guardrail: Phase 4 does not support a blanket "Word2Vec is better" claim. The evidence supports stronger retrieval utility for Word2Vec, but not stronger classification or clearly superior clustering geometry.

## Notes on runtime
- Full Notebook 4 run is expected to be long.
- The heaviest steps are TF-IDF/LSA feature building and classification model fitting.

## Next tracked actions
1. Phase 5: add interpretation outputs (UMAP, nearest-neighbor examples, error/failure cases).
2. Phase 6: draft final report and presentation using current tables.
3. Optional: update `master_plan.md` wording from "subset/smoke" to "full-run verified" for consistency.
