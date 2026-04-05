# Master Plan

## 1. Project identity

**Title**  
Understanding Representation Learning at Scale on Amazon Reviews 2023

**Research question**  
Do learned dense representations produce more useful geometry than sparse count-based features on a large, practical review-mining dataset?

**Core thesis**  
Objective -> representation -> geometry -> utility

## 2. Why this project

The original idea was scientifically good but too broad.  
This project narrows the work into one real-world, high-volume domain so the result is deeper, more rigorous, and easier to defend.

## 3. Final scope

### Dataset

Use Amazon Reviews 2023.

### Selected categories

- Electronics
- Home_and_Kitchen
- Beauty_and_Personal_Care
- Sports_and_Outdoors

### Working subset

- 300,000 reviews per category
- 1.2M total reviews
- review text + rating + helpful votes + timestamp + verified purchase + parent_asin + main_category

### Split strategy

Split by `parent_asin` to reduce leakage across train / validation / test.

## 4. Representations to compare

### Classical baselines

1. TF-IDF
2. TF-IDF + Truncated SVD (LSA)

### Learned representation

1. Word2Vec skip-gram
2. review vector = pooled word vectors

### Optional extension

- FastText
- CBOW variant
- temporal drift analysis

## 5. Tasks

### Primary task A: sentiment / rating classification

Map ratings to:
- negative = 1, 2
- neutral = 3
- positive = 4, 5

Model on frozen features:
- Logistic Regression
- Linear SVM if needed

Metric:
- Macro-F1
- Accuracy

### Primary task B: category classification

Predict 4-category product domain from review text.

Metric:
- Macro-F1
- Accuracy

### Secondary task C: nearest-neighbor retrieval

Given a query review, retrieve semantically similar reviews.

Positive match options:
- same product family (`parent_asin`)
- same category + similar rating
- manually inspected semantic neighbors

Metric:
- Recall@10
- MRR@10

### Secondary task D: clustering quality

Cluster review vectors.

Metric:
- NMI / ARI against category and sentiment labels
- cluster coherence by top terms

## 6. Latent-space analysis

This section is essential because it directly answers the advisor feedback.

### Required analyses

1. UMAP projection colored by:
   - category
   - rating group
   - verified purchase
2. neighborhood purity
3. representative nearest-neighbor examples
4. cluster-level keyword summaries
5. outlier review inspection
6. failure-case gallery

### Strong result criteria

A good latent-space section does **not** just show a pretty plot.  
It must explain:
- what clusters exist
- what semantic direction separates them
- what the model gets wrong
- how this connects back to the training objective

## 7. Execution phases

### Phase 0 — Scope freeze
Output:
- confirmed title
- confirmed categories
- confirmed metrics
- confirmed split strategy

Done when:
- team agrees not to add another modality

### Phase 1 — Data readiness
Output:
- raw files downloaded
- balanced subset script
- leakage-safe split
- data dictionary
- class balance report

Done when:
- team can reproduce the same rows and splits from a seed

### Phase 2 — Baseline features
Output:
- TF-IDF vectors
- LSA vectors
- baseline notebook with first tables

Done when:
- baseline numbers exist for both primary tasks

### Phase 3 — Learned embeddings
Output:
- trained Word2Vec model
- saved document vectors
- nearest-neighbor sanity check

Done when:
- embeddings are saved and queryable

### Phase 4 — Controlled evaluation
Output:
- unified result table
- classification metrics
- retrieval metrics
- clustering metrics

Done when:
- every metric is tied to one config and one split

### Phase 5 — Interpretation
Output:
- UMAP plots
- neighborhood examples
- failure analysis
- key insights section for report

Done when:
- at least 5 non-trivial latent-space findings are documented

### Phase 6 — Delivery
Output:
- report
- slides
- final cleaned repo
- reproducibility checklist

Done when:
- another teammate can clone the repo and understand it in 10 minutes

## 8. Non-goals

To stay controlled, do **not** add these unless core work is done:

- cross-modal learning
- transformer fine-tuning
- full recommendation pipeline
- multimodal product-image fusion
- huge hyperparameter search

## 9. Final expected contribution

A rigorous, applied demonstration that:
- sparse features remain strong on direct lexical tasks
- learned embeddings provide better geometry for retrieval and clustering
- latent-space interpretation explains **why**
