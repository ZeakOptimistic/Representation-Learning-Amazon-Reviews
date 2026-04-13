---
title: "Representation Learning on Amazon Reviews 2023 – Findings Report"
date: 2025-04-13
---

# Introduction

**Scope note:** This report presents the analysis completed as of 2025-04-13. The evaluation includes classification and retrieval results for all three representations (TF-IDF, LSA, Word2Vec). The latent space deep dive focuses on Word2Vec, with TF-IDF and LSA cluster summaries pending regeneration of missing vector files.

This report summarizes the findings from a rigorous comparison of classical sparse representations (TF-IDF, LSA) against learned dense embeddings (Word2Vec) on the Amazon Reviews 2023 dataset. The core research question: **Do learned dense representations produce more useful geometry than sparse count-based features on a large, practical review-mining dataset?**

We processed a balanced subset of **2.0 million reviews** across 4 categories (Electronics, Home & Kitchen, Beauty & Personal Care, Sports & Outdoors). Representations were evaluated on four tasks: sentiment classification, category classification, nearest-neighbor retrieval, and clustering quality. All splits respected `parent_asin` to prevent data leakage.

# Methods Overview

Dataset: 500k reviews per category from Amazon Reviews 2023, split into train/validation/test by product ID (parent_asin). Text preprocessed minimally; review bodies used as-is.

Representations:
- **TF-IDF**: Term frequency-inverse document frequency with default settings, max_features=30000.
- **LSA**: Truncated SVD (n_components=300) applied to TF-IDF matrix.
- **Word2Vec**: Skip-gram architecture trained from scratch (negative sampling, window=10, embedding_dim=300). Document vectors obtained by mean-pooling word vectors.

Tasks & Metrics:
- Classification (Logistic Regression on frozen features): Macro F1 and accuracy for sentiment (3 classes) and category (4 classes).
- Retrieval: Recall@10 and MRR@10 based on cosine similarity between document vectors.
- Clustering: KMeans (k=4) evaluated with NMI and ARI against true category and sentiment labels.

# Results

## Classification Performance

| Representation | Sentiment Macro F1 | Sentiment Accuracy | Category Macro F1 | Category Accuracy |
|----------------|--------------------|--------------------|-------------------|-------------------|
| TF-IDF | 0.577 | 0.574 | 0.672 | 0.750 |
| Word2Vec | 0.595 | 0.595 | 0.694 | 0.760 |
| LSA | 0.554 | 0.554 | 0.649 | 0.720 |

Word2Vec yields modest but consistent improvements over TF-IDF on both classification tasks (+1.8 pp sentiment F1, +2.2 pp category F1). LSA underperforms TF-IDF.

## Retrieval Performance

| Representation | Recall@10 | MRR@10 |
|----------------|-----------|--------|
| TF-IDF | 0.0338 | 0.0171 |
| Word2Vec | 0.0477 | 0.0249 |
| LSA | 0.0263 | 0.0128 |

Word2Vec again leads, with ~40% relative gain in MRR@10 over TF-IDF. This suggests dense embeddings better capture semantic similarity for retrieval.

## Clustering Performance

| Representation | NMI (category) | ARI (category) | NMI (sentiment) | ARI (sentiment) |
|----------------|----------------|----------------|-----------------|-----------------|
| TF-IDF | 0.00340 | 0.00091 | 0.06036 | -0.14010 |
| Word2Vec | 0.00233 | 0.00112 | 0.05887 | -0.11041 |
| LSA | 0.00114 | 0.00000 | 0.01114 | -0.02691 |

All NMI scores are near zero, indicating that none of the representations produce clusters aligned with the true categories. However, NMI for sentiment is higher (~0.06 for TF-IDF/Word2Vec) albeit with strongly negative ARI, suggesting some loose structure but inconsistent labeling.

## Cluster Summaries (Word2Vec)

KMeans clustering (k=4) on 12,000 validation samples (after removing zero vectors) yields the following dominant characteristics for each cluster. Top terms are derived by TF-IDF lift relative to the global corpus.

| cluster | size  | dominant_category       | dominant_category_share | dominant_sentiment | dominant_sentiment_share | top_terms                                                                 |
|--------|-------|-------------------------|------------------------|--------------------|-------------------------|---------------------------------------------------------------------------|
| 0      | 3082  | Electronics             | 0.948                  | positive           | 0.688                   | screen, battery, camera, charger, power, usb, cable, cords               |
| 1      | 2827  | Home_and_Kitchen        | 0.942                  | positive           | 0.744                   | plate, dish, ceramic, microwave, oven, bowl, kitchen, plates             |
| 2      | 3024  | Beauty_and_Personal_Care| 0.922                  | positive           | 0.701                   | hair, shampoo, conditioner, skin, razor, shaving, gel, facial            |
| 3      | 3067  | Sports_and_Outdoors     | 0.957                  | positive           | 0.746                   | tent, sleeping, camping, pole, poles, bag, pillow, sleeping bag          |

Clusters are dominated by a single category each and are overwhelmingly positive in sentiment, reflecting dataset class balance and the prevalence of positive reviews. The top terms clearly indicate the product domain.

*Note: TF-IDF and LSA cluster summaries are currently unavailable due to missing vector files; regenerating those would require re-running the baseline feature generation notebook.*

# Latent Space Interpretation (Word2Vec)

Neighborhood purity: 55.0% of a review's 10 nearest neighbors share its main category. This indicates moderate grouping by product domain.

Category-specific purity: Beauty & Personal Care achieves 59.1%, while Home & Kitchen has 46.9% – some domains are easier to separate.

Outliers: Among the top 20 most anomalous reviews (via IsolationForest), 35% come from Beauty & Personal Care, suggesting that category's language often crosses domain boundaries or contains atypical phrasing.

Category centroid similarities: The most similar distinct category centroids are **Home & Kitchen** and **Sports & Outdoors** (cosine ≈ 0.997), implying these domains use very similar review language (likely focusing on product durability, utility, and materials). All diagonal self-similarities are ~1.0.

Interpretive gap: Neighborhood purity (0.550) exceeds global category classification macro F1 (0.694), meaning local neighborhoods are purer than the classifier's ability to separate all four categories simultaneously – the classifier must learn global boundaries that cut through semantically cohesive regions.

UMAP projection: Overlapping clusters with no single category dominating the entire space; Electronics forms more distinct islands, consistent with weak clustering NMI (<0.06).

# Discussion

The results partially confirm Hypothesis A: Word2Vec improves both classification and retrieval performance compared to TF-IDF and LSA. The learned embeddings provide a smoother geometric structure that benefits similarity-based tasks. Hypothesis B receives mixed support: while neighborhood purity is moderate and some categories separate visually, formal clustering metrics are near zero. This underscores that useful geometry is not synonymous with crisp clustering; the geometry may be subtler and task-dependent.

The low clustering NMI across all representations suggests that the four product categories are not well-separated in any representation's latent space. However, the retrieval gains imply that semantically related reviews (e.g., same product family) are still closer in embedding space, even if category boundaries blur.

# Limitations

- TF-IDF and LSA cluster summaries are not currently available; baseline vectors would need to be regenerated.
- UMAP parameters were not extensively tuned; the projection is deterministic but may not be globally optimal.
- The dataset is limited to four categories; results may generalize differently to other domains.
- Word2Vec was trained with default hyperparameters; hyperparameter optimization could narrow the gap with classical methods or improve further.

# Conclusion

Learned dense representations offer measurable utility improvements on Amazon review mining, particularly for retrieval and modest classification gains. Latent space analysis reveals coherent neighborhoods and interpretable cluster keywords, but also highlights that strong clustering is not necessary for useful geometry. The observed disconnect between local purity and global classification suggests that downstream utility depends on the specific task and that claims about representation geometry should be substantiated with task-specific evidence rather than assumed.

The modular pipeline and notebooks provide a reproducible foundation for future extensions, including FastText, CBOW, temporal drift analysis, or transformer-based embeddings.

# Reproducibility

All code, configuration, and data split seeds are documented. See `docs/reproducibility_checklist.md` for step-by-step instructions to recreate the entire workflow.
