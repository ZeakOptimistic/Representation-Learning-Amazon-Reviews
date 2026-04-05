# `configs/`

Centralized configuration lives here so experiments are reproducible and easy to compare.

## Files

- `data.yaml`: sampling and split choices
- `baseline_tfidf.yaml`: sparse baseline settings
- `lsa.yaml`: dense classical baseline settings
- `word2vec.yaml`: learned representation settings

## Rule

If a metric table changes, the config that produced it must also be saved.
