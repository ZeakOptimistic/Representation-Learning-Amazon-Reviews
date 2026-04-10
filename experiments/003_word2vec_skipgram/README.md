# Word2Vec Skip-Gram

## Question
Can a learned skip-gram embedding produce stable, queryable document vectors with meaningful nearest-neighbor structure?

## Inputs
- data split: `data/processed/train.parquet`, `val.parquet`, `test.parquet`
- config file: `configs/word2vec.yaml` + `params.yaml`
- seed: 42
- run command (latest verified smoke run): `python -m src.models.train_word2vec --max-train-rows 15000 --max-vector-rows 5000`

## Representation
Word2Vec skip-gram token embeddings with `vector_size=300`, then mean pooling over in-vocab tokens per review.

## Evaluations
- sentiment classification: deferred to Phase 4
- category classification: deferred to Phase 4
- retrieval: sanity-only nearest neighbors (word-level and document-level)
- clustering: deferred to Phase 5

## Result summary
- vocabulary size: 3,470
- vector export coverage:
	- train: 0.8703 (13,055 / 15,000 non-zero vectors)
	- val: 0.8602 (4,301 / 5,000 non-zero vectors)
	- test: 0.8662 (4,331 / 5,000 non-zero vectors)
- artifacts generated:
	- `artifacts/models/word2vec_skipgram.model`
	- `artifacts/models/word2vec_skipgram.kv`
	- `artifacts/vectors/word2vec_skipgram_{train,val,test}_vectors.npy`
	- `artifacts/vectors/word2vec_skipgram_{train,val,test}_meta.parquet`
	- `artifacts/metrics/word2vec_skipgram_summary.json`
	- `artifacts/metrics/word2vec_skipgram_sanity_checks.txt`

## Decision
Proceed to Phase 4 linear-probe evaluation using the saved document vectors. Before final report freeze, rerun this training at full scale (no row caps) so final metrics and examples come from the full data regime.
