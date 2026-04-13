"""Train Word2Vec and export document vectors for downstream tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gensim.models import Word2Vec

from src.features.document_embeddings import mean_pool_from_tokens
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


DEFAULT_SANITY_WORDS = [
    "battery",
    "charger",
    "kitchen",
    "beauty",
    "sports",
]


_CATEGORY_ALIASES = {
    "electronics": "Electronics",
    "homeandkitchen": "Home_and_Kitchen",
    "homekitchen": "Home_and_Kitchen",
    "home_and_kitchen": "Home_and_Kitchen",
    "beauty": "Beauty_and_Personal_Care",
    "beautyandpersonalcare": "Beauty_and_Personal_Care",
    "beauty_and_personal_care": "Beauty_and_Personal_Care",
    "sports": "Sports_and_Outdoors",
    "sportsandoutdoors": "Sports_and_Outdoors",
    "sports_and_outdoors": "Sports_and_Outdoors",
}


@dataclass
class Word2VecConfig:
    architecture: str
    vector_size: int
    window: int
    min_count: int
    negative: int
    epochs: int
    workers: int
    seed: int
    text_column: str
    category_column: str
    categories: list[str]
    min_review_tokens: int
    max_review_tokens: int
    vector_batch_size: int
    max_train_rows: int | None
    max_vector_rows: int | None
    sanity_words: list[str]
    sanity_doc_pool_size: int
    sanity_top_k: int
    strict_categories: bool

    @property
    def sg(self) -> int:
        return 1 if self.architecture.lower() in {"skipgram", "skip-gram", "sg"} else 0

    @property
    def model_name(self) -> str:
        return f"word2vec_{self.architecture.lower().replace('-', '')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/word2vec.yaml"))
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/train.parquet"))
    parser.add_argument("--val-path", type=Path, default=Path("data/processed/val.parquet"))
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/test.parquet"))
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-vector-rows", type=int, default=None)
    parser.add_argument("--skip-test-vectors", action="store_true")
    return parser.parse_args()


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _pick(exp_value, params_value, default_value):
    if exp_value is not None:
        return exp_value
    if params_value is not None:
        return params_value
    return default_value


def build_config(params: dict, exp: dict, cli: argparse.Namespace) -> Word2VecConfig:
    p_data = params.get("data", {})
    p_word2vec = params.get("word2vec", {})
    p_project = params.get("project", {})

    return Word2VecConfig(
        architecture=str(_pick(exp.get("architecture"), p_word2vec.get("architecture"), "skipgram")),
        vector_size=int(_pick(exp.get("vector_size"), p_word2vec.get("vector_size"), 300)),
        window=int(_pick(exp.get("window"), p_word2vec.get("window"), 5)),
        min_count=int(_pick(exp.get("min_count"), p_word2vec.get("min_count"), 10)),
        negative=int(_pick(exp.get("negative"), p_word2vec.get("negative"), 10)),
        epochs=int(_pick(exp.get("epochs"), p_word2vec.get("epochs"), 10)),
        workers=int(_pick(exp.get("workers"), p_word2vec.get("workers"), 4)),
        seed=int(_pick(exp.get("seed"), p_project.get("random_seed"), 42)),
        text_column=str(_pick(exp.get("text_column"), p_data.get("text_column"), "cleaned_text")),
        category_column=str(_pick(exp.get("category_column"), p_data.get("category_column"), "main_category")),
        categories=list(
            _pick(
                exp.get("categories"),
                p_data.get("categories"),
                [
                    "Electronics",
                    "Home_and_Kitchen",
                    "Beauty_and_Personal_Care",
                    "Sports_and_Outdoors",
                ],
            )
        ),
        min_review_tokens=int(
            _pick(exp.get("min_review_tokens"), p_data.get("min_review_tokens"), 1)
        ),
        max_review_tokens=int(
            _pick(exp.get("max_review_tokens"), p_data.get("max_review_tokens"), 256)
        ),
        vector_batch_size=int(_pick(exp.get("vector_batch_size"), None, 4096)),
        max_train_rows=cli.max_train_rows
        if cli.max_train_rows is not None
        else exp.get("max_train_rows"),
        max_vector_rows=cli.max_vector_rows
        if cli.max_vector_rows is not None
        else exp.get("max_vector_rows"),
        sanity_words=list(exp.get("sanity_words", DEFAULT_SANITY_WORDS)),
        sanity_doc_pool_size=int(_pick(exp.get("sanity_doc_pool_size"), None, 5000)),
        sanity_top_k=int(_pick(exp.get("sanity_top_k"), None, 5)),
        strict_categories=bool(_pick(exp.get("strict_categories"), None, True)),
    )


def _normalize_label(value: str) -> str:
    chars = [ch.lower() for ch in str(value) if ch.isalnum() or ch == "_"]
    return "".join(chars).replace("_", "")


def _canonicalize_category(value: str) -> str | None:
    key = _normalize_label(value)
    return _CATEGORY_ALIASES.get(key)


def _canonicalize_list(values: list[str]) -> list[str]:
    canonical = []
    for value in values:
        mapped = _canonicalize_category(value)
        canonical.append(mapped if mapped is not None else str(value))
    return canonical


def enforce_category_scope(df: pd.DataFrame, cfg: Word2VecConfig, split_name: str) -> pd.DataFrame:
    if cfg.category_column not in df.columns:
        raise ValueError(
            f"Missing category column '{cfg.category_column}' in {split_name} split."
        )

    allowed = set(_canonicalize_list(cfg.categories))
    original = df[cfg.category_column].astype(str)
    canonical = original.map(lambda value: _canonicalize_category(value) or value)

    unknown = sorted({value for value in canonical.unique() if value not in allowed})
    if unknown and cfg.strict_categories:
        allowed_preview = ", ".join(sorted(allowed))
        unknown_preview = ", ".join(unknown)
        raise ValueError(
            f"Unexpected categories in {split_name}: {unknown_preview}. "
            f"Allowed categories: {allowed_preview}."
        )

    filtered = df.copy()
    filtered[cfg.category_column] = canonical
    filtered = filtered[filtered[cfg.category_column].isin(allowed)].reset_index(drop=True)
    return filtered


def tokenize(text: str, min_tokens: int, max_tokens: int) -> list[str]:
    tokens = str(text).split()
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    if len(tokens) < min_tokens:
        return []
    return tokens


class TokenizedCorpus:
    """Re-iterable tokenized corpus for gensim Word2Vec."""

    def __init__(self, texts: pd.Series, min_tokens: int, max_tokens: int) -> None:
        self.texts = texts.reset_index(drop=True)
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def __iter__(self):
        for text in self.texts:
            tokens = tokenize(text, self.min_tokens, self.max_tokens)
            if tokens:
                yield tokens


def load_split(
    path: Path,
    text_column: str,
) -> tuple[pd.DataFrame, str]:
    df = pd.read_parquet(path)
    resolved_text_column = text_column
    if text_column not in df.columns:
        fallback = "cleaned_text" if "cleaned_text" in df.columns else "text"
        if fallback not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in {path} and no text fallback was found."
            )
        resolved_text_column = fallback

    df = df.dropna(subset=[resolved_text_column]).reset_index(drop=True)
    return df, resolved_text_column


def apply_row_cap(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def text_to_vector(model: Word2Vec, text: str, cfg: Word2VecConfig) -> np.ndarray:
    tokens = tokenize(text, cfg.min_review_tokens, cfg.max_review_tokens)
    return mean_pool_from_tokens(tokens=tokens, keyed_vectors=model.wv, vector_size=cfg.vector_size)


def export_split_vectors(
    model: Word2Vec,
    df: pd.DataFrame,
    split_name: str,
    text_column: str,
    cfg: Word2VecConfig,
    vectors_dir: Path,
) -> dict:
    n_rows = len(df)
    out_path = vectors_dir / f"{cfg.model_name}_{split_name}_vectors.npy"
    meta_path = vectors_dir / f"{cfg.model_name}_{split_name}_meta.parquet"

    try:
        vectors = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_rows, cfg.vector_size),
        )
    except OSError as e:
        # On Windows, overwriting a memmap'd file currently open in another
        # Python process (e.g., a Jupyter kernel running NB04 or NB06) fails
        # with Errno 13 (Permission) or Errno 22 (Invalid argument).
        raise RuntimeError(
            f"\n\n"
            f"❌ Cannot overwrite {out_path.name} because it is LOCKED by another process.\n"
            f"This usually happens when another Jupyter Notebook (like 04 or 06) is currently open\n"
            f"and has the old vectors loaded in memory.\n\n"
            f"ACTION REQUIRED: Go to your Jupyter interface, shut down all other running kernels,\n"
            f"and then run this training cell again.\n\n"
            f"Original OS Error: {e}"
        ) from e

    nonzero_rows = 0
    for start in range(0, n_rows, cfg.vector_batch_size):
        end = min(start + cfg.vector_batch_size, n_rows)
        batch = [text_to_vector(model, text, cfg) for text in df[text_column].iloc[start:end]]
        batch_vectors = np.vstack(batch).astype(np.float32, copy=False)
        vectors[start:end] = batch_vectors
        nonzero_rows += int(np.sum(np.linalg.norm(batch_vectors, axis=1) > 0.0))

    del vectors

    meta_cols = ["sentiment", "main_category", "parent_asin"]
    available_cols = [col for col in meta_cols if col in df.columns]
    meta_df = df[available_cols].copy()
    meta_df.insert(0, "row_id", np.arange(n_rows, dtype=np.int64))
    meta_df.to_parquet(meta_path, index=False)

    return {
        "split": split_name,
        "rows": n_rows,
        "nonzero_rows": nonzero_rows,
        "coverage": float(nonzero_rows / n_rows) if n_rows else 0.0,
        "vectors_path": str(out_path),
        "meta_path": str(meta_path),
    }


def build_word_sanity(model: Word2Vec, sanity_words: list[str], top_k: int) -> list[dict]:
    checks: list[dict] = []
    for word in sanity_words:
        if word not in model.wv:
            checks.append({"word": word, "found": False, "neighbors": []})
            continue

        neighbors = model.wv.most_similar(word, topn=top_k)
        checks.append(
            {
                "word": word,
                "found": True,
                "neighbors": [
                    {"token": token, "score": float(score)} for token, score in neighbors
                ],
            }
        )
    return checks


def build_document_sanity(
    val_df: pd.DataFrame,
    val_vectors_path: Path,
    text_column: str,
    seed: int,
    pool_size: int,
    top_k: int,
) -> list[dict]:
    if len(val_df) < 2:
        return []

    vectors = np.load(val_vectors_path, mmap_mode="r")
    pool_size = min(pool_size, len(val_df))
    rng = np.random.default_rng(seed)
    pool_indices = rng.choice(len(val_df), size=pool_size, replace=False)

    pooled_vectors = np.asarray(vectors[pool_indices])
    vector_norms = np.linalg.norm(pooled_vectors, axis=1)
    nonzero_positions = np.where(vector_norms > 0.0)[0]
    if len(nonzero_positions) == 0:
        return []

    query_positions = nonzero_positions[: min(3, len(nonzero_positions))]
    checks: list[dict] = []

    for query_pos in query_positions:
        sims = pooled_vectors @ pooled_vectors[query_pos]
        denom = np.linalg.norm(pooled_vectors, axis=1) * np.linalg.norm(pooled_vectors[query_pos])
        sims = np.divide(sims, denom, out=np.zeros_like(sims), where=denom > 0)
        sims[query_pos] = -1.0
        top_idx = np.argsort(sims)[-top_k:][::-1]

        query_row = int(pool_indices[query_pos])
        neighbors = []
        for pos in top_idx:
            row_id = int(pool_indices[pos])
            neighbors.append(
                {
                    "row_id": row_id,
                    "score": float(sims[pos]),
                    "category": str(val_df.iloc[row_id].get("main_category", "")),
                    "text_preview": str(val_df.iloc[row_id][text_column])[:180],
                }
            )

        checks.append(
            {
                "query_row": query_row,
                "query_category": str(val_df.iloc[query_row].get("main_category", "")),
                "query_text_preview": str(val_df.iloc[query_row][text_column])[:180],
                "neighbors": neighbors,
            }
        )

    return checks


def main() -> None:
    cli = parse_args()
    params = _read_yaml(cli.params)
    exp = _read_yaml(cli.config)
    cfg = build_config(params=params, exp=exp, cli=cli)

    set_seed(cfg.seed)

    train_df, train_text_column = load_split(
        path=cli.train_path,
        text_column=cfg.text_column,
    )
    val_df, val_text_column = load_split(
        path=cli.val_path,
        text_column=cfg.text_column,
    )
    test_df, test_text_column = load_split(
        path=cli.test_path,
        text_column=cfg.text_column,
    )

    # Enforce the allowed category scope before any optional row-capping.
    train_df = enforce_category_scope(train_df, cfg=cfg, split_name="train")
    val_df = enforce_category_scope(val_df, cfg=cfg, split_name="val")
    test_df = enforce_category_scope(test_df, cfg=cfg, split_name="test")

    train_df = apply_row_cap(train_df, max_rows=cfg.max_train_rows, seed=cfg.seed)
    val_df = apply_row_cap(val_df, max_rows=cfg.max_vector_rows, seed=cfg.seed)
    test_df = apply_row_cap(test_df, max_rows=cfg.max_vector_rows, seed=cfg.seed)

    cfg.text_column = train_text_column

    corpus = TokenizedCorpus(
        texts=train_df[train_text_column],
        min_tokens=cfg.min_review_tokens,
        max_tokens=cfg.max_review_tokens,
    )

    model = Word2Vec(
        vector_size=cfg.vector_size,
        window=cfg.window,
        min_count=cfg.min_count,
        sg=cfg.sg,
        negative=cfg.negative,
        workers=cfg.workers,
        seed=cfg.seed,
        epochs=cfg.epochs,
    )

    print("Building Word2Vec vocabulary...")
    model.build_vocab(corpus_iterable=corpus)
    print(f"Vocabulary size: {len(model.wv)}")

    print("Training Word2Vec...")
    model.train(corpus_iterable=corpus, total_examples=model.corpus_count, epochs=cfg.epochs)

    models_dir = ensure_dir(Path("artifacts") / "models")
    vectors_dir = ensure_dir(Path("artifacts") / "vectors")
    metrics_dir = ensure_dir(Path("artifacts") / "metrics")

    model_path = models_dir / f"{cfg.model_name}.model"
    kv_path = models_dir / f"{cfg.model_name}.kv"
    model.save(str(model_path))
    model.wv.save(str(kv_path))

    split_exports = []
    split_exports.append(
        export_split_vectors(
            model=model,
            df=train_df,
            split_name="train",
            text_column=train_text_column,
            cfg=cfg,
            vectors_dir=vectors_dir,
        )
    )
    split_exports.append(
        export_split_vectors(
            model=model,
            df=val_df,
            split_name="val",
            text_column=val_text_column,
            cfg=cfg,
            vectors_dir=vectors_dir,
        )
    )
    if not cli.skip_test_vectors:
        split_exports.append(
            export_split_vectors(
                model=model,
                df=test_df,
                split_name="test",
                text_column=test_text_column,
                cfg=cfg,
                vectors_dir=vectors_dir,
            )
        )

    word_checks = build_word_sanity(
        model=model,
        sanity_words=cfg.sanity_words,
        top_k=cfg.sanity_top_k,
    )

    val_vectors_path = vectors_dir / f"{cfg.model_name}_val_vectors.npy"
    doc_checks = build_document_sanity(
        val_df=val_df,
        val_vectors_path=val_vectors_path,
        text_column=val_text_column,
        seed=cfg.seed,
        pool_size=cfg.sanity_doc_pool_size,
        top_k=cfg.sanity_top_k,
    )

    summary = {
        "model_name": cfg.model_name,
        "config": {
            "architecture": cfg.architecture,
            "vector_size": cfg.vector_size,
            "window": cfg.window,
            "min_count": cfg.min_count,
            "negative": cfg.negative,
            "epochs": cfg.epochs,
            "workers": cfg.workers,
            "seed": cfg.seed,
            "text_column": cfg.text_column,
            "text_columns": {
                "train": train_text_column,
                "val": val_text_column,
                "test": test_text_column,
            },
            "category_column": cfg.category_column,
            "categories": _canonicalize_list(cfg.categories),
            "min_review_tokens": cfg.min_review_tokens,
            "max_review_tokens": cfg.max_review_tokens,
            "vector_batch_size": cfg.vector_batch_size,
            "max_train_rows": cfg.max_train_rows,
            "max_vector_rows": cfg.max_vector_rows,
            "strict_categories": cfg.strict_categories,
        },
        "dataset": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        },
        "vocabulary_size": len(model.wv),
        "model_path": str(model_path),
        "keyed_vectors_path": str(kv_path),
        "split_vector_exports": split_exports,
        "word_sanity_checks": word_checks,
        "document_sanity_checks": doc_checks,
    }

    summary_path = metrics_dir / f"{cfg.model_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sanity_txt = metrics_dir / f"{cfg.model_name}_sanity_checks.txt"
    with sanity_txt.open("w", encoding="utf-8") as handle:
        handle.write("Word-level nearest neighbors\n")
        handle.write("=" * 80 + "\n")
        for item in word_checks:
            handle.write(f"Word: {item['word']} | found={item['found']}\n")
            for neighbor in item["neighbors"]:
                handle.write(f"  - {neighbor['token']}: {neighbor['score']:.4f}\n")
            handle.write("\n")

        handle.write("Document-level nearest neighbors\n")
        handle.write("=" * 80 + "\n")
        for block in doc_checks:
            handle.write(f"Query row={block['query_row']} category={block['query_category']}\n")
            handle.write(f"Query text: {block['query_text_preview']}\n")
            for neighbor in block["neighbors"]:
                handle.write(
                    f"  - row={neighbor['row_id']} score={neighbor['score']:.4f} "
                    f"category={neighbor['category']}\n"
                )
                handle.write(f"    text: {neighbor['text_preview']}\n")
            handle.write("\n")

    print("Phase 3 outputs saved:")
    print(f"- model: {model_path}")
    print(f"- keyed vectors: {kv_path}")
    print(f"- summary: {summary_path}")
    print(f"- sanity checks: {sanity_txt}")
    for item in split_exports:
        print(f"- {item['split']} vectors: {item['vectors_path']}")


if __name__ == "__main__":
    main()
