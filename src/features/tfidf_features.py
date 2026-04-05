"""TF-IDF feature utilities."""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 100000, ngram_range: tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    """Return a configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=5,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
    )
