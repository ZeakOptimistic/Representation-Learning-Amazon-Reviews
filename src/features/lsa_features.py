"""LSA feature utilities."""

from sklearn.decomposition import TruncatedSVD


def build_lsa(n_components: int = 300) -> TruncatedSVD:
    """Return a configured truncated SVD transformer."""
    return TruncatedSVD(n_components=n_components, random_state=42)
