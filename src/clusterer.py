"""
Clusterer: groups tickets by recurring issue using TF-IDF + KMeans.

Produces a 'cluster' label per ticket and per-cluster summaries
(size + top TF-IDF terms) for passing to the KB generator.
"""

import html
import re

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS


# Strip PII placeholder tokens inserted by pii_masker before TF-IDF sees them.
# Otherwise [ORG_NAME], [PERSON_NAME] etc. dominate every cluster's top terms.
_PII_RE = re.compile(
    r'\[('
    r'ORG_NAME|PERSON_NAME|EMAIL|PHONE|PAN|AADHAAR|BANK_ACCOUNT|IFSC|IP_ADDRESS'
    r'|JWT_TOKEN|BEARER_TOKEN|CLIENT_ID|CLIENT_SECRET'
    r'|SP_CLIENT_TOKEN|SP_CLIENT_ID|API_KEY|OAUTH_TOKEN'
    r')\]',
    re.IGNORECASE,
)

# Extra stop words beyond sklearn's English list: email/URL fragments and
# Freshdesk boilerplate that carry no issue-specific signal.
_EXTRA_STOP_WORDS: frozenset[str] = frozenset({
    "https", "http", "www", "mailto", "com", "net", "org", "co",
    "regards", "dear", "hello", "hi", "thanks", "thank",
    "intended", "recipient", "confidential",
    "ticket", "id", "sr", "re", "fw", "fwd",
})

_STOP_WORDS: list[str] = sorted(frozenset(ENGLISH_STOP_WORDS) | _EXTRA_STOP_WORDS)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Strip HTML tags, decode entities, and remove PII mask tokens."""
    text = _HTML_TAG_RE.sub(" ", text)          # <div>, <span style="...">, etc.
    text = html.unescape(text)                   # &amp; &nbsp; &lt; &gt; &#39; …
    text = _PII_RE.sub(" ", text)               # [PERSON_NAME], [EMAIL], etc.
    return _WHITESPACE_RE.sub(" ", text).strip()


def _build_corpus(df: pd.DataFrame) -> list[str]:
    """Combine subject, description, and conversations into one string per ticket."""
    raw = (
        df["subject"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["conversations"].fillna("")
    ).str.strip()
    return [_clean(t) for t in raw.tolist()]


def cluster_tickets(df: pd.DataFrame, n_clusters: int = 8) -> pd.DataFrame:
    """
    Cluster tickets by recurring issue using TF-IDF + KMeans.

    If n_clusters exceeds the number of tickets it is capped automatically.

    Args:
        df: DataFrame from loader / categorizer (must have subject, description,
            conversations columns).
        n_clusters: Target number of clusters (default 8).

    Returns:
        Copy of df with an additional integer 'cluster' column (0-indexed).
    """
    result = df.copy()

    if df.empty:
        result["cluster"] = pd.Series(dtype=int)
        return result

    n = min(n_clusters, len(df))
    corpus = _build_corpus(df)

    vectorizer = TfidfVectorizer(stop_words=_STOP_WORDS, max_features=5_000)
    try:
        X = vectorizer.fit_transform(corpus)
    except ValueError:
        # Corpus too short to survive stop-word removal — retry without filter
        vectorizer = TfidfVectorizer(max_features=5_000)
        X = vectorizer.fit_transform(corpus)

    model = KMeans(n_clusters=n, random_state=42, n_init=10)
    result["cluster"] = model.fit_predict(X)

    return result


def cluster_summaries(df: pd.DataFrame, n_terms: int = 10) -> pd.DataFrame:
    """
    Compute per-cluster top TF-IDF terms and ticket count.

    TF-IDF is fitted independently on each cluster's documents so the terms
    reflect issues specific to that cluster rather than the whole corpus.

    Args:
        df: DataFrame with a 'cluster' column (from cluster_tickets()).
        n_terms: Number of top terms to extract per cluster (default 10).

    Returns:
        DataFrame with columns: cluster, size, top_terms (comma-separated).
        Sorted by cluster id. Index is reset.

    Raises:
        ValueError: if 'cluster' column is not present in df.
    """
    if "cluster" not in df.columns:
        raise ValueError(
            "DataFrame is missing 'cluster' column. "
            "Run cluster_tickets() before cluster_summaries()."
        )

    if df.empty:
        return pd.DataFrame(columns=["cluster", "size", "top_terms"])

    rows = []
    for cluster_id, group in df.groupby("cluster"):
        corpus = _build_corpus(group)
        vectorizer = TfidfVectorizer(stop_words=_STOP_WORDS, max_features=1_000)
        try:
            X = vectorizer.fit_transform(corpus)
            scores = X.sum(axis=0).A1
            top_indices = scores.argsort()[::-1][:n_terms]
            terms = list(vectorizer.get_feature_names_out()[top_indices])
        except ValueError:
            # All tokens removed by stop-word filter (e.g. very short text)
            terms = []

        rows.append({
            "cluster": int(cluster_id),
            "size": len(group),
            "top_terms": ", ".join(terms),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("cluster")
        .reset_index(drop=True)
    )
