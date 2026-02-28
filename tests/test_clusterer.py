"""
Tests for src/clusterer.py
"""

import pandas as pd
import pytest

from src.clusterer import cluster_summaries, cluster_tickets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame as loader.load_tickets() would produce."""
    defaults = {
        "id": 0, "subject": "", "description": "",
        "conversations": "", "category": "", "agent": "",
        "status": "open", "created_at": "",
    }
    if not rows:
        cols = list(defaults.keys())
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _tickets(texts: list[str]) -> pd.DataFrame:
    """Shorthand: create a DataFrame where each text becomes the subject."""
    return _df([{"id": i, "subject": t} for i, t in enumerate(texts)])


# Diverse enough that KMeans can split them into meaningful groups
_API_TEXTS = [
    "API returns 500 error on loan apply endpoint",
    "500 internal server error calling the loan API",
    "REST endpoint crashes with server error 500",
    "HTTP 500 on POST /loan/apply response",
]
_AUTH_TEXTS = [
    "JWT token expired cannot authenticate",
    "authentication failed invalid bearer token",
    "token validation error on login request",
    "OAuth token rejected by the auth service",
]


# ---------------------------------------------------------------------------
# cluster_tickets
# ---------------------------------------------------------------------------

class TestClusterTickets:

    def test_returns_dataframe(self):
        assert isinstance(cluster_tickets(_tickets(["hello"])), pd.DataFrame)

    def test_adds_cluster_column(self):
        df = _tickets(["ticket one", "ticket two"])
        result = cluster_tickets(df, n_clusters=1)
        assert "cluster" in result.columns

    def test_cluster_column_is_integer(self):
        df = _tickets(_API_TEXTS)
        result = cluster_tickets(df, n_clusters=2)
        assert result["cluster"].dtype == int or pd.api.types.is_integer_dtype(result["cluster"])

    def test_row_count_unchanged(self):
        df = _tickets(_API_TEXTS)
        result = cluster_tickets(df, n_clusters=2)
        assert len(result) == len(_API_TEXTS)

    def test_original_df_not_mutated(self):
        df = _tickets(_API_TEXTS[:2])
        original_cols = list(df.columns)
        cluster_tickets(df, n_clusters=1)
        assert "cluster" not in df.columns
        assert list(df.columns) == original_cols

    def test_cluster_labels_within_range(self):
        df = _tickets(_API_TEXTS + _AUTH_TEXTS)
        n = 3
        result = cluster_tickets(df, n_clusters=n)
        assert result["cluster"].between(0, n - 1).all()

    def test_n_clusters_capped_at_ticket_count(self):
        # 2 tickets, requesting 10 clusters — should not raise
        df = _tickets(["ticket one", "ticket two"])
        result = cluster_tickets(df, n_clusters=10)
        assert len(result) == 2
        assert result["cluster"].nunique() <= 2

    def test_single_ticket_gets_cluster_zero(self):
        df = _tickets(["only one ticket"])
        result = cluster_tickets(df, n_clusters=1)
        assert result.iloc[0]["cluster"] == 0

    def test_empty_df_returns_cluster_column(self):
        df = _df([])
        result = cluster_tickets(df, n_clusters=3)
        assert "cluster" in result.columns
        assert len(result) == 0

    def test_similar_tickets_tend_to_cluster_together(self):
        """API-error tickets and auth tickets should mostly land in different clusters."""
        df = _df([
            {"id": i, "subject": t}
            for i, t in enumerate(_API_TEXTS + _AUTH_TEXTS)
        ])
        result = cluster_tickets(df, n_clusters=2)
        api_clusters = set(result.iloc[:4]["cluster"])
        auth_clusters = set(result.iloc[4:]["cluster"])
        # At least one cluster is predominantly one theme
        assert len(api_clusters) == 1 or len(auth_clusters) == 1

    def test_uses_description_and_conversations(self):
        # No subject text — clustering should still work via description
        df = _df([
            {"id": i, "subject": "", "description": t}
            for i, t in enumerate(_API_TEXTS + _AUTH_TEXTS)
        ])
        result = cluster_tickets(df, n_clusters=2)
        assert "cluster" in result.columns
        assert len(result) == len(_API_TEXTS) + len(_AUTH_TEXTS)

    def test_other_columns_preserved(self):
        df = _df([{"id": 42, "subject": "test", "status": "resolved"}])
        result = cluster_tickets(df, n_clusters=1)
        assert result.iloc[0]["id"] == 42
        assert result.iloc[0]["status"] == "resolved"


# ---------------------------------------------------------------------------
# cluster_summaries
# ---------------------------------------------------------------------------

class TestClusterSummaries:

    def test_returns_dataframe(self):
        df = cluster_tickets(_tickets(_API_TEXTS), n_clusters=1)
        assert isinstance(cluster_summaries(df), pd.DataFrame)

    def test_output_columns(self):
        df = cluster_tickets(_tickets(_API_TEXTS), n_clusters=1)
        result = cluster_summaries(df)
        assert list(result.columns) == ["cluster", "size", "top_terms"]

    def test_one_row_per_cluster(self):
        df = _df([{"id": i, "subject": t} for i, t in enumerate(_API_TEXTS + _AUTH_TEXTS)])
        df = cluster_tickets(df, n_clusters=2)
        result = cluster_summaries(df)
        assert len(result) == 2

    def test_size_sums_to_total_tickets(self):
        df = _df([{"id": i, "subject": t} for i, t in enumerate(_API_TEXTS + _AUTH_TEXTS)])
        df = cluster_tickets(df, n_clusters=2)
        result = cluster_summaries(df)
        assert result["size"].sum() == len(_API_TEXTS) + len(_AUTH_TEXTS)

    def test_top_terms_is_string(self):
        df = cluster_tickets(_tickets(["API error on endpoint"]), n_clusters=1)
        result = cluster_summaries(df)
        assert isinstance(result.iloc[0]["top_terms"], str)

    def test_sorted_by_cluster_id(self):
        df = _df([{"id": i, "subject": t} for i, t in enumerate(_API_TEXTS + _AUTH_TEXTS)])
        df = cluster_tickets(df, n_clusters=2)
        result = cluster_summaries(df)
        assert list(result["cluster"]) == sorted(result["cluster"])

    def test_index_is_reset(self):
        df = _df([{"id": i, "subject": t} for i, t in enumerate(_API_TEXTS + _AUTH_TEXTS)])
        df = cluster_tickets(df, n_clusters=2)
        result = cluster_summaries(df)
        assert list(result.index) == list(range(len(result)))

    def test_empty_df_returns_empty_with_columns(self):
        df = _df([])
        df["cluster"] = pd.Series(dtype=int)
        result = cluster_summaries(df)
        assert len(result) == 0
        assert list(result.columns) == ["cluster", "size", "top_terms"]

    def test_missing_cluster_column_raises(self):
        df = _tickets(["a", "b"])
        with pytest.raises(ValueError, match="cluster"):
            cluster_summaries(df)

    def test_n_terms_respected(self):
        texts = _API_TEXTS + _AUTH_TEXTS
        df = _df([{"id": i, "subject": t} for i, t in enumerate(texts)])
        df = cluster_tickets(df, n_clusters=1)
        result = cluster_summaries(df, n_terms=3)
        terms = result.iloc[0]["top_terms"].split(", ")
        assert len(terms) <= 3
