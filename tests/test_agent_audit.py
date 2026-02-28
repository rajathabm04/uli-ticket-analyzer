"""
Tests for src/agent_audit.py
"""

import pandas as pd
import pytest

from src.agent_audit import error_patterns, summarise_by_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLS = ["id", "subject", "assigned_category", "inferred_category", "agent", "status", "created_at"]
_DEFAULTS = {
    "id": 0, "subject": "", "assigned_category": "", "inferred_category": "",
    "agent": "", "status": "open", "created_at": "",
}

def _df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame([{**_DEFAULTS, **r} for r in rows])


# ---------------------------------------------------------------------------
# summarise_by_agent
# ---------------------------------------------------------------------------

class TestSummariseByAgent:

    def test_returns_dataframe(self):
        assert isinstance(summarise_by_agent(_df([])), pd.DataFrame)

    def test_empty_input_returns_empty_with_columns(self):
        result = summarise_by_agent(_df([]))
        assert len(result) == 0
        assert list(result.columns) == ["agent", "mismatch_count"]

    def test_single_agent_single_mismatch(self):
        df = _df([{"agent": "Alice"}])
        result = summarise_by_agent(df)
        assert len(result) == 1
        assert result.iloc[0]["agent"] == "Alice"
        assert result.iloc[0]["mismatch_count"] == 1

    def test_single_agent_multiple_mismatches(self):
        df = _df([{"agent": "Alice"}, {"agent": "Alice"}, {"agent": "Alice"}])
        result = summarise_by_agent(df)
        assert len(result) == 1
        assert result.iloc[0]["mismatch_count"] == 3

    def test_multiple_agents_counted_separately(self):
        df = _df([{"agent": "Alice"}, {"agent": "Bob"}, {"agent": "Alice"}])
        result = summarise_by_agent(df)
        counts = dict(zip(result["agent"], result["mismatch_count"]))
        assert counts["Alice"] == 2
        assert counts["Bob"] == 1

    def test_sorted_by_mismatch_count_descending(self):
        df = _df([
            {"agent": "Bob"},
            {"agent": "Alice"}, {"agent": "Alice"}, {"agent": "Alice"},
            {"agent": "Charlie"}, {"agent": "Charlie"},
        ])
        result = summarise_by_agent(df)
        assert list(result["agent"]) == ["Alice", "Charlie", "Bob"]
        assert list(result["mismatch_count"]) == [3, 2, 1]

    def test_index_is_reset(self):
        df = _df([{"agent": "Alice"}, {"agent": "Bob"}])
        result = summarise_by_agent(df)
        assert list(result.index) == list(range(len(result)))

    def test_output_columns(self):
        df = _df([{"agent": "Alice"}])
        assert list(summarise_by_agent(df).columns) == ["agent", "mismatch_count"]

    def test_input_not_mutated(self):
        df = _df([{"agent": "Alice"}])
        original_cols = list(df.columns)
        summarise_by_agent(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# error_patterns
# ---------------------------------------------------------------------------

class TestErrorPatterns:

    def test_returns_dataframe(self):
        assert isinstance(error_patterns(_df([])), pd.DataFrame)

    def test_empty_input_returns_empty_with_columns(self):
        result = error_patterns(_df([]))
        assert len(result) == 0
        assert list(result.columns) == ["agent", "assigned_category", "inferred_category", "count"]

    def test_single_pattern_count_one(self):
        df = _df([{"agent": "Alice", "assigned_category": "API Error", "inferred_category": "Integration"}])
        result = error_patterns(df)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["agent"] == "Alice"
        assert row["assigned_category"] == "API Error"
        assert row["inferred_category"] == "Integration"
        assert row["count"] == 1

    def test_same_pattern_repeated_aggregated(self):
        df = _df([
            {"agent": "Alice", "assigned_category": "API Error", "inferred_category": "Integration"},
            {"agent": "Alice", "assigned_category": "API Error", "inferred_category": "Integration"},
            {"agent": "Alice", "assigned_category": "API Error", "inferred_category": "Integration"},
        ])
        result = error_patterns(df)
        assert len(result) == 1
        assert result.iloc[0]["count"] == 3

    def test_different_patterns_same_agent_kept_separate(self):
        df = _df([
            {"agent": "Alice", "assigned_category": "API Error", "inferred_category": "Integration"},
            {"agent": "Alice", "assigned_category": "Onboarding", "inferred_category": "Configuration"},
        ])
        result = error_patterns(df)
        assert len(result) == 2

    def test_sorted_by_agent_asc_then_count_desc(self):
        df = _df([
            {"agent": "Bob",   "assigned_category": "X", "inferred_category": "Y"},
            {"agent": "Alice", "assigned_category": "A", "inferred_category": "B"},
            {"agent": "Alice", "assigned_category": "A", "inferred_category": "B"},
            {"agent": "Alice", "assigned_category": "C", "inferred_category": "D"},
        ])
        result = error_patterns(df)
        # Alice comes before Bob
        assert result.iloc[0]["agent"] == "Alice"
        assert result.iloc[1]["agent"] == "Alice"
        assert result.iloc[2]["agent"] == "Bob"
        # Alice's most frequent pattern (count=2) comes first
        assert result.iloc[0]["count"] == 2
        assert result.iloc[1]["count"] == 1

    def test_multiple_agents_patterns_grouped(self):
        df = _df([
            {"agent": "Alice", "assigned_category": "API Error", "inferred_category": "Auth"},
            {"agent": "Bob",   "assigned_category": "API Error", "inferred_category": "Auth"},
            {"agent": "Bob",   "assigned_category": "API Error", "inferred_category": "Auth"},
        ])
        result = error_patterns(df)
        counts = {row["agent"]: row["count"] for _, row in result.iterrows()}
        assert counts["Alice"] == 1
        assert counts["Bob"] == 2

    def test_index_is_reset(self):
        df = _df([
            {"agent": "Alice", "assigned_category": "X", "inferred_category": "Y"},
            {"agent": "Bob",   "assigned_category": "X", "inferred_category": "Y"},
        ])
        result = error_patterns(df)
        assert list(result.index) == list(range(len(result)))

    def test_output_columns(self):
        df = _df([{"agent": "Alice", "assigned_category": "X", "inferred_category": "Y"}])
        assert list(error_patterns(df).columns) == [
            "agent", "assigned_category", "inferred_category", "count"
        ]

    def test_input_not_mutated(self):
        df = _df([{"agent": "Alice", "assigned_category": "X", "inferred_category": "Y"}])
        original_cols = list(df.columns)
        error_patterns(df)
        assert list(df.columns) == original_cols
