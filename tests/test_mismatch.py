"""
Tests for src/mismatch.py
"""

import pandas as pd
import pytest

from src.mismatch import find_mismatches, mismatch_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLS = ["id", "subject", "category", "inferred_category", "agent", "status", "created_at"]
_DEFAULTS = {
    "id": 0, "subject": "", "category": "",
    "inferred_category": "", "agent": "", "status": "open", "created_at": "",
}

def _df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame with the columns categorize_all produces."""
    if not rows:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame([{**_DEFAULTS, **r} for r in rows])


# ---------------------------------------------------------------------------
# find_mismatches
# ---------------------------------------------------------------------------

class TestFindMismatches:

    def test_returns_dataframe(self):
        assert isinstance(find_mismatches(_df([])), pd.DataFrame)

    def test_empty_input_returns_empty(self):
        result = find_mismatches(_df([]))
        assert len(result) == 0

    def test_output_columns(self):
        result = find_mismatches(_df([]))
        assert list(result.columns) == [
            "id", "subject", "assigned_category", "inferred_category",
            "agent", "status", "created_at",
        ]

    def test_category_renamed_to_assigned_category(self):
        df = _df([{"category": "API Error", "inferred_category": "Onboarding"}])
        result = find_mismatches(df)
        assert "assigned_category" in result.columns
        assert "category" not in result.columns

    def test_all_matching_returns_empty(self):
        df = _df([
            {"category": "API Error", "inferred_category": "API Error"},
            {"category": "Onboarding", "inferred_category": "Onboarding"},
        ])
        assert len(find_mismatches(df)) == 0

    def test_all_mismatching_returns_all(self):
        df = _df([
            {"id": 1, "category": "API Error", "inferred_category": "Onboarding"},
            {"id": 2, "category": "Integration", "inferred_category": "Data Mismatch"},
        ])
        result = find_mismatches(df)
        assert len(result) == 2

    def test_only_mismatches_returned(self):
        df = _df([
            {"id": 1, "category": "API Error", "inferred_category": "API Error"},
            {"id": 2, "category": "Integration", "inferred_category": "Onboarding"},
            {"id": 3, "category": "Onboarding", "inferred_category": "Onboarding"},
        ])
        result = find_mismatches(df)
        assert len(result) == 1
        assert result.iloc[0]["id"] == 2

    def test_case_insensitive_comparison(self):
        # Same category, different casing — should NOT be a mismatch
        df = _df([{"category": "API Error", "inferred_category": "api error"}])
        assert len(find_mismatches(df)) == 0

    def test_whitespace_normalised(self):
        # Leading/trailing whitespace should not create false positives
        df = _df([{"category": "  Onboarding  ", "inferred_category": "Onboarding"}])
        assert len(find_mismatches(df)) == 0

    def test_empty_assigned_vs_inferred_is_mismatch(self):
        df = _df([{"category": "", "inferred_category": "API Error"}])
        result = find_mismatches(df)
        assert len(result) == 1

    def test_nan_assigned_vs_inferred_is_mismatch(self):
        df = _df([{"category": None, "inferred_category": "Onboarding"}])
        result = find_mismatches(df)
        assert len(result) == 1

    def test_both_empty_is_not_mismatch(self):
        df = _df([{"category": "", "inferred_category": ""}])
        assert len(find_mismatches(df)) == 0

    def test_index_is_reset(self):
        df = _df([
            {"id": 10, "category": "X", "inferred_category": "Y"},
            {"id": 11, "category": "A", "inferred_category": "A"},
            {"id": 12, "category": "B", "inferred_category": "C"},
        ])
        result = find_mismatches(df)
        assert list(result.index) == [0, 1]

    def test_row_data_is_correct(self):
        df = _df([{
            "id": 99, "subject": "test subject",
            "category": "API Error", "inferred_category": "Integration",
            "agent": "Alice", "status": "open", "created_at": "2024-01-01",
        }])
        row = find_mismatches(df).iloc[0]
        assert row["id"] == 99
        assert row["subject"] == "test subject"
        assert row["assigned_category"] == "API Error"
        assert row["inferred_category"] == "Integration"
        assert row["agent"] == "Alice"
        assert row["status"] == "open"
        assert row["created_at"] == "2024-01-01"

    def test_missing_inferred_category_column_raises(self):
        df = pd.DataFrame([{"id": 1, "category": "API Error"}])
        with pytest.raises(ValueError, match="inferred_category"):
            find_mismatches(df)

    def test_input_df_not_mutated(self):
        df = _df([{"category": "API Error", "inferred_category": "Onboarding"}])
        original_cols = list(df.columns)
        find_mismatches(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# mismatch_rate
# ---------------------------------------------------------------------------

class TestMismatchRate:

    def test_empty_df_returns_zero(self):
        assert mismatch_rate(_df([])) == 0.0

    def test_no_mismatches_returns_zero(self):
        df = _df([
            {"category": "API Error", "inferred_category": "API Error"},
            {"category": "Onboarding", "inferred_category": "Onboarding"},
        ])
        assert mismatch_rate(df) == 0.0

    def test_all_mismatches_returns_one(self):
        df = _df([
            {"category": "API Error", "inferred_category": "Onboarding"},
            {"category": "Integration", "inferred_category": "Other"},
        ])
        assert mismatch_rate(df) == 1.0

    def test_half_mismatches(self):
        df = _df([
            {"category": "API Error", "inferred_category": "API Error"},
            {"category": "Integration", "inferred_category": "Onboarding"},
        ])
        assert mismatch_rate(df) == 0.5

    def test_returns_float(self):
        assert isinstance(mismatch_rate(_df([])), float)
