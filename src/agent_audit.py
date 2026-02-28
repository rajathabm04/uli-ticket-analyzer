"""
Agent audit: aggregates category mismatches per agent to surface
who is miscategorising and what their specific error patterns look like.
"""

import pandas as pd


def summarise_by_agent(mismatches: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-agent mismatch count, sorted worst-first.

    Args:
        mismatches: DataFrame from mismatch.find_mismatches(); expected
            columns: agent, assigned_category, inferred_category.

    Returns:
        DataFrame with columns: agent, mismatch_count
        Sorted by mismatch_count descending. Index is reset.
    """
    if mismatches.empty:
        return pd.DataFrame(columns=["agent", "mismatch_count"])

    return (
        mismatches.groupby("agent", sort=False)
        .size()
        .reset_index(name="mismatch_count")
        .sort_values("mismatch_count", ascending=False)
        .reset_index(drop=True)
    )


def error_patterns(mismatches: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-agent breakdown of (assigned → inferred) category pairs,
    surfacing systematic miscategorisation patterns.

    Args:
        mismatches: DataFrame from mismatch.find_mismatches().

    Returns:
        DataFrame with columns: agent, assigned_category, inferred_category, count
        Sorted by agent asc, count desc. Index is reset.
    """
    if mismatches.empty:
        return pd.DataFrame(
            columns=["agent", "assigned_category", "inferred_category", "count"]
        )

    return (
        mismatches.groupby(
            ["agent", "assigned_category", "inferred_category"], sort=False
        )
        .size()
        .reset_index(name="count")
        .sort_values(["agent", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
