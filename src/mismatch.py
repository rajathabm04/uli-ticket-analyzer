"""
Mismatch detector: compares Freshdesk-assigned categories against
Claude-inferred categories and returns the discrepant tickets.
"""

import pandas as pd


def find_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare the assigned category (from Freshdesk) against the inferred
    category (from Claude) and return only the tickets where they differ.

    Comparison is case-insensitive and whitespace-normalised so minor
    formatting differences in the Freshdesk-assigned value don't create
    false positives.

    Args:
        df: DataFrame produced by categorizer.categorize_all(); must contain
            'category' and 'inferred_category' columns.

    Returns:
        DataFrame with columns:
            id, subject, assigned_category, inferred_category,
            agent, status, created_at
        Row order matches the input. Index is reset.

    Raises:
        ValueError: if 'inferred_category' is not present in df.
    """
    if "inferred_category" not in df.columns:
        raise ValueError(
            "DataFrame is missing 'inferred_category'. "
            "Run categorizer.categorize_all() before calling find_mismatches()."
        )

    assigned_norm = df["category"].fillna("").str.strip().str.lower()
    inferred_norm = df["inferred_category"].fillna("").str.strip().str.lower()

    # Only flag tickets where the agent actually set a category AND it differs
    # from the inferred one. Blank/unset assigned category is not a mismatch.
    mismatch_mask = assigned_norm.ne("") & assigned_norm.ne(inferred_norm)

    result = (
        df[mismatch_mask][["id", "subject", "category", "inferred_category", "agent", "status", "created_at"]]
        .rename(columns={"category": "assigned_category"})
        .reset_index(drop=True)
    )

    return result


def mismatch_rate(df: pd.DataFrame) -> float:
    """
    Return the fraction of tickets where assigned != inferred category.

    Returns 0.0 for an empty DataFrame.
    """
    if len(df) == 0:
        return 0.0
    return len(find_mismatches(df)) / len(df)
