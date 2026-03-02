"""
Categorizer: keyword-based classifier for ULI support tickets.

Scores each ticket against per-category keyword lists and returns the
highest-scoring category. Falls back to "Other" when no keywords match.
No API calls — runs entirely in-process.
"""

import re

import anthropic
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn


# ---------------------------------------------------------------------------
# Valid categories for ULI platform support tickets
# ---------------------------------------------------------------------------

CATEGORIES: list[str] = [
    "API Error",
    "Authentication / Authorisation",
    "Onboarding",
    "Integration",
    "Data Mismatch",
    "Performance / Latency",
    "Configuration",
    "Documentation Request",
    "Other",
]


# ---------------------------------------------------------------------------
# Keyword rules — ordered by specificity within each category
# ---------------------------------------------------------------------------

_KEYWORDS: dict[str, list[str]] = {
    "API Error": [
        "api error", "api failure", "api down", "api issue", "api not working",
        "bad request", "internal server error", "invalid response", "response error",
        "http 400", "http 500", "http 502", "http 503", "status 400", "status 500",
        "error code", "error response", "request failed", "call failed",
    ],
    "Authentication / Authorisation": [
        "authentication failed", "authorisation failed", "authorization failed",
        "access denied", "access forbidden", "not authorized", "unauthorized",
        "invalid token", "token expired", "token invalid", "invalid credentials",
        "login failed", "cannot log in", "unable to login", "sign in failed",
        "permission denied", "403", "401", "api key invalid", "api key expired",
        "oauth", "jwt", "2fa", "otp", "credential", "password reset",
    ],
    "Onboarding": [
        "onboarding", "onboard", "new registration", "account registration",
        "account setup", "getting started", "initial setup", "new lender",
        "new user", "first time", "kyc", "verification pending", "not yet activated",
        "activation", "enroll", "sign up", "signup",
    ],
    "Integration": [
        "integration issue", "integration error", "integration failure",
        "webhook", "third-party", "third party", "partner system",
        "data flow", "middleware", "sdk issue", "library", "incompatible",
        "lender integration", "connecting to", "connection refused", "cannot connect",
        "integrate with", "sync issue", "data sync",
    ],
    "Data Mismatch": [
        "data mismatch", "mismatch", "discrepancy", "incorrect data",
        "wrong data", "data inconsistency", "inconsistent data",
        "does not match", "doesn't match", "different value",
        "inaccurate", "incorrect value", "data quality", "wrong amount",
        "wrong field", "data error", "field mismatch",
    ],
    "Performance / Latency": [
        "slow response", "high latency", "response time", "timed out",
        "timeout", "too slow", "performance issue", "performance degraded",
        "throughput", "bottleneck", "lag", "delay", "sla breach",
        "taking long", "takes too long", "very slow",
    ],
    "Configuration": [
        "misconfigured", "misconfiguration", "wrong configuration",
        "config error", "configuration issue", "configuration error",
        "wrong setting", "incorrect setting", "parameter",
        "environment variable", "env variable", "flag", "property value",
        "needs to be configured", "not configured",
    ],
    "Documentation Request": [
        "documentation", "where is the doc", "need docs", "need documentation",
        "api reference", "swagger", "openapi", "api spec", "specification",
        "how to", "howto", "tutorial", "guide", "example", "sample code",
        "readme", "faq", "manual",
    ],
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_ticket(text: str) -> dict[str, int]:
    """Return a match count per category for the given text (lowercased)."""
    lower = text.lower()
    scores: dict[str, int] = {}
    for cat, phrases in _KEYWORDS.items():
        scores[cat] = sum(1 for phrase in phrases if phrase in lower)
    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def categorize_ticket(ticket: dict, client: anthropic.Anthropic) -> str:  # noqa: ARG001
    """
    Infer the correct category for a single ticket dict using keyword rules.

    Args:
        ticket: Dict with at least 'subject', 'description', 'conversations'.
        client: Unused — kept for API compatibility with the rest of the pipeline.

    Returns:
        One of the strings in CATEGORIES.
    """
    subject = str(ticket.get("subject", ""))
    description = str(ticket.get("description", ""))
    conversations = str(ticket.get("conversations", ""))[:500]

    text = f"{subject}\n{description}\n{conversations}"
    scores = _score_ticket(text)

    best_cat = max(scores, key=lambda c: scores[c])
    if scores[best_cat] == 0:
        return "Other"
    return best_cat


def categorize_all(df: pd.DataFrame, client: anthropic.Anthropic) -> pd.DataFrame:
    """
    Add an 'inferred_category' column to df by classifying each ticket.

    Args:
        df: DataFrame produced by loader.load_tickets().
        client: Unused — kept for API compatibility.

    Returns:
        New DataFrame with an additional 'inferred_category' column.
    """
    df = df.copy()
    inferred = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Categorizing tickets..."),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} tickets"),
    ) as progress:
        task = progress.add_task("categorizing", total=len(df))
        for _, row in df.iterrows():
            inferred.append(categorize_ticket(row.to_dict(), client))
            progress.advance(task)

    df["inferred_category"] = inferred
    return df
