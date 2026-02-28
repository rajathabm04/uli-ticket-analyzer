"""
Categorizer: uses Claude Sonnet to infer the correct support ticket category.

For each ticket, the subject, description, and conversation thread are sent
to Claude with a fixed list of ULI-specific categories. Claude returns exactly
one category name; unrecognised responses fall back to "Other".
"""

import time

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

_SYSTEM = (
    "You are a support ticket classifier for ULI (Unified Lending Interface), "
    "a financial data-sharing platform that connects lenders (banks, NBFCs, fintechs) "
    "with data providers.\n\n"
    "Given a support ticket, respond with ONLY the single most appropriate category "
    "from the list below — no explanation, no punctuation, just the category name "
    "exactly as written:\n\n"
    + "\n".join(f"- {c}" for c in CATEGORIES)
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def categorize_ticket(ticket: dict, client: anthropic.Anthropic) -> str:
    """
    Infer the correct category for a single ticket dict.

    Args:
        ticket: Dict with at least 'subject', 'description', 'conversations'.
        client: Initialised anthropic.Anthropic client.

    Returns:
        One of the strings in CATEGORIES.
    """
    subject = ticket.get("subject", "")
    description = ticket.get("description", "")
    conversations = ticket.get("conversations", "")

    content = f"Subject: {subject}\n\nDescription:\n{description}"
    if conversations:
        content += f"\n\nConversation (excerpt):\n{str(conversations)[:500]}"

    # Retry with exponential backoff on rate limit errors.
    # Rate limit window is 60 s, so each wait must exceed that.
    for attempt in range(5):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=32,
                system=_SYSTEM,
                messages=[{"role": "user", "content": content}],
            )
            break
        except anthropic.RateLimitError:
            if attempt == 4:
                raise
            time.sleep(60 * (attempt + 1))  # 60 s, 120 s, 180 s, 240 s

    raw = response.content[0].text.strip()

    # Case-insensitive match against known categories; fall back to "Other"
    for cat in CATEGORIES:
        if cat.lower() == raw.lower():
            return cat
    return "Other"


def categorize_all(df: pd.DataFrame, client: anthropic.Anthropic) -> pd.DataFrame:
    """
    Add an 'inferred_category' column to df by classifying each ticket with Claude.

    Args:
        df: DataFrame produced by loader.load_tickets().
        client: Initialised anthropic.Anthropic client.

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
