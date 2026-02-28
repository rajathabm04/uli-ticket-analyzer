"""
KB Generator: generates markdown KB articles from ticket clusters.

For each cluster, Claude Sonnet produces two articles:
  - Internal  (output/internal/)      : detailed, technical, for support agents
  - Lender-facing (output/lender-facing/) : simplified, self-serve, for lenders
"""

import os

import anthropic
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

_OUTPUT_DIRS = {
    "internal": "output/internal",
    "lender":   "output/lender-facing",
}

# ---------------------------------------------------------------------------
# Claude system prompts
# ---------------------------------------------------------------------------

_INTERNAL_SYSTEM = (
    "You are a technical writer creating internal KB articles for support agents "
    "at RBIH (Reserve Bank Innovation Hub) who handle ULI (Unified Lending Interface) "
    "platform tickets.\n\n"
    "Write a detailed, structured markdown article that helps agents diagnose and "
    "resolve this class of issue. Use exactly these sections:\n\n"
    "## Issue Summary\n"
    "## Common Root Causes\n"
    "## Diagnostic Steps\n"
    "## Resolution\n"
    "## Escalation Path\n\n"
    "Be technical and specific. Use markdown throughout."
)

_LENDER_SYSTEM = (
    "You are a technical writer creating self-serve KB articles for lenders "
    "(banks, NBFCs, fintechs) integrating with the ULI (Unified Lending Interface) "
    "platform.\n\n"
    "Write a clear, concise markdown article that helps lenders understand and "
    "resolve this issue themselves. Use exactly these sections:\n\n"
    "## What is this issue?\n"
    "## How to resolve it\n"
    "## When to contact support\n\n"
    "Keep language simple. Avoid internal jargon. Use markdown throughout."
)

_MAX_SAMPLE = 5   # tickets per cluster included in the prompt
_DESC_LIMIT  = 500  # chars of description to include
_CONV_LIMIT  = 300  # chars of conversation snippet to include


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_tickets(df: pd.DataFrame, cluster_id: int, n: int = _MAX_SAMPLE) -> pd.DataFrame:
    """Return up to n tickets from the given cluster, in their original order."""
    return df[df["cluster"] == cluster_id].head(n)


def _build_prompt(top_terms: str, sample: pd.DataFrame) -> str:
    """Build the user prompt from cluster top terms and sampled tickets."""
    lines = [
        f"Recurring issue theme (top terms): {top_terms}\n",
        f"Sample tickets ({len(sample)} examples):\n",
    ]
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        lines.append(f"--- Ticket {i} ---")
        lines.append(f"Subject: {row.get('subject', '')}")
        desc = str(row.get("description", ""))[:_DESC_LIMIT]
        if desc:
            lines.append(f"Description: {desc}")
        conv = str(row.get("conversations", ""))[:_CONV_LIMIT]
        if conv:
            lines.append(f"Conversation snippet: {conv}")
        lines.append("")
    return "\n".join(lines)


def _generate_article(prompt: str, system: str, client: anthropic.Anthropic) -> str:
    """Call Claude Sonnet to generate one KB article."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1_024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _write_article(content: str, path: str) -> None:
    """Write article content to path, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_kb_articles(
    df: pd.DataFrame,
    summaries: pd.DataFrame,
    client: anthropic.Anthropic,
) -> list[dict]:
    """
    Generate internal and lender-facing KB articles for every cluster.

    For each cluster in `summaries`, samples up to 5 tickets from `df`,
    calls Claude Sonnet twice (once per audience), and writes markdown
    files to output/internal/ and output/lender-facing/.

    Args:
        df: DataFrame with 'cluster' column (from clusterer.cluster_tickets()).
        summaries: DataFrame with columns cluster, size, top_terms
            (from clusterer.cluster_summaries()).
        client: Initialised anthropic.Anthropic client.

    Returns:
        List of dicts, one per cluster:
            {cluster, size, top_terms, internal_path, lender_path}
    """
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Generating KB articles..."),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} clusters"),
    ) as progress:
        task = progress.add_task("generating", total=len(summaries))

        for _, summary_row in summaries.iterrows():
            cluster_id = int(summary_row["cluster"])
            top_terms  = str(summary_row["top_terms"])
            size       = int(summary_row["size"])

            sample  = _sample_tickets(df, cluster_id)
            prompt  = _build_prompt(top_terms, sample)

            internal_article = _generate_article(prompt, _INTERNAL_SYSTEM, client)
            lender_article   = _generate_article(prompt, _LENDER_SYSTEM,   client)

            internal_path = os.path.join(_OUTPUT_DIRS["internal"], f"cluster_{cluster_id}.md")
            lender_path   = os.path.join(_OUTPUT_DIRS["lender"],   f"cluster_{cluster_id}.md")

            _write_article(internal_article, internal_path)
            _write_article(lender_article,   lender_path)

            results.append({
                "cluster":       cluster_id,
                "size":          size,
                "top_terms":     top_terms,
                "internal_path": internal_path,
                "lender_path":   lender_path,
            })

            progress.advance(task)

    return results
