#!/usr/bin/env python3
"""
ULI Ticket Analyzer — entry point.
Usage: python analyze_tickets.py [--sample N]
"""

import argparse
import os
import sys

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.freshdesk import FreshdeskClient
from src.loader import load_tickets
from src.categorizer import categorize_all
from src.mismatch import find_mismatches, mismatch_rate
from src.agent_audit import summarise_by_agent, error_patterns
from src.clusterer import cluster_tickets, cluster_summaries
from src.kb_generator import generate_kb_articles

load_dotenv()

console = Console()


def _get_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        console.print(f"[red]Error:[/red] {name} is not set. Add it to your .env file.")
        sys.exit(1)
    return val


def main() -> None:
    parser = argparse.ArgumentParser(description="ULI Ticket Analyzer")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Process only the first N tickets (for quick testing)")
    parser.add_argument("--clusters", type=int, default=20, metavar="N",
                        help="Number of clusters for KB generation (default: 20)")
    args = parser.parse_args()

    anthropic_api_key = _get_env("ANTHROPIC_API_KEY")
    freshdesk_domain = _get_env("FRESHDESK_DOMAIN")
    freshdesk_api_key = _get_env("FRESHDESK_API_KEY")

    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
    freshdesk_client = FreshdeskClient(domain=freshdesk_domain, api_key=freshdesk_api_key)

    console.print("[bold]ULI Ticket Analyzer[/bold]")
    if args.sample:
        console.print(f"[yellow]Sample mode: processing first {args.sample} tickets only.[/yellow]")
    console.print(f"Fetching tickets from [cyan]{freshdesk_domain}.freshdesk.com[/cyan]...\n")

    df = load_tickets(freshdesk_client, anthropic_client, sample=args.sample)

    console.print(f"\n[green]Loaded {len(df)} tickets.[/green]\n")

    df = categorize_all(df, anthropic_client)
    console.print(f"[green]Categorization complete.[/green]\n")

    # Summary table
    table = Table(title="Ticket Summary (sample — masked)", show_lines=True)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Subject", max_width=40)
    table.add_column("Assigned Category")
    table.add_column("Inferred Category")
    table.add_column("Status")
    table.add_column("Description (snippet)", max_width=50)

    for _, row in df.head(10).iterrows():
        assigned = str(row["category"])
        inferred = str(row["inferred_category"])
        inferred_display = f"[red]{inferred}[/red]" if inferred != assigned else inferred
        desc_snippet = str(row["description"])[:120].replace("\n", " ")
        table.add_row(
            str(row["id"]),
            str(row["subject"]),
            assigned,
            inferred_display,
            str(row["status"]),
            desc_snippet,
        )

    console.print(table)

    if len(df) > 10:
        console.print(f"[dim]... and {len(df) - 10} more tickets.[/dim]")

    # Category breakdown
    if "category" in df.columns and not df["category"].isna().all():
        console.print("\n[bold]Category breakdown:[/bold]")
        breakdown = df["category"].value_counts()
        for cat, count in breakdown.items():
            console.print(f"  {cat or '(unset)'}: {count}")

    # Mismatch report
    mismatches = find_mismatches(df)
    rate = mismatch_rate(df)
    console.print(f"\n[bold]Mismatch report:[/bold] {len(mismatches)} / {len(df)} tickets miscategorised ({rate:.0%})")

    if not mismatches.empty:
        mismatch_table = Table(title="Miscategorised Tickets", show_lines=True)
        mismatch_table.add_column("ID", style="dim", no_wrap=True)
        mismatch_table.add_column("Subject", max_width=40)
        mismatch_table.add_column("Assigned", style="red")
        mismatch_table.add_column("Inferred", style="green")
        mismatch_table.add_column("Agent")

        for _, row in mismatches.iterrows():
            mismatch_table.add_row(
                str(row["id"]),
                str(row["subject"]),
                str(row["assigned_category"]),
                str(row["inferred_category"]),
                str(row["agent"]),
            )

        console.print(mismatch_table)

    # Agent audit
    agent_summary = summarise_by_agent(mismatches)
    if not agent_summary.empty:
        console.print("\n[bold]Agent audit — mismatch counts:[/bold]")
        audit_table = Table(show_lines=True)
        audit_table.add_column("Agent")
        audit_table.add_column("Mismatches", justify="right")
        for _, row in agent_summary.iterrows():
            audit_table.add_row(str(row["agent"]), str(row["mismatch_count"]))
        console.print(audit_table)

        patterns = error_patterns(mismatches)
        console.print("\n[bold]Agent audit — error patterns:[/bold]")
        pattern_table = Table(show_lines=True)
        pattern_table.add_column("Agent")
        pattern_table.add_column("Assigned", style="red")
        pattern_table.add_column("Should be", style="green")
        pattern_table.add_column("Count", justify="right")
        for _, row in patterns.iterrows():
            pattern_table.add_row(
                str(row["agent"]),
                str(row["assigned_category"]),
                str(row["inferred_category"]),
                str(row["count"]),
            )
        console.print(pattern_table)

    # Clustering
    console.print("\n[bold]Clustering tickets by recurring issue...[/bold]")
    df = cluster_tickets(df, n_clusters=args.clusters)
    summaries = cluster_summaries(df)
    console.print(f"[green]Found {len(summaries)} clusters.[/green]\n")

    cluster_table = Table(title="Ticket Clusters", show_lines=True)
    cluster_table.add_column("Cluster", justify="right", style="dim")
    cluster_table.add_column("Size", justify="right")
    cluster_table.add_column("Top Terms", max_width=60)
    for _, row in summaries.iterrows():
        cluster_table.add_row(
            str(row["cluster"]),
            str(row["size"]),
            str(row["top_terms"]),
        )
    console.print(cluster_table)

    # KB generation
    console.print("\n[bold]Generating KB articles...[/bold]")
    articles = generate_kb_articles(df, summaries, anthropic_client)
    console.print(f"[green]Generated {len(articles) * 2} articles ({len(articles)} internal + {len(articles)} lender-facing).[/green]\n")

    kb_table = Table(title="KB Articles Written", show_lines=True)
    kb_table.add_column("Cluster", justify="right", style="dim")
    kb_table.add_column("Size", justify="right")
    kb_table.add_column("Internal article")
    kb_table.add_column("Lender article")
    for a in articles:
        kb_table.add_row(
            str(a["cluster"]),
            str(a["size"]),
            a["internal_path"],
            a["lender_path"],
        )
    console.print(kb_table)


if __name__ == "__main__":
    main()
