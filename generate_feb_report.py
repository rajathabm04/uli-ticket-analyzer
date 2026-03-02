#!/usr/bin/env python3
"""
February 2026 Report Generator — ULI Ticket Analyzer.

Fetches tickets from Freshdesk, filters to February 2026, runs
keyword-based categorization (zero Claude API calls), produces:
  - output/feb_deck_insights.md   — slide-ready monthly recap
  - output/freshdesk_export/      — KB articles in HTML + manifest.json

Usage:
    python generate_feb_report.py
    python generate_feb_report.py --since 2026-02-01 --until 2026-03-01
"""

import argparse
import os
import sys

import anthropic
from dotenv import load_dotenv
from rich.console import Console

from src.freshdesk import FreshdeskClient
from src.loader import load_tickets
from src.categorizer import categorize_all
from src.mismatch import find_mismatches
from src.agent_audit import summarise_by_agent
from src.clusterer import cluster_tickets, cluster_summaries
from src.feb_analyzer import generate_deck_insights
from src.kb_exporter import export_for_freshdesk

load_dotenv()

console = Console()


def _get_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        console.print(f"[red]Error:[/red] {name} is not set. Add it to your .env file.")
        sys.exit(1)
    return val


def main() -> None:
    parser = argparse.ArgumentParser(description="ULI February Recap Report Generator")
    parser.add_argument(
        "--since",
        default="2026-02-01",
        metavar="YYYY-MM-DD",
        help="Start date (inclusive). Default: 2026-02-01",
    )
    parser.add_argument(
        "--until",
        default="2026-03-01",
        metavar="YYYY-MM-DD",
        help="End date (exclusive). Default: 2026-03-01",
    )
    args = parser.parse_args()

    anthropic_api_key = _get_env("ANTHROPIC_API_KEY")
    freshdesk_domain  = _get_env("FRESHDESK_DOMAIN")
    freshdesk_api_key = _get_env("FRESHDESK_API_KEY")

    # anthropic_client is instantiated for API compatibility but never invoked
    # (keyword categorizer is API-free; KB export is file I/O only)
    anthropic_client  = anthropic.Anthropic(api_key=anthropic_api_key)
    freshdesk_client  = FreshdeskClient(domain=freshdesk_domain, api_key=freshdesk_api_key)

    console.print("[bold]ULI Ticket Analyzer — February Recap[/bold]")
    console.print(f"Date range: [cyan]{args.since}[/cyan] → [cyan]{args.until}[/cyan]")
    console.print(f"Fetching tickets from [cyan]{freshdesk_domain}.freshdesk.com[/cyan]...\n")

    # 1. Load + date-filter
    df = load_tickets(
        freshdesk_client,
        anthropic_client,
        since=args.since,
        until=args.until,
    )
    console.print(f"[green]Loaded {len(df)} tickets ({args.since} → {args.until})[/green]\n")

    if df.empty:
        console.print("[yellow]No tickets found in the given date range. Exiting.[/yellow]")
        sys.exit(0)

    # 2. Keyword-based categorization (zero API calls)
    df = categorize_all(df, anthropic_client)

    # 3. Mismatch detection + agent audit
    mismatches    = find_mismatches(df)
    agent_summary = summarise_by_agent(mismatches)

    # 4. Clustering
    df       = cluster_tickets(df)
    summaries = cluster_summaries(df)
    console.print(f"[green]Clusters: {len(summaries)}[/green]")

    # 5. Write deck insights
    console.print("\n[bold]Writing feb_deck_insights.md...[/bold]")
    generate_deck_insights(
        df,
        summaries,
        mismatches,
        agent_summary,
        since=args.since,
        until=args.until,
        output_path="output/feb_deck_insights.md",
    )
    console.print("[green]output/feb_deck_insights.md written.[/green]")

    # 6. Export existing KB articles to Freshdesk HTML + manifest
    console.print("\n[bold]Exporting KB articles for Freshdesk...[/bold]")
    manifest = export_for_freshdesk(
        internal_dir="output/internal",
        lender_dir="output/lender-facing",
        output_dir="output/freshdesk_export",
    )
    console.print(f"[green]output/freshdesk_export/ populated ({len(manifest)} articles).[/green]")

    console.print("\n[bold green]Done.[/bold green]")
    console.print("  Deck insights : output/feb_deck_insights.md")
    console.print("  Freshdesk KB  : output/freshdesk_export/")


if __name__ == "__main__":
    main()
