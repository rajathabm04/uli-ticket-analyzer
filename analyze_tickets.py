#!/usr/bin/env python3
"""
ULI Ticket Analyzer — entry point.
Usage: python analyze_tickets.py
"""

import os
import sys

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.freshdesk import FreshdeskClient
from src.loader import load_tickets

load_dotenv()

console = Console()


def _get_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        console.print(f"[red]Error:[/red] {name} is not set. Add it to your .env file.")
        sys.exit(1)
    return val


def main() -> None:
    anthropic_api_key = _get_env("ANTHROPIC_API_KEY")
    freshdesk_domain = _get_env("FRESHDESK_DOMAIN")
    freshdesk_api_key = _get_env("FRESHDESK_API_KEY")

    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
    freshdesk_client = FreshdeskClient(domain=freshdesk_domain, api_key=freshdesk_api_key)

    console.print("[bold]ULI Ticket Analyzer[/bold]")
    console.print(f"Fetching tickets from [cyan]{freshdesk_domain}.freshdesk.com[/cyan]...\n")

    df = load_tickets(freshdesk_client, anthropic_client)

    console.print(f"\n[green]Loaded {len(df)} tickets.[/green]\n")

    # Summary table
    table = Table(title="Ticket Summary (sample — masked)", show_lines=True)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Subject", max_width=40)
    table.add_column("Category")
    table.add_column("Status")
    table.add_column("Description (snippet)", max_width=50)

    for _, row in df.head(10).iterrows():
        desc_snippet = str(row["description"])[:120].replace("\n", " ")
        table.add_row(
            str(row["id"]),
            str(row["subject"]),
            str(row["category"]),
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


if __name__ == "__main__":
    main()
