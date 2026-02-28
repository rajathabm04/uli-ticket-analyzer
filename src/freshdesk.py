"""
Freshdesk API client for fetching support tickets.
"""

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.auth import HTTPBasicAuth
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn


_CONVERSATION_WORKERS = 10   # parallel threads for conversation fetching


class FreshdeskClient:
    def __init__(self, domain: str, api_key: str):
        """
        Args:
            domain: Freshdesk subdomain (e.g. "yourcompany" → yourcompany.freshdesk.com)
            api_key: Freshdesk API key
        """
        self.base_url = f"https://{domain}.freshdesk.com/api/v2"
        self.auth = HTTPBasicAuth(api_key, "X")

    def _fetch_conversations(self, ticket_id: int) -> list[dict]:
        """
        Fetch all conversations for a single ticket.
        Freshdesk conversations are not available via the list-tickets include
        parameter — they require a dedicated endpoint.
        """
        try:
            response = requests.get(
                f"{self.base_url}/tickets/{ticket_id}/conversations",
                auth=self.auth,
                timeout=30,
            )
        except requests.exceptions.ConnectionError:
            return []
        if response.status_code == 200:
            return response.json()
        # 404 means no conversations; any other error surfaces as empty
        return []

    def fetch_all_tickets(self) -> list[dict]:
        """
        Paginates through all tickets with descriptions, then attaches
        conversations in parallel via per-ticket API calls.

        Returns:
            List of raw ticket dicts from the Freshdesk API, each with a
            'conversations' key containing the list of conversation dicts.
        """
        tickets = []
        page = 1

        while True:
            response = requests.get(
                f"{self.base_url}/tickets",
                auth=self.auth,
                params={
                    "include": "description",
                    "per_page": 100,
                    "page": page,
                },
            )
            response.raise_for_status()
            batch = response.json()

            if not batch:
                break

            tickets.extend(batch)

            if len(batch) < 100:
                break

            page += 1

        # Fetch conversations in parallel
        conversations: dict[int, list] = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Fetching conversations..."),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total} tickets"),
        ) as progress:
            task = progress.add_task("conversations", total=len(tickets))
            with ThreadPoolExecutor(max_workers=_CONVERSATION_WORKERS) as executor:
                future_to_id = {
                    executor.submit(self._fetch_conversations, t["id"]): t["id"]
                    for t in tickets
                }
                for future in as_completed(future_to_id):
                    tid = future_to_id[future]
                    conversations[tid] = future.result()
                    progress.advance(task)

        for ticket in tickets:
            ticket["conversations"] = conversations.get(ticket["id"], [])

        return tickets
