"""
Freshdesk API client for fetching support tickets.
"""

import requests
from requests.auth import HTTPBasicAuth


class FreshdeskClient:
    def __init__(self, domain: str, api_key: str):
        """
        Args:
            domain: Freshdesk subdomain (e.g. "yourcompany" → yourcompany.freshdesk.com)
            api_key: Freshdesk API key
        """
        self.base_url = f"https://{domain}.freshdesk.com/api/v2"
        self.auth = HTTPBasicAuth(api_key, "X")

    def fetch_all_tickets(self) -> list[dict]:
        """
        Paginates through all tickets, including description and conversations.

        Returns:
            List of raw ticket dicts from the Freshdesk API.
        """
        tickets = []
        page = 1

        while True:
            response = requests.get(
                f"{self.base_url}/tickets",
                auth=self.auth,
                params={
                    "include": "description,conversations",
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

        return tickets
