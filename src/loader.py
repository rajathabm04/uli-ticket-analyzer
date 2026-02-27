"""
Loads tickets from the Freshdesk API, masks PII, and returns a normalised DataFrame.
"""

import anthropic
import pandas as pd

from src.freshdesk import FreshdeskClient
from src.pii_masker import mask_all_tickets  # no longer needs anthropic_client


# Mapping of possible Freshdesk field names → canonical column names
_COLUMN_MAP = {
    # ticket id
    "id": "id",
    "ticket_id": "id",
    # subject
    "subject": "subject",
    # description
    "description_text": "description",
    "description": "description",
    # category
    "category": "category",
    "type": "category",
    # agent / responder
    "responder_id": "agent",
    "agent_name": "agent",
    "agent": "agent",
    # status
    "status": "status",
    # created at
    "created_at": "created_at",
}

_REQUIRED_COLUMNS = ["id", "subject", "description", "conversations", "category", "agent", "status", "created_at"]


def _extract_conversations(ticket: dict) -> str:
    """Flatten conversation thread into a single string."""
    conversations = ticket.get("conversations", [])
    parts = []
    for conv in conversations:
        body = conv.get("body_text") or conv.get("body") or ""
        if body:
            parts.append(body.strip())
    return "\n---\n".join(parts)


def _normalise(tickets: list[dict]) -> pd.DataFrame:
    """Convert list of masked ticket dicts to a normalised DataFrame."""
    rows = []
    for t in tickets:
        row = {
            "id": t.get("id"),
            "subject": t.get("subject", ""),
            "description": t.get("description_text") or t.get("description", ""),
            "conversations": _extract_conversations(t),
            "category": t.get("type") or t.get("category", ""),
            "agent": t.get("responder_id") or t.get("agent_name") or t.get("agent", ""),
            "status": t.get("status", ""),
            "created_at": t.get("created_at", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all required columns exist (fill with empty string if absent)
    for col in _REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[["id", "subject", "description", "conversations", "category", "agent", "status", "created_at"]]


def load_tickets(freshdesk: FreshdeskClient, anthropic_client: anthropic.Anthropic) -> pd.DataFrame:
    """
    Fetch all tickets from Freshdesk, mask PII, and return a normalised DataFrame.

    Args:
        freshdesk: Initialised FreshdeskClient.
        anthropic_client: Initialised anthropic.Anthropic client (used by downstream
            categorizer/kb_generator — not forwarded to the masker).

    Returns:
        DataFrame with columns: id, subject, description, conversations,
        category, agent, status, created_at.
    """
    raw_tickets = freshdesk.fetch_all_tickets()
    masked_tickets = mask_all_tickets(raw_tickets)
    return _normalise(masked_tickets)
