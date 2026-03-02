"""
Loads tickets from the Freshdesk API, masks PII, and returns a normalised DataFrame.
"""

import anthropic
import pandas as pd
from typing import Optional

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

# Freshdesk numeric status codes → human-readable labels
_STATUS_MAP: dict[int, str] = {
    2: "open",
    3: "pending",
    4: "resolved",
    5: "closed",
    6: "waiting on customer",
    7: "waiting on third party",
}


def _extract_conversations(ticket: dict) -> str:
    """Flatten conversation thread into a single string."""
    conversations = ticket.get("conversations", [])
    parts = []
    for conv in conversations:
        body = conv.get("body_text") or conv.get("body") or ""
        if body:
            parts.append(body.strip())
    return "\n---\n".join(parts)


def _normalise(tickets: list[dict], agent_map: Optional[dict[int, str]] = None) -> pd.DataFrame:
    """Convert list of masked ticket dicts to a normalised DataFrame."""
    agent_map = agent_map or {}
    rows = []
    for t in tickets:
        raw_agent = t.get("responder_id") or t.get("agent_name") or t.get("agent", "")
        agent = agent_map.get(raw_agent, raw_agent) if isinstance(raw_agent, int) else raw_agent
        row = {
            "id": t.get("id"),
            "subject": t.get("subject", ""),
            "description": t.get("description_text") or t.get("description", ""),
            "conversations": _extract_conversations(t),
            "category": t.get("type") or t.get("category", ""),
            "agent": agent,
            "status": _STATUS_MAP.get(t.get("status"), t.get("status", "")),
            "created_at": t.get("created_at", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all required columns exist (fill with empty string if absent)
    for col in _REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[["id", "subject", "description", "conversations", "category", "agent", "status", "created_at"]]


def load_tickets(
    freshdesk: FreshdeskClient,
    anthropic_client: anthropic.Anthropic,
    sample: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch all tickets from Freshdesk, mask PII, and return a normalised DataFrame.

    Args:
        freshdesk: Initialised FreshdeskClient.
        anthropic_client: Initialised anthropic.Anthropic client (used by downstream
            categorizer/kb_generator — not forwarded to the masker).
        sample: If set, fetch and process only this many tickets.
        since: ISO date string (YYYY-MM-DD). Keep tickets created on or after this date.
        until: ISO date string (YYYY-MM-DD). Keep tickets created before this date.

    Returns:
        DataFrame with columns: id, subject, description, conversations,
        category, agent, status, created_at.
    """
    agent_map = freshdesk.fetch_agents()
    raw_tickets = freshdesk.fetch_all_tickets(max_tickets=sample)
    masked_tickets = mask_all_tickets(raw_tickets)
    result = _normalise(masked_tickets, agent_map=agent_map)

    if since or until:
        dates = pd.to_datetime(result["created_at"], utc=True, errors="coerce")
        if since:
            result = result[dates >= pd.Timestamp(since, tz="UTC")]
        if until:
            result = result[dates < pd.Timestamp(until, tz="UTC")]
        result = result.reset_index(drop=True)

    return result
