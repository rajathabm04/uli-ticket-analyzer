"""
Tests for src/categorizer.py

The Anthropic client is mocked so tests run without API calls.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.categorizer import CATEGORIES, categorize_all, categorize_ticket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(response_text: str) -> MagicMock:
    """Return a mock anthropic client whose messages.create returns response_text."""
    content_block = MagicMock()
    content_block.text = response_text
    message = MagicMock()
    message.content = [content_block]
    client = MagicMock()
    client.messages.create.return_value = message
    return client


def _make_df(tickets: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame as loader.load_tickets would produce."""
    rows = []
    for t in tickets:
        rows.append({
            "id": t.get("id", 1),
            "subject": t.get("subject", ""),
            "description": t.get("description", ""),
            "conversations": t.get("conversations", ""),
            "category": t.get("category", ""),
            "agent": t.get("agent", ""),
            "status": t.get("status", "open"),
            "created_at": t.get("created_at", ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CATEGORIES constant
# ---------------------------------------------------------------------------

class TestCategories:

    def test_categories_is_nonempty_list(self):
        assert isinstance(CATEGORIES, list)
        assert len(CATEGORIES) > 0

    def test_other_is_present(self):
        assert "Other" in CATEGORIES

    def test_all_entries_are_nonempty_strings(self):
        for cat in CATEGORIES:
            assert isinstance(cat, str) and cat.strip()

    def test_no_duplicates(self):
        assert len(CATEGORIES) == len(set(CATEGORIES))


# ---------------------------------------------------------------------------
# categorize_ticket
# ---------------------------------------------------------------------------

class TestCategorizeTicket:

    def test_returns_known_category(self):
        client = _make_client("API Error")
        result = categorize_ticket({"subject": "500 error", "description": "keeps failing"}, client)
        assert result == "API Error"

    def test_case_insensitive_match(self):
        client = _make_client("api error")
        result = categorize_ticket({"subject": "x", "description": "y"}, client)
        assert result == "API Error"

    def test_strips_whitespace_from_response(self):
        client = _make_client("  Onboarding  \n")
        result = categorize_ticket({"subject": "x", "description": "y"}, client)
        assert result == "Onboarding"

    def test_unknown_response_falls_back_to_other(self):
        client = _make_client("Something completely unknown")
        result = categorize_ticket({"subject": "x", "description": "y"}, client)
        assert result == "Other"

    def test_empty_response_falls_back_to_other(self):
        client = _make_client("")
        result = categorize_ticket({"subject": "x", "description": "y"}, client)
        assert result == "Other"

    def test_result_is_always_in_categories(self):
        for cat in CATEGORIES:
            client = _make_client(cat)
            result = categorize_ticket({"subject": "x", "description": "y"}, client)
            assert result in CATEGORIES

    def test_subject_included_in_prompt(self):
        client = _make_client("Other")
        categorize_ticket({"subject": "login broken", "description": "desc"}, client)
        call_kwargs = client.messages.create.call_args
        content = call_kwargs[1]["messages"][0]["content"]
        assert "login broken" in content

    def test_description_included_in_prompt(self):
        client = _make_client("Other")
        categorize_ticket({"subject": "s", "description": "JWT token expired"}, client)
        content = client.messages.create.call_args[1]["messages"][0]["content"]
        assert "JWT token expired" in content

    def test_conversations_included_when_present(self):
        client = _make_client("Other")
        categorize_ticket({"subject": "s", "description": "d", "conversations": "conv text"}, client)
        content = client.messages.create.call_args[1]["messages"][0]["content"]
        assert "conv text" in content

    def test_conversations_omitted_when_empty(self):
        client = _make_client("Other")
        categorize_ticket({"subject": "s", "description": "d", "conversations": ""}, client)
        content = client.messages.create.call_args[1]["messages"][0]["content"]
        assert "Conversation:" not in content

    def test_uses_correct_model(self):
        client = _make_client("Other")
        categorize_ticket({"subject": "s", "description": "d"}, client)
        model = client.messages.create.call_args[1]["model"]
        assert model == "claude-sonnet-4-6"

    def test_missing_fields_do_not_raise(self):
        client = _make_client("Other")
        result = categorize_ticket({}, client)
        assert result in CATEGORIES

    def test_calls_claude_exactly_once(self):
        client = _make_client("Integration")
        categorize_ticket({"subject": "s", "description": "d"}, client)
        client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# categorize_all
# ---------------------------------------------------------------------------

class TestCategorizeAll:

    def test_adds_inferred_category_column(self):
        df = _make_df([{"id": 1, "subject": "test"}])
        client = _make_client("API Error")
        result = categorize_all(df, client)
        assert "inferred_category" in result.columns

    def test_original_df_not_mutated(self):
        df = _make_df([{"id": 1}])
        client = _make_client("Other")
        categorize_all(df, client)
        assert "inferred_category" not in df.columns

    def test_row_count_unchanged(self):
        df = _make_df([{"id": i} for i in range(5)])
        client = _make_client("Onboarding")
        result = categorize_all(df, client)
        assert len(result) == 5

    def test_all_inferred_values_in_categories(self):
        df = _make_df([{"id": i} for i in range(3)])
        client = _make_client("Data Mismatch")
        result = categorize_all(df, client)
        for val in result["inferred_category"]:
            assert val in CATEGORIES

    def test_empty_dataframe_returns_empty_with_column(self):
        df = _make_df([])
        client = _make_client("Other")
        result = categorize_all(df, client)
        assert "inferred_category" in result.columns
        assert len(result) == 0

    def test_claude_called_once_per_ticket(self):
        n = 4
        df = _make_df([{"id": i} for i in range(n)])
        client = _make_client("Integration")
        categorize_all(df, client)
        assert client.messages.create.call_count == n

    def test_other_columns_preserved(self):
        df = _make_df([{"id": 99, "subject": "keep me", "status": "resolved"}])
        client = _make_client("Other")
        result = categorize_all(df, client)
        assert result.iloc[0]["subject"] == "keep me"
        assert result.iloc[0]["status"] == "resolved"
        assert result.iloc[0]["id"] == 99
