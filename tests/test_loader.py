"""
Tests for src/loader.py

External dependencies (Freshdesk API, PII masker) are mocked so these tests
run without network access or a spaCy model.
"""

from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from src.loader import _extract_conversations, _normalise, load_tickets


# ---------------------------------------------------------------------------
# _extract_conversations
# ---------------------------------------------------------------------------

class TestExtractConversations:

    def test_empty_conversations_list(self):
        assert _extract_conversations({"conversations": []}) == ""

    def test_missing_conversations_key(self):
        assert _extract_conversations({}) == ""

    def test_single_body_text(self):
        ticket = {"conversations": [{"body_text": "API is down"}]}
        assert _extract_conversations(ticket) == "API is down"

    def test_body_fallback_when_no_body_text(self):
        ticket = {"conversations": [{"body": "fallback body"}]}
        assert _extract_conversations(ticket) == "fallback body"

    def test_body_text_preferred_over_body(self):
        ticket = {"conversations": [{"body_text": "primary", "body": "secondary"}]}
        assert _extract_conversations(ticket) == "primary"

    def test_multiple_conversations_joined_with_separator(self):
        ticket = {
            "conversations": [
                {"body_text": "First message"},
                {"body_text": "Second message"},
                {"body_text": "Third message"},
            ]
        }
        result = _extract_conversations(ticket)
        assert result == "First message\n---\nSecond message\n---\nThird message"

    def test_empty_body_skipped(self):
        ticket = {
            "conversations": [
                {"body_text": ""},
                {"body_text": "real content"},
            ]
        }
        assert _extract_conversations(ticket) == "real content"

    def test_conversation_with_neither_field_skipped(self):
        ticket = {"conversations": [{"from_email": "a@b.com"}]}
        assert _extract_conversations(ticket) == ""

    def test_all_empty_bodies_returns_empty(self):
        ticket = {"conversations": [{"body_text": ""}, {"body": ""}]}
        assert _extract_conversations(ticket) == ""

    def test_whitespace_stripped_from_body(self):
        ticket = {"conversations": [{"body_text": "  trimmed  "}]}
        assert _extract_conversations(ticket) == "trimmed"

    def test_mixed_body_text_and_body_fields(self):
        ticket = {
            "conversations": [
                {"body_text": "via body_text"},
                {"body": "via body"},
            ]
        }
        result = _extract_conversations(ticket)
        assert result == "via body_text\n---\nvia body"


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------

class TestNormalise:

    def test_returns_dataframe(self):
        assert isinstance(_normalise([]), pd.DataFrame)

    def test_column_names_and_order(self):
        df = _normalise([])
        assert list(df.columns) == [
            "id", "subject", "description", "conversations",
            "category", "agent", "status", "created_at",
        ]

    def test_empty_list_preserves_columns(self):
        df = _normalise([])
        assert len(df) == 0
        assert set(df.columns) == {
            "id", "subject", "description", "conversations",
            "category", "agent", "status", "created_at",
        }

    def test_single_ticket_becomes_one_row(self):
        df = _normalise([{"id": 1, "subject": "test"}])
        assert len(df) == 1

    def test_multiple_tickets_correct_row_count(self):
        tickets = [{"id": i} for i in range(5)]
        assert len(_normalise(tickets)) == 5

    # --- field mapping ---

    def test_id_preserved(self):
        df = _normalise([{"id": 42}])
        assert df.iloc[0]["id"] == 42

    def test_subject_preserved(self):
        df = _normalise([{"subject": "Login failure"}])
        assert df.iloc[0]["subject"] == "Login failure"

    def test_description_text_used_over_description(self):
        ticket = {"description_text": "from description_text", "description": "from description"}
        df = _normalise([ticket])
        assert df.iloc[0]["description"] == "from description_text"

    def test_description_fallback_when_no_description_text(self):
        df = _normalise([{"description": "fallback description"}])
        assert df.iloc[0]["description"] == "fallback description"

    def test_type_used_over_category(self):
        ticket = {"type": "API Error", "category": "Other"}
        df = _normalise([ticket])
        assert df.iloc[0]["category"] == "API Error"

    def test_category_fallback_when_no_type(self):
        df = _normalise([{"category": "Onboarding"}])
        assert df.iloc[0]["category"] == "Onboarding"

    def test_responder_id_used_as_agent(self):
        ticket = {"responder_id": 101, "agent_name": "Alice", "agent": "Bob"}
        df = _normalise([ticket])
        assert df.iloc[0]["agent"] == 101

    def test_agent_name_fallback_when_no_responder_id(self):
        ticket = {"agent_name": "Alice", "agent": "Bob"}
        df = _normalise([ticket])
        assert df.iloc[0]["agent"] == "Alice"

    def test_agent_last_fallback(self):
        df = _normalise([{"agent": "Bob"}])
        assert df.iloc[0]["agent"] == "Bob"

    def test_status_preserved(self):
        df = _normalise([{"status": "resolved"}])
        assert df.iloc[0]["status"] == "resolved"

    def test_created_at_preserved(self):
        df = _normalise([{"created_at": "2024-01-15T10:30:00Z"}])
        assert df.iloc[0]["created_at"] == "2024-01-15T10:30:00Z"

    # --- missing / empty fields ---

    def test_missing_fields_default_to_empty_string(self):
        df = _normalise([{"id": 1}])
        row = df.iloc[0]
        assert row["subject"] == ""
        assert row["description"] == ""
        assert row["category"] == ""
        assert row["agent"] == ""
        assert row["status"] == ""
        assert row["created_at"] == ""

    def test_conversations_flattened(self):
        ticket = {
            "conversations": [
                {"body_text": "message one"},
                {"body_text": "message two"},
            ]
        }
        df = _normalise([ticket])
        assert df.iloc[0]["conversations"] == "message one\n---\nmessage two"

    def test_no_conversations_gives_empty_string(self):
        df = _normalise([{"id": 1}])
        assert df.iloc[0]["conversations"] == ""

    # --- multi-row integrity ---

    def test_row_order_matches_input_order(self):
        tickets = [{"id": i, "subject": f"ticket {i}"} for i in range(3)]
        df = _normalise(tickets)
        assert list(df["id"]) == [0, 1, 2]

    def test_each_row_has_independent_data(self):
        tickets = [
            {"id": 1, "subject": "first"},
            {"id": 2, "subject": "second"},
        ]
        df = _normalise(tickets)
        assert df.iloc[0]["subject"] == "first"
        assert df.iloc[1]["subject"] == "second"


# ---------------------------------------------------------------------------
# load_tickets
# ---------------------------------------------------------------------------

class TestLoadTickets:

    def _make_freshdesk(self, raw_tickets):
        fd = MagicMock()
        fd.fetch_all_tickets.return_value = raw_tickets
        return fd

    @patch("src.loader.mask_all_tickets")
    def test_fetches_tickets_from_freshdesk(self, mock_mask):
        mock_mask.return_value = []
        freshdesk = self._make_freshdesk([])
        load_tickets(freshdesk, MagicMock())
        freshdesk.fetch_all_tickets.assert_called_once()

    @patch("src.loader.mask_all_tickets")
    def test_passes_raw_tickets_to_masker(self, mock_mask):
        raw = [{"id": 1, "subject": "raw subject"}]
        mock_mask.return_value = raw
        freshdesk = self._make_freshdesk(raw)
        load_tickets(freshdesk, MagicMock())
        mock_mask.assert_called_once_with(raw)

    @patch("src.loader.mask_all_tickets")
    def test_anthropic_client_not_forwarded_to_masker(self, mock_mask):
        """anthropic_client is accepted by load_tickets but must not reach mask_all_tickets."""
        mock_mask.return_value = []
        freshdesk = self._make_freshdesk([])
        anthropic_client = MagicMock()
        load_tickets(freshdesk, anthropic_client)
        # mask_all_tickets must have been called without the anthropic client
        args, kwargs = mock_mask.call_args
        assert anthropic_client not in args
        assert anthropic_client not in kwargs.values()

    @patch("src.loader.mask_all_tickets")
    def test_returns_dataframe(self, mock_mask):
        mock_mask.return_value = [{"id": 1, "subject": "test"}]
        freshdesk = self._make_freshdesk([{"id": 1}])
        result = load_tickets(freshdesk, MagicMock())
        assert isinstance(result, pd.DataFrame)

    @patch("src.loader.mask_all_tickets")
    def test_returned_dataframe_has_correct_columns(self, mock_mask):
        mock_mask.return_value = [{"id": 1, "subject": "test"}]
        freshdesk = self._make_freshdesk([{"id": 1}])
        df = load_tickets(freshdesk, MagicMock())
        assert list(df.columns) == [
            "id", "subject", "description", "conversations",
            "category", "agent", "status", "created_at",
        ]

    @patch("src.loader.mask_all_tickets")
    def test_masked_tickets_are_normalised_into_rows(self, mock_mask):
        masked = [
            {"id": 1, "subject": "ticket one"},
            {"id": 2, "subject": "ticket two"},
        ]
        mock_mask.return_value = masked
        freshdesk = self._make_freshdesk([{}, {}])
        df = load_tickets(freshdesk, MagicMock())
        assert len(df) == 2
        assert df.iloc[0]["subject"] == "ticket one"
        assert df.iloc[1]["subject"] == "ticket two"

    @patch("src.loader.mask_all_tickets")
    def test_empty_freshdesk_returns_empty_dataframe(self, mock_mask):
        mock_mask.return_value = []
        freshdesk = self._make_freshdesk([])
        df = load_tickets(freshdesk, MagicMock())
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
