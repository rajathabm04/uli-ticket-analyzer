"""
Tests for src/pii_masker.py

spaCy's en_core_web_lg model is mocked at the sys.modules level so these tests
run without requiring `python -m spacy download en_core_web_lg`.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock spaCy before src.pii_masker is imported so the module-level
# `_nlp = spacy.load("en_core_web_lg")` call uses our stub.
# ---------------------------------------------------------------------------
_mock_spacy = MagicMock()
_mock_nlp = MagicMock()
_mock_spacy.load.return_value = _mock_nlp
sys.modules["spacy"] = _mock_spacy

from src.pii_masker import _ner_mask, _regex_mask, mask_all_tickets, mask_ticket  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(*entities):
    """
    Build a mock spaCy Doc with the given entity spans.

    Each entity is a (start_char, end_char, label_) tuple.
    Entities should be provided in ascending char-offset order, as spaCy
    would return them; _ner_mask reverses them internally.
    """
    mock_ents = []
    for start, end, label in entities:
        ent = MagicMock()
        ent.start_char = start
        ent.end_char = end
        ent.label_ = label
        mock_ents.append(ent)
    doc = MagicMock()
    doc.ents = mock_ents
    return doc


def _empty_doc():
    """Return a mock spaCy Doc with no entities."""
    doc = MagicMock()
    doc.ents = []
    return doc


# ---------------------------------------------------------------------------
# _regex_mask
# ---------------------------------------------------------------------------

class TestRegexMask:

    # --- email ---

    def test_simple_email(self):
        assert _regex_mask("contact john.doe@example.com for help") == "contact [EMAIL] for help"

    def test_email_with_subdomains(self):
        assert _regex_mask("ops@mail.rbih.org.in") == "[EMAIL]"

    def test_email_with_plus(self):
        assert _regex_mask("user+tag@company.co") == "[EMAIL]"

    # --- phone ---

    def test_phone_with_plus91_space(self):
        assert _regex_mask("+91 9876543210") == "[PHONE]"

    def test_phone_with_plus91_hyphen(self):
        assert _regex_mask("+91-9876543210") == "[PHONE]"

    def test_phone_with_plus91_no_separator(self):
        assert _regex_mask("+919876543210") == "[PHONE]"

    def test_phone_standalone_10_digit(self):
        assert _regex_mask("call me at 9876543210 please") == "call me at [PHONE] please"

    def test_short_number_not_phone(self):
        # 8 digits — below the 10-digit threshold
        result = _regex_mask("code 12345678")
        assert "[PHONE]" not in result

    # --- PAN ---

    def test_pan(self):
        assert _regex_mask("PAN ABCDE1234F submitted") == "PAN [PAN] submitted"

    def test_pan_lowercase_not_matched(self):
        # PAN must be uppercase
        result = _regex_mask("abcde1234f")
        assert "[PAN]" not in result

    # --- Aadhaar ---

    def test_aadhaar_spaced(self):
        assert _regex_mask("uid: 1234 5678 9012") == "uid: [AADHAAR]"

    def test_aadhaar_hyphenated(self):
        assert _regex_mask("1234-5678-9012") == "[AADHAAR]"

    def test_aadhaar_continuous(self):
        assert _regex_mask("uid=123456789012") == "uid=[AADHAAR]"

    # --- IFSC ---

    def test_ifsc(self):
        assert _regex_mask("IFSC: SBIN0001234") == "IFSC: [IFSC]"

    def test_ifsc_hdfc(self):
        assert _regex_mask("branch HDFC0123456") == "branch [IFSC]"

    # --- Bank account ---

    def test_bank_account_9_digits(self):
        assert _regex_mask("account 123456789 linked") == "account [BANK_ACCOUNT] linked"

    def test_bank_account_18_digits(self):
        # Use digits 0-5 only so the phone regex ([6-9]\d{9}\b) never fires
        # on any 10-digit suffix at a word boundary.
        assert _regex_mask("acc=100000000000000001") == "acc=[BANK_ACCOUNT]"

    def test_8_digits_not_bank_account(self):
        result = _regex_mask("code 12345678")
        assert "[BANK_ACCOUNT]" not in result

    # --- IP address ---

    def test_ipv4(self):
        assert _regex_mask("request from 192.168.1.100 blocked") == "request from [IP_ADDRESS] blocked"

    def test_localhost_ip(self):
        assert _regex_mask("127.0.0.1") == "[IP_ADDRESS]"

    # --- edge cases ---

    def test_empty_string(self):
        assert _regex_mask("") == ""

    def test_none_returns_none(self):
        assert _regex_mask(None) is None

    def test_no_pii_unchanged(self):
        text = "The API returned HTTP 200 OK with status resolved"
        assert _regex_mask(text) == text

    def test_multiple_patterns_in_one_text(self):
        text = "User ABCDE1234F called from 9876543210, IP 10.0.0.1"
        result = _regex_mask(text)
        assert "[PAN]" in result
        assert "[PHONE]" in result
        assert "[IP_ADDRESS]" in result
        assert "ABCDE1234F" not in result
        assert "9876543210" not in result
        assert "10.0.0.1" not in result

    def test_email_and_phone_together(self):
        text = "Contact john@acme.com or call 9123456789"
        result = _regex_mask(text)
        assert "[EMAIL]" in result
        assert "[PHONE]" in result

    def test_already_tokenised_text_unchanged(self):
        # Text that's already been masked should not be double-masked
        text = "User [EMAIL] reported issue from [IP_ADDRESS]"
        assert _regex_mask(text) == text


# ---------------------------------------------------------------------------
# _ner_mask
# ---------------------------------------------------------------------------

class TestNerMask:

    def test_person_replaced(self):
        text = "John Smith reported the issue"
        with patch("src.pii_masker._nlp", return_value=_make_doc((0, 10, "PERSON"))):
            assert _ner_mask(text) == "[PERSON_NAME] reported the issue"

    def test_org_replaced(self):
        text = "Ticket raised by Acme Bank"
        with patch("src.pii_masker._nlp", return_value=_make_doc((17, 26, "ORG"))):
            assert _ner_mask(text) == "Ticket raised by [ORG_NAME]"

    def test_person_and_org_both_replaced(self):
        text = "John Smith from Acme Corp called"
        # Offsets: "John Smith"=[0,10], "Acme Corp"=[16,25]
        with patch("src.pii_masker._nlp", return_value=_make_doc((0, 10, "PERSON"), (16, 25, "ORG"))):
            result = _ner_mask(text)
        assert "[PERSON_NAME]" in result
        assert "[ORG_NAME]" in result
        assert "John Smith" not in result
        assert "Acme Corp" not in result

    def test_reverse_order_preserves_offsets(self):
        """
        Processing in reverse char order means the PERSON replacement at offset 0
        is not invalidated by the earlier ORG replacement at a higher offset.
        """
        text = "Priya Sharma joined HDFC Bank"
        # "Priya Sharma"=[0,12], "HDFC Bank"=[20,29]
        with patch("src.pii_masker._nlp", return_value=_make_doc((0, 12, "PERSON"), (20, 29, "ORG"))):
            result = _ner_mask(text)
        assert result == "[PERSON_NAME] joined [ORG_NAME]"

    def test_non_pii_entity_label_not_replaced(self):
        # GPE (geo-political entity) is not in the label map
        text = "The issue originated from Mumbai"
        with patch("src.pii_masker._nlp", return_value=_make_doc((26, 32, "GPE"))):
            assert _ner_mask(text) == text

    def test_date_entity_not_replaced(self):
        text = "Created on Monday"
        with patch("src.pii_masker._nlp", return_value=_make_doc((11, 17, "DATE"))):
            assert _ner_mask(text) == text

    def test_no_entities_text_unchanged(self):
        text = "API timeout error on endpoint /loan/apply"
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            assert _ner_mask(text) == text

    def test_empty_string_returns_unchanged_without_calling_nlp(self):
        with patch("src.pii_masker._nlp") as mock_nlp:
            result = _ner_mask("")
        assert result == ""
        mock_nlp.assert_not_called()

    def test_whitespace_only_returns_unchanged_without_calling_nlp(self):
        with patch("src.pii_masker._nlp") as mock_nlp:
            result = _ner_mask("   ")
        assert result == "   "
        mock_nlp.assert_not_called()

    def test_multiple_persons_replaced(self):
        text = "Alice and Bob both reported this"
        # "Alice"=[0,5], "Bob"=[10,13]
        with patch("src.pii_masker._nlp", return_value=_make_doc((0, 5, "PERSON"), (10, 13, "PERSON"))):
            result = _ner_mask(text)
        assert "Alice" not in result
        assert "Bob" not in result
        assert result.count("[PERSON_NAME]") == 2


# ---------------------------------------------------------------------------
# mask_ticket
# ---------------------------------------------------------------------------

class TestMaskTicket:

    # --- regex pass (via mask_ticket) ---

    def test_subject_email_masked(self):
        ticket = {"id": 1, "subject": "john.doe@example.com login issue"}
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert "[EMAIL]" in result["subject"]
        assert "john.doe@example.com" not in result["subject"]

    def test_description_pan_masked(self):
        ticket = {"id": 2, "description": "My PAN is ABCDE1234F"}
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert "[PAN]" in result["description"]
        assert "ABCDE1234F" not in result["description"]

    def test_description_text_ip_masked(self):
        ticket = {"id": 3, "description_text": "Request came from 10.0.0.5"}
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert "[IP_ADDRESS]" in result["description_text"]

    def test_conversation_body_text_masked(self):
        ticket = {
            "id": 4,
            "conversations": [{"body_text": "call me at 9876543210", "body": ""}],
        }
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert "[PHONE]" in result["conversations"][0]["body_text"]
        assert "9876543210" not in result["conversations"][0]["body_text"]

    def test_conversation_body_masked(self):
        ticket = {
            "id": 5,
            "conversations": [{"body": "reply to a@b.com"}],
        }
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert "[EMAIL]" in result["conversations"][0]["body"]

    def test_multiple_conversations_all_masked(self):
        ticket = {
            "id": 6,
            "conversations": [
                {"body_text": "user@a.com"},
                {"body_text": "other@b.com"},
            ],
        }
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        for conv in result["conversations"]:
            assert "[EMAIL]" in conv["body_text"]

    # --- NER pass (via mask_ticket) ---

    def test_person_name_in_subject_masked(self):
        ticket = {"id": 7, "subject": "Rahul Sharma reported error"}
        # "Rahul Sharma" = chars 0-12
        doc = _make_doc((0, 12, "PERSON"))
        with patch("src.pii_masker._nlp", return_value=doc):
            result = mask_ticket(ticket)
        assert "[PERSON_NAME]" in result["subject"]
        assert "Rahul Sharma" not in result["subject"]

    def test_org_name_in_description_masked(self):
        ticket = {"id": 8, "description": "Reported by HDFC Bank team"}
        # "HDFC Bank" = chars 12-21
        doc = _make_doc((12, 21, "ORG"))
        with patch("src.pii_masker._nlp", return_value=doc):
            result = mask_ticket(ticket)
        assert "[ORG_NAME]" in result["description"]
        assert "HDFC Bank" not in result["description"]

    # --- preserved fields ---

    def test_agent_name_not_masked(self):
        ticket = {"id": 9, "agent": "Rahul Sharma", "subject": ""}
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert result["agent"] == "Rahul Sharma"

    def test_id_not_masked(self):
        ticket = {"id": 42, "subject": ""}
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert result["id"] == 42

    def test_status_category_created_at_not_masked(self):
        ticket = {
            "id": 10,
            "status": "resolved",
            "category": "API Error",
            "created_at": "2024-01-15T10:30:00Z",
        }
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert result["status"] == "resolved"
        assert result["category"] == "API Error"
        assert result["created_at"] == "2024-01-15T10:30:00Z"

    # --- mutation / edge cases ---

    def test_original_dict_not_mutated(self):
        ticket = {"id": 11, "subject": "test@example.com", "conversations": []}
        original_subject = ticket["subject"]
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            mask_ticket(ticket)
        assert ticket["subject"] == original_subject

    def test_empty_subject_handled(self):
        ticket = {"id": 12, "subject": ""}
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert result["subject"] == ""

    def test_missing_text_fields_handled(self):
        # Ticket with none of the text fields present
        ticket = {"id": 13, "status": "open"}
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert result["id"] == 13

    def test_missing_conversations_handled(self):
        ticket = {"id": 14, "subject": "no convs"}
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert result["id"] == 14

    def test_empty_conversations_list_handled(self):
        ticket = {"id": 15, "conversations": []}
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert result["conversations"] == []

    def test_conversation_with_no_body_fields_handled(self):
        ticket = {"id": 16, "conversations": [{"from_email": "a@b.com"}]}
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        # from_email is not a body field — should be untouched
        assert result["conversations"][0]["from_email"] == "a@b.com"

    def test_returns_new_dict_not_same_object(self):
        ticket = {"id": 17, "subject": "test"}
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_ticket(ticket)
        assert result is not ticket


# ---------------------------------------------------------------------------
# mask_all_tickets
# ---------------------------------------------------------------------------

class TestMaskAllTickets:

    def test_returns_same_count(self):
        tickets = [
            {"id": 1, "subject": "a@b.com"},
            {"id": 2, "subject": "c@d.com"},
            {"id": 3, "subject": "no pii"},
        ]
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_all_tickets(tickets)
        assert len(result) == 3

    def test_each_ticket_is_masked(self):
        tickets = [
            {"id": 1, "subject": "user@example.com"},
            {"id": 2, "subject": "other@example.com"},
        ]
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_all_tickets(tickets)
        for t in result:
            assert "[EMAIL]" in t["subject"]
            assert "@" not in t["subject"]

    def test_empty_list_returns_empty_list(self):
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_all_tickets([])
        assert result == []

    def test_ticket_order_preserved(self):
        tickets = [{"id": i, "subject": ""} for i in range(5)]
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            result = mask_all_tickets(tickets)
        assert [t["id"] for t in result] == [0, 1, 2, 3, 4]

    def test_originals_not_mutated(self):
        tickets = [{"id": 1, "subject": "user@example.com"}]
        with patch("src.pii_masker._nlp", return_value=_empty_doc()):
            mask_all_tickets(tickets)
        assert tickets[0]["subject"] == "user@example.com"
