"""
Pytest configuration for the uli-ticket-analyzer test suite.

Sets up the spaCy sys.modules mock before any test file imports src.pii_masker,
so tests run without requiring `python -m spacy download en_core_web_lg`.
"""

import sys
from unittest.mock import MagicMock

_mock_spacy = MagicMock()
_mock_nlp = MagicMock()
_mock_spacy.load.return_value = _mock_nlp
sys.modules["spacy"] = _mock_spacy
