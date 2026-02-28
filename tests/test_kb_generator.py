"""
Tests for src/kb_generator.py

Claude API calls and file I/O are mocked throughout.
"""

import os
from unittest.mock import MagicMock, call, mock_open, patch

import pandas as pd
import pytest

from src.kb_generator import (
    _build_prompt,
    _generate_article,
    _sample_tickets,
    _write_article,
    generate_kb_articles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(text: str = "Generated article content") -> MagicMock:
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    client = MagicMock()
    client.messages.create.return_value = msg
    return client


def _make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "id": 0, "subject": "", "description": "",
        "conversations": "", "cluster": 0,
    }
    if not rows:
        return pd.DataFrame(columns=list(defaults.keys()))
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_summaries(clusters: list[dict]) -> pd.DataFrame:
    defaults = {"cluster": 0, "size": 1, "top_terms": "api error timeout"}
    if not clusters:
        return pd.DataFrame(columns=["cluster", "size", "top_terms"])
    return pd.DataFrame([{**defaults, **c} for c in clusters])


# ---------------------------------------------------------------------------
# _sample_tickets
# ---------------------------------------------------------------------------

class TestSampleTickets:

    def test_returns_tickets_from_correct_cluster(self):
        df = _make_df([
            {"id": 1, "cluster": 0},
            {"id": 2, "cluster": 1},
            {"id": 3, "cluster": 0},
        ])
        result = _sample_tickets(df, cluster_id=0)
        assert set(result["id"]) == {1, 3}

    def test_caps_at_n(self):
        df = _make_df([{"id": i, "cluster": 0} for i in range(10)])
        result = _sample_tickets(df, cluster_id=0, n=3)
        assert len(result) == 3

    def test_empty_cluster_returns_empty(self):
        df = _make_df([{"id": 1, "cluster": 1}])
        result = _sample_tickets(df, cluster_id=0)
        assert len(result) == 0

    def test_returns_all_when_fewer_than_n(self):
        df = _make_df([{"id": i, "cluster": 0} for i in range(2)])
        result = _sample_tickets(df, cluster_id=0, n=5)
        assert len(result) == 2

    def test_preserves_original_order(self):
        df = _make_df([{"id": i, "cluster": 0} for i in range(5)])
        result = _sample_tickets(df, cluster_id=0, n=5)
        assert list(result["id"]) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:

    def test_includes_top_terms(self):
        df = _make_df([{"subject": "test issue"}])
        prompt = _build_prompt("api error timeout", df)
        assert "api error timeout" in prompt

    def test_includes_subject(self):
        df = _make_df([{"subject": "JWT token expired"}])
        prompt = _build_prompt("auth", df)
        assert "JWT token expired" in prompt

    def test_includes_description(self):
        df = _make_df([{"description": "detailed error log here"}])
        prompt = _build_prompt("error", df)
        assert "detailed error log here" in prompt

    def test_includes_conversations_when_present(self):
        df = _make_df([{"conversations": "agent reply here"}])
        prompt = _build_prompt("error", df)
        assert "agent reply here" in prompt

    def test_omits_empty_description(self):
        df = _make_df([{"subject": "test", "description": ""}])
        prompt = _build_prompt("error", df)
        assert "Description:" not in prompt

    def test_omits_empty_conversations(self):
        df = _make_df([{"subject": "test", "conversations": ""}])
        prompt = _build_prompt("error", df)
        assert "Conversation snippet:" not in prompt

    def test_description_truncated_at_500_chars(self):
        long_desc = "x" * 600
        df = _make_df([{"description": long_desc}])
        prompt = _build_prompt("error", df)
        assert "x" * 501 not in prompt
        assert "x" * 500 in prompt

    def test_conversations_truncated_at_300_chars(self):
        long_conv = "y" * 400
        df = _make_df([{"conversations": long_conv}])
        prompt = _build_prompt("error", df)
        assert "y" * 301 not in prompt
        assert "y" * 300 in prompt

    def test_shows_ticket_count(self):
        df = _make_df([{"subject": "t1"}, {"subject": "t2"}, {"subject": "t3"}])
        prompt = _build_prompt("error", df)
        assert "3 examples" in prompt

    def test_multiple_tickets_all_appear(self):
        df = _make_df([{"subject": "first issue"}, {"subject": "second issue"}])
        prompt = _build_prompt("error", df)
        assert "first issue" in prompt
        assert "second issue" in prompt

    def test_returns_string(self):
        df = _make_df([{"subject": "test"}])
        assert isinstance(_build_prompt("terms", df), str)


# ---------------------------------------------------------------------------
# _generate_article
# ---------------------------------------------------------------------------

class TestGenerateArticle:

    def test_returns_claude_response_text(self):
        client = _make_client("## Issue Summary\nAPI is down.")
        result = _generate_article("prompt text", "system text", client)
        assert result == "## Issue Summary\nAPI is down."

    def test_uses_correct_model(self):
        client = _make_client()
        _generate_article("prompt", "system", client)
        kwargs = client.messages.create.call_args[1]
        assert kwargs["model"] == "claude-sonnet-4-6"

    def test_passes_system_prompt(self):
        client = _make_client()
        _generate_article("user prompt", "my system prompt", client)
        kwargs = client.messages.create.call_args[1]
        assert kwargs["system"] == "my system prompt"

    def test_passes_user_prompt_as_message(self):
        client = _make_client()
        _generate_article("user prompt text", "system", client)
        kwargs = client.messages.create.call_args[1]
        assert kwargs["messages"][0]["content"] == "user prompt text"

    def test_calls_claude_exactly_once(self):
        client = _make_client()
        _generate_article("p", "s", client)
        client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# _write_article
# ---------------------------------------------------------------------------

class TestWriteArticle:

    def test_creates_file_with_content(self, tmp_path):
        path = str(tmp_path / "article.md")
        _write_article("# Title\nContent here", path)
        assert open(path, encoding="utf-8").read() == "# Title\nContent here"

    def test_creates_missing_parent_directories(self, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "article.md")
        _write_article("content", path)
        assert os.path.exists(path)

    def test_overwrites_existing_file(self, tmp_path):
        path = str(tmp_path / "article.md")
        _write_article("original", path)
        _write_article("updated", path)
        assert open(path, encoding="utf-8").read() == "updated"


# ---------------------------------------------------------------------------
# generate_kb_articles
# ---------------------------------------------------------------------------

class TestGenerateKbArticles:

    def _run(self, df, summaries, client=None, write_patch=None):
        """Helper: run generate_kb_articles with _write_article mocked."""
        client = client or _make_client()
        with patch("src.kb_generator._write_article") as mock_write:
            results = generate_kb_articles(df, summaries, client)
        return results, mock_write

    def test_returns_list(self):
        df = _make_df([{"cluster": 0}])
        summaries = _make_summaries([{"cluster": 0, "size": 1}])
        results, _ = self._run(df, summaries)
        assert isinstance(results, list)

    def test_one_result_per_cluster(self):
        df = _make_df([{"cluster": 0}, {"cluster": 1}])
        summaries = _make_summaries([
            {"cluster": 0, "size": 1},
            {"cluster": 1, "size": 1},
        ])
        results, _ = self._run(df, summaries)
        assert len(results) == 2

    def test_empty_summaries_returns_empty_list(self):
        df = _make_df([])
        summaries = _make_summaries([])
        results, _ = self._run(df, summaries)
        assert results == []

    def test_two_claude_calls_per_cluster(self):
        df = _make_df([{"cluster": 0}, {"cluster": 1}])
        summaries = _make_summaries([
            {"cluster": 0, "size": 1},
            {"cluster": 1, "size": 1},
        ])
        client = _make_client()
        with patch("src.kb_generator._write_article"):
            generate_kb_articles(df, summaries, client)
        assert client.messages.create.call_count == 4  # 2 clusters × 2 audiences

    def test_two_files_written_per_cluster(self):
        df = _make_df([{"cluster": 0}])
        summaries = _make_summaries([{"cluster": 0, "size": 1}])
        _, mock_write = self._run(df, summaries)
        assert mock_write.call_count == 2

    def test_result_dict_keys(self):
        df = _make_df([{"cluster": 0}])
        summaries = _make_summaries([{"cluster": 0, "size": 1, "top_terms": "api error"}])
        results, _ = self._run(df, summaries)
        keys = set(results[0].keys())
        assert keys == {"cluster", "size", "top_terms", "internal_path", "lender_path"}

    def test_internal_path_in_output_internal(self):
        df = _make_df([{"cluster": 0}])
        summaries = _make_summaries([{"cluster": 0, "size": 1}])
        results, _ = self._run(df, summaries)
        assert "output/internal" in results[0]["internal_path"]

    def test_lender_path_in_output_lender_facing(self):
        df = _make_df([{"cluster": 0}])
        summaries = _make_summaries([{"cluster": 0, "size": 1}])
        results, _ = self._run(df, summaries)
        assert "output/lender-facing" in results[0]["lender_path"]

    def test_filenames_include_cluster_id(self):
        df = _make_df([{"cluster": 3}])
        summaries = _make_summaries([{"cluster": 3, "size": 1}])
        results, _ = self._run(df, summaries)
        assert "cluster_3" in results[0]["internal_path"]
        assert "cluster_3" in results[0]["lender_path"]

    def test_result_preserves_size_and_top_terms(self):
        df = _make_df([{"cluster": 0}])
        summaries = _make_summaries([{"cluster": 0, "size": 7, "top_terms": "jwt auth token"}])
        results, _ = self._run(df, summaries)
        assert results[0]["size"] == 7
        assert results[0]["top_terms"] == "jwt auth token"

    def test_different_system_prompts_used_per_audience(self):
        """Internal and lender articles must be requested with different system prompts."""
        df = _make_df([{"cluster": 0}])
        summaries = _make_summaries([{"cluster": 0, "size": 1}])
        client = _make_client()
        with patch("src.kb_generator._write_article"):
            generate_kb_articles(df, summaries, client)
        system_prompts = [
            c[1]["system"] for c in client.messages.create.call_args_list
        ]
        assert len(set(system_prompts)) == 2  # two distinct prompts
