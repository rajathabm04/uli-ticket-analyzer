# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This tool is used by the Support Lead at RBIH (Reserve Bank Innovation Hub) to analyze Freshdesk support tickets for the **ULI (Unified Lending Interface)** platform. ULI is a financial data-sharing infrastructure connecting lenders (banks, NBFCs, fintechs) with data providers. Support tickets come from lenders experiencing integration issues, API errors, onboarding problems, etc.

The tool fetches tickets live from the Freshdesk API, masks PII before any analysis, then:
1. Infers the correct category for each ticket (vs the agent-assigned category)
2. Flags mismatches and identifies which agents are miscategorizing
3. Clusters recurring issues across tickets
4. Auto-generates KB articles in markdown — both internal (for support agents) and lender-facing (for self-serve)

## Commands

```bash
# Activate virtual environment (required before running anything)
source venv/bin/activate

# Install / sync dependencies
pip install -r requirements.txt

# One-time: download the spaCy model used for local PII NER
python -m spacy download en_core_web_sm

# Run the analyzer
python analyze_tickets.py
```

Set the following in a `.env` file at the project root (loaded via `python-dotenv`):

```
ANTHROPIC_API_KEY=sk-ant-...
FRESHDESK_DOMAIN=yourcompany        # → yourcompany.freshdesk.com
FRESHDESK_API_KEY=your_fd_api_key
```

## Architecture

**Entry point:** `analyze_tickets.py` — loads env vars, initialises clients, orchestrates the pipeline.

**Pipeline stages (each in `src/`):**
- `freshdesk.py` — `FreshdeskClient` class; paginates `GET /tickets?include=description,conversations` with HTTP Basic auth
- `pii_masker.py` — hybrid masker: regex pass (email, phone, PAN, Aadhaar, IFSC, bank account, IP) then spaCy NER pass (`en_core_web_lg`, local — no data egress) for person names/org names
- `loader.py` — calls Freshdesk API → PII masker → normalises into DataFrame with columns: `id, subject, description, conversations, category, agent, status, created_at`
- `categorizer.py` — sends ticket content to Claude to infer the correct category
- `mismatch.py` — compares assigned vs inferred categories, returns mismatch records
- `agent_audit.py` — aggregates mismatches per agent, surfaces miscategorization patterns
- `clusterer.py` — clusters tickets by recurring issue (uses embeddings or TF-IDF + sklearn)
- `kb_generator.py` — takes clusters and generates markdown KB articles via Claude

**Output directories:**
- `output/internal/` — KB articles for support agents (detailed, technical)
- `output/lender-facing/` — KB articles for lenders (self-serve, simplified)

## Key Conventions

- The sibling project `uli-yaml-generator/` uses the same `anthropic` SDK pattern: a single script calling `anthropic.Anthropic()` with `client.messages.create(...)`. Follow the same style.
- Use `claude-sonnet-4-6` as the default model for categorization and KB generation.
- PII NER masking uses spaCy `en_core_web_sm` (local model, zero data egress). Claude Haiku is no longer used for masking.
- Freshdesk API: auth is HTTP Basic with the API key as username and `"X"` as password.
- Agent names are **not** masked (kept for the agent audit feature). Ticket ID, status, timestamps, and category label are also preserved.
- All Claude prompts should be in `src/` modules, not in the entry point.
