"""
Hybrid PII masker for Freshdesk ticket data.

Pass 1 — Regex: structured PII (email, phone, PAN, Aadhaar, bank account, IFSC, IP).
Pass 2 — spaCy NER (en_core_web_lg, local): person names, org names.
         Runs entirely in-process — no data leaves the machine.
"""

import re
import spacy
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

try:
    _nlp = spacy.load("en_core_web_lg")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_lg' is not installed. "
        "Run: python -m spacy download en_core_web_lg"
    )


# ---------------------------------------------------------------------------
# Pass 1: regex patterns
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Email
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
    # Indian mobile: +91 followed by 10 digits, or standalone 10-digit starting 6-9
    (re.compile(r"(?:\+91[\s\-]?)?[6-9]\d{9}\b"), "[PHONE]"),
    # PAN: 5 uppercase letters, 4 digits, 1 uppercase letter
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[PAN]"),
    # Aadhaar: 12 digits, optionally separated by spaces or hyphens
    (re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[AADHAAR]"),
    # IFSC: 4 uppercase letters, '0', then 6 alphanumeric characters
    (re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), "[IFSC]"),
    # Bank account: 9–18 consecutive digits (after other patterns so Aadhaar wins)
    (re.compile(r"\b\d{9,18}\b"), "[BANK_ACCOUNT]"),
    # IP address (v4)
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP_ADDRESS]"),
]


def _regex_mask(text: str) -> str:
    """Apply all regex patterns to a string."""
    if not text:
        return text
    for pattern, token in _PATTERNS:
        text = pattern.sub(token, text)
    return text


# ---------------------------------------------------------------------------
# Pass 2: spaCy NER
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    "PERSON": "[PERSON_NAME]",
    "ORG": "[ORG_NAME]",
}

_CHUNK_SIZE = 80_000  # chars per spaCy pass (well under the 1M default limit)


def _ner_mask(text: str) -> str:
    """Use spaCy to redact PERSON and ORG entities from already regex-masked text.

    Long texts are split into chunks so they stay within spaCy's max_length.
    Entity offsets are translated back to absolute positions before replacement.
    """
    if not text or not text.strip():
        return text

    # Collect (abs_start, abs_end, replacement_token) across all chunks
    replacements = []
    for chunk_start in range(0, len(text), _CHUNK_SIZE):
        chunk = text[chunk_start: chunk_start + _CHUNK_SIZE]
        doc = _nlp(chunk)
        for ent in doc.ents:
            token = _LABEL_MAP.get(ent.label_)
            if token:
                replacements.append((
                    chunk_start + ent.start_char,
                    chunk_start + ent.end_char,
                    token,
                ))

    # Replace in reverse order to preserve offsets
    for start, end, token in sorted(replacements, key=lambda x: x[0], reverse=True):
        text = text[:start] + token + text[end:]

    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_TEXT_FIELDS = ("subject", "description_text", "description")


def mask_ticket(ticket: dict) -> dict:
    """
    Apply hybrid PII masking to a single ticket dict.

    - Regex pass on every text field and conversation bodies.
    - spaCy NER pass on each text field (local, no data egress).
    - Agent name, ticket ID, status, timestamps, and category are NOT masked.

    Returns a new dict with masked text fields.
    """
    masked = dict(ticket)

    # --- Pass 1 + 2: regex then spaCy NER on individual text fields ---
    for field in _TEXT_FIELDS:
        if field in masked and masked[field]:
            val = _regex_mask(str(masked[field]))
            masked[field] = _ner_mask(val)

    # Regex + NER on conversation bodies
    if "conversations" in masked and masked["conversations"]:
        masked_convs = []
        for conv in masked["conversations"]:
            conv = dict(conv)
            for body_field in ("body_text", "body"):
                if conv.get(body_field):
                    val = _regex_mask(str(conv[body_field]))
                    conv[body_field] = _ner_mask(val)
            masked_convs.append(conv)
        masked["conversations"] = masked_convs

    return masked


def mask_all_tickets(tickets: list[dict]) -> list[dict]:
    """
    Mask PII in a list of tickets, showing a rich progress bar.

    Returns list of masked ticket dicts.
    """
    masked_tickets = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Masking PII (local spaCy NER)..."),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} tickets"),
    ) as progress:
        task = progress.add_task("masking", total=len(tickets))
        for ticket in tickets:
            masked_tickets.append(mask_ticket(ticket))
            progress.advance(task)

    return masked_tickets
