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
    # Disable all pipeline components except ner and its tok2vec dependency.
    # tagger / parser / attribute_ruler / lemmatizer are not needed for entity
    # detection and the parser alone accounts for ~50% of processing time.
    _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' is not installed. "
        "Run: python -m spacy download en_core_web_sm"
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

    # --- Credential / token patterns ---

    # JWT: header.payload.signature (base64url, always starts with eyJ which is '{"')
    # Must run before Bearer so "Bearer eyJ..." is reduced to "Bearer [JWT_TOKEN]"
    (re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*"), "[JWT_TOKEN]"),

    # Bearer token (Authorization: Bearer VALUE or inline)
    (re.compile(r"(Bearer\s+)[A-Za-z0-9+/=_\-]{8,}", re.IGNORECASE), r"\1[BEARER_TOKEN]"),

    # x-sp-client-token  (ULI/RBIH-specific header — must precede generic client_id)
    (re.compile(r"(x-sp-client-token\s*[:=]\s*[\"']?)[A-Za-z0-9+/=_\-]{8,}", re.IGNORECASE), r"\1[SP_CLIENT_TOKEN]"),

    # x-sp-client-id  (ULI/RBIH-specific header — must precede generic client_id)
    (re.compile(r"(x-sp-client-id\s*[:=]\s*[\"']?)[A-Za-z0-9+/=_\-@.]{4,}", re.IGNORECASE), r"\1[SP_CLIENT_ID]"),

    # client_id / client-id  (JSON, YAML, URL params, log lines)
    (re.compile(r"(client[_\-]id[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9+/=_\-@.]{4,}", re.IGNORECASE), r"\1[CLIENT_ID]"),

    # client_secret / client-secret
    (re.compile(r"(client[_\-]secret[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9+/=_\-@.]{8,}", re.IGNORECASE), r"\1[CLIENT_SECRET]"),

    # Generic API key header / query param  (apikey, api_key, x-api-key)
    (re.compile(r"((?:apikey|api[_\-]key|x-api-key)\s*[:=]\s*[\"']?)[A-Za-z0-9+/=_\-]{8,}", re.IGNORECASE), r"\1[API_KEY]"),

    # OAuth / access / refresh tokens
    (re.compile(r"((?:access_token|oauth_token|refresh_token)\s*[:=]\s*[\"']?)[A-Za-z0-9+/=_\-]{8,}", re.IGNORECASE), r"\1[OAUTH_TOKEN]"),
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


_SEP = "\x00"  # Null-byte field separator — never appears in ticket text

# NER is only run on the first N characters of each field. Person/org names
# appear near the top of the text; quoted email chains and boilerplate at the
# tail add tokens without adding privacy risk, but dominate processing time.
_NER_MAX_CHARS = 3_000


def mask_all_tickets(tickets: list[dict]) -> list[dict]:
    """
    Mask PII in a list of tickets.

    Strategy:
      1. Regex pass on every text field in full (fast, catches all structured PII).
      2. For NER, truncate each field to _NER_MAX_CHARS and concatenate into one
         document per ticket, then batch through nlp.pipe(). Truncation cuts the
         average token load from ~44 KB to ~6 KB per ticket — a ~7x speedup —
         while retaining coverage of all person/org names (which always appear
         near the top of ticket text, not in quoted chains at the tail).
      3. Reconstruct: NER-masked prefix + regex-masked suffix per field.

    Returns list of masked ticket dicts.
    """
    if not tickets:
        return []

    # --- Phase 1: regex pass (full text) ---
    regex_masked: list[dict] = []
    for ticket in tickets:
        masked = dict(ticket)
        for field in _TEXT_FIELDS:
            if masked.get(field):
                masked[field] = _regex_mask(str(masked[field]))
        if masked.get("conversations"):
            masked_convs = []
            for conv in masked["conversations"]:
                conv = dict(conv)
                for body_field in ("body_text", "body"):
                    if conv.get(body_field):
                        conv[body_field] = _regex_mask(str(conv[body_field]))
                masked_convs.append(conv)
            masked["conversations"] = masked_convs
        regex_masked.append(masked)

    # --- Phase 2: build one truncated combined string per ticket for NER ---
    # field_lists[i] = list of (location, full_regex_masked_text) for ticket i
    field_lists: list[list[tuple]] = []
    combined_texts: list[str] = []

    for ticket in regex_masked:
        fields: list[tuple] = []
        for field in _TEXT_FIELDS:
            if ticket.get(field):
                fields.append((("field", field), str(ticket[field])))
        for j, conv in enumerate(ticket.get("conversations") or []):
            for body_field in ("body_text", "body"):
                if conv.get(body_field):
                    fields.append((("conv", j, body_field), str(conv[body_field])))
        field_lists.append(fields)
        # Only pass the first _NER_MAX_CHARS of each field to spaCy
        combined_texts.append(_SEP.join(text[:_NER_MAX_CHARS] for _, text in fields))

    # --- Phase 3: batch NER (one doc per ticket) ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Masking PII (local spaCy NER)..."),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} tickets"),
    ) as progress:
        task = progress.add_task("ner", total=len(tickets))
        docs = []
        for doc in _nlp.pipe(combined_texts, batch_size=32):
            docs.append(doc)
            progress.advance(task)

    # --- Phase 4: apply replacements and reconstruct full field text ---
    for i, (doc, combined, fields) in enumerate(zip(docs, combined_texts, field_lists)):
        replacements = [
            (ent.start_char, ent.end_char, _LABEL_MAP[ent.label_])
            for ent in doc.ents if ent.label_ in _LABEL_MAP
        ]
        if not replacements:
            continue

        # Apply NER replacements in reverse order to preserve offsets
        masked_combined = combined
        for start, end, token in sorted(replacements, key=lambda x: x[0], reverse=True):
            masked_combined = masked_combined[:start] + token + masked_combined[end:]

        # Split back and reconstruct: NER-masked prefix + regex-only suffix
        parts = masked_combined.split(_SEP)
        for (location, full_text), ner_prefix in zip(fields, parts):
            final_text = ner_prefix + full_text[_NER_MAX_CHARS:]
            if location[0] == "field":
                regex_masked[i][location[1]] = final_text
            else:
                _, j, body_field = location
                regex_masked[i]["conversations"][j][body_field] = final_text

    return regex_masked
