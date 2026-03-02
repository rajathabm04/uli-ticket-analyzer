"""
KB Exporter: converts existing KB markdown articles to Freshdesk-compatible
HTML files and a JSON manifest matching the Freshdesk Solution Articles API.

Zero API calls — pure file I/O + markdown → HTML conversion.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import markdown as md_lib


# ---------------------------------------------------------------------------
# Tag inference
# ---------------------------------------------------------------------------

_TAG_KEYWORDS: dict[str, str] = {
    "penny-drop":         "penny-drop",
    "penny drop":         "penny-drop",
    "pan":                "pan",
    "pan-protean":        "pan",
    "account-aggregator": "account-aggregator",
    "account aggregator": "account-aggregator",
    "aaconsent":          "account-aggregator",
    "ekyc":               "ekyc",
    "land record":        "land-record",
    "vehicle hypothecation": "vehicle-hypothecation",
    "nesl":               "nesl",
    "cibil":              "cibil",
    "opv":                "opv",
    "pfx":                "pfx-certificate",
    "certificate":        "pfx-certificate",
    "ifsc":               "ifsc",
    "authentication":     "authentication",
    "onboarding":         "onboarding",
    "integration":        "integration",
    "api error":          "api-error",
    "latency":            "performance",
    "performance":        "performance",
    "configuration":      "configuration",
}


def _infer_tags(text: str) -> list[str]:
    """Return a deduplicated list of tags inferred from the article text."""
    lower = text.lower()
    seen: set[str] = set()
    tags: list[str] = ["uli"]  # always present
    for keyword, tag in _TAG_KEYWORDS.items():
        if keyword in lower and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def _extract_title(markdown_text: str) -> str:
    """Return the text of the first H1 heading, or a fallback."""
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return "Untitled Article"


def _md_to_html(markdown_text: str) -> str:
    """Convert markdown to HTML using the standard `markdown` library."""
    return md_lib.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_for_freshdesk(
    internal_dir: str = "output/internal",
    lender_dir: str = "output/lender-facing",
    output_dir: str = "output/freshdesk_export",
) -> list[dict]:
    """
    Convert KB markdown articles to Freshdesk-compatible HTML + JSON manifest.

    Reads every .md file from internal_dir and lender_dir, converts each to
    HTML, writes them under output_dir, and writes a manifest.json.

    Args:
        internal_dir: Directory containing internal KB markdown files.
        lender_dir: Directory containing lender-facing KB markdown files.
        output_dir: Root output directory for the Freshdesk export.

    Returns:
        List of manifest dicts (one per article), matching the Freshdesk
        Solution Articles API shape.
    """
    folders = [
        ("internal",       internal_dir),
        ("lender-facing",  lender_dir),
    ]

    manifest: list[dict] = []

    for folder_name, source_dir in folders:
        source_path = Path(source_dir)
        if not source_path.exists():
            continue

        out_path = Path(output_dir) / folder_name
        out_path.mkdir(parents=True, exist_ok=True)

        for md_file in sorted(source_path.glob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            title = _extract_title(text)
            html_body = _md_to_html(text)
            tags = _infer_tags(text)

            html_filename = md_file.stem + ".html"
            html_path = out_path / html_filename
            html_path.write_text(html_body, encoding="utf-8")

            manifest.append({
                "title":        title,
                "description":  html_body,
                "article_type": 1,
                "folder":       folder_name,
                "tags":         tags,
                "source_file":  md_file.name,
            })

    # Write manifest
    manifest_path = Path(output_dir) / "manifest.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    internal_count = sum(1 for m in manifest if m["folder"] == "internal")
    lender_count = sum(1 for m in manifest if m["folder"] == "lender-facing")
    print(
        f"  Freshdesk export: {len(manifest)} articles "
        f"({internal_count} internal, {lender_count} lender-facing) → {output_dir}/"
    )

    return manifest
