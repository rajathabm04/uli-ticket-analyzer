"""
Feb Analyzer: generates a slide-ready monthly recap insights document.

Writes output/feb_deck_insights.md with structured sections covering
ticket volume, API/service breakdown, issue types, signal vs noise,
clusters, agent audit, and data-driven recommendations.

Zero Claude API calls — all analysis is pure Python.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone

import pandas as pd


# ---------------------------------------------------------------------------
# API / service keyword map (keyword → display label)
# ---------------------------------------------------------------------------

_API_KEYWORDS: dict[str, str] = {
    "penny-drop":            "Penny Drop",
    "penny drop":            "Penny Drop",
    "pan-protean":           "PAN Verification (Protean)",
    "pan protean":           "PAN Verification (Protean)",
    "pan verification":      "PAN Verification (Protean)",
    "account-aggregator":    "Account Aggregator",
    "account aggregator":    "Account Aggregator",
    "aaconsent":             "Account Aggregator",
    "ekyc":                  "eKYC / Land Record",
    "land record":           "eKYC / Land Record",
    "vehicle hypothecation": "Vehicle Hypothecation",
    "nesl":                  "NESL",
    "cibil":                 "CIBIL",
    "opv":                   "OPV / PFX Certificate",
    "pfx certificate":       "OPV / PFX Certificate",
    "pfx":                   "OPV / PFX Certificate",
    "ifsc":                  "IFSC Lookup",
}

# Patterns used to classify a ticket as automated noise
_NOISE_PATTERNS = re.compile(
    r"\b(firing|resolved alert|test ticket|junk|out of office|ooo|auto-reply|automatic reply)\b",
    re.IGNORECASE,
)


def _count_api_mentions(df: pd.DataFrame) -> dict[str, int]:
    """Return count of tickets mentioning each API/service (by display label)."""
    label_counts: dict[str, int] = {}
    combined = (
        df["subject"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["conversations"].fillna("")
    ).str.lower()

    seen: dict[int, set[str]] = {}  # row_idx → set of labels already counted

    for keyword, label in _API_KEYWORDS.items():
        mask = combined.str.contains(re.escape(keyword), na=False)
        for idx in combined[mask].index:
            if idx not in seen:
                seen[idx] = set()
            if label not in seen[idx]:
                seen[idx].add(label)
                label_counts[label] = label_counts.get(label, 0) + 1

    return label_counts


def _weekly_counts(df: pd.DataFrame, since: str, until: str) -> list[tuple[str, int]]:
    """Return (week_label, count) for each week in the period."""
    dates = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    weeks = []
    # Build 4 week bins: Feb 1-7, 8-14, 15-21, 22-28/29
    since_dt = pd.Timestamp(since, tz="UTC")
    until_dt = pd.Timestamp(until, tz="UTC")
    week_start = since_dt
    week_num = 1
    while week_start < until_dt:
        week_end = week_start + pd.Timedelta(days=7)
        if week_end > until_dt:
            week_end = until_dt
        count = int(((dates >= week_start) & (dates < week_end)).sum())
        label = f"Week {week_num} ({week_start.strftime('%b %d')} – {(week_end - pd.Timedelta(days=1)).strftime('%b %d')})"
        weeks.append((label, count))
        week_start = week_end
        week_num += 1
    return weeks


def _is_noise(row: pd.Series) -> bool:
    text = f"{row.get('subject', '')} {row.get('description', '')} {row.get('conversations', '')}"
    return bool(_NOISE_PATTERNS.search(text))


def _derive_recommendations(
    df: pd.DataFrame,
    api_counts: dict[str, int],
    cluster_df: pd.DataFrame | None,
    noise_pct: float,
) -> dict[str, list[str]]:
    """Derive product/engineering/support recommendations from data."""
    product: list[str] = []
    engineering: list[str] = []
    support: list[str] = []

    # Product: high-volume APIs
    for label, count in sorted(api_counts.items(), key=lambda x: -x[1]):
        if count > 10:
            product.append(
                f"**{label}** ({count} tickets) — improve error messages and add retry/backoff guidance in docs"
            )

    data_mismatch_count = int((df.get("inferred_category", pd.Series(dtype=str)) == "Data Mismatch").sum()) if "inferred_category" in df.columns else 0
    if data_mismatch_count > 0:
        product.append(
            f"{data_mismatch_count} Data Mismatch tickets — review API contract versioning and field-level change-log"
        )

    if not product:
        product.append("No single API exceeded 10 tickets — monitor for emerging hotspots next month")

    # Engineering
    alert_cluster_exists = False
    pfx_cluster_exists = False
    if cluster_df is not None and not cluster_df.empty:
        terms_col = cluster_df["top_terms"].str.lower() if "top_terms" in cluster_df.columns else pd.Series(dtype=str)
        alert_cluster_exists = terms_col.str.contains(r"firing|alert|resolved", na=False, regex=True).any()
        pfx_cluster_exists = terms_col.str.contains(r"pfx|certificate|cert", na=False, regex=True).any()

    if alert_cluster_exists:
        engineering.append(
            "Alert noise cluster detected — tune Alertmanager thresholds and add `for: 5m` stabilization windows"
        )
    if pfx_cluster_exists:
        engineering.append(
            "PFX / certificate cluster detected — automate cert expiry scanning and build proactive renewal workflow"
        )
    if noise_pct > 0.20:
        engineering.append(
            f"Email noise is {noise_pct:.0%} of tickets — implement RFC 3834 header filtering in Freshdesk email gateway"
        )
    if not engineering:
        engineering.append("No critical engineering signals this month — continue monitoring alert cluster ratio")

    # Support
    support.append("Publish KB articles for all clusters identified this month (see Recurring Clusters section)")
    support.append("Add triage rule: auto-close FIRING / RESOLVED alert tickets within 10 min of creation")
    if noise_pct > 0.10:
        support.append(
            f"Noise tickets ({noise_pct:.0%} of volume) are diluting SLA metrics — exclude from SLA reporting or auto-resolve"
        )
    if data_mismatch_count > 0:
        support.append(
            f"Train agents on Data Mismatch vs API Error distinction ({data_mismatch_count} mismatch tickets this month)"
        )

    return {"Product": product, "Engineering": engineering, "Support": support}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_deck_insights(
    df: pd.DataFrame,
    summaries: pd.DataFrame,
    mismatches: pd.DataFrame,
    agent_summary: pd.DataFrame,
    since: str = "2026-02-01",
    until: str = "2026-03-01",
    output_path: str = "output/feb_deck_insights.md",
) -> None:
    """
    Write a slide-ready monthly insights document to output_path.

    Args:
        df: Full ticket DataFrame (must have inferred_category column).
        summaries: Cluster summaries DataFrame from clusterer.cluster_summaries().
        mismatches: Mismatch DataFrame from mismatch.find_mismatches().
        agent_summary: Per-agent mismatch counts from agent_audit.summarise_by_agent().
        since: Start date for labelling (YYYY-MM-DD).
        until: End (exclusive) date for labelling (YYYY-MM-DD).
        output_path: File path to write the markdown document.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = len(df)
    since_label = datetime.strptime(since, "%Y-%m-%d").strftime("%B %-d, %Y")
    until_label = datetime.strptime(until, "%Y-%m-%d").strftime("%B %-d, %Y")

    # Status breakdown
    status_counts: dict = df["status"].value_counts().to_dict() if "status" in df.columns else {}
    resolved_count = status_counts.get("resolved", 0) + status_counts.get("closed", 0)
    resolved_pct = resolved_count / total * 100 if total else 0

    # Weekly volume
    weekly = _weekly_counts(df, since, until)

    # API/service mentions
    api_counts = _count_api_mentions(df)
    api_sorted = sorted(api_counts.items(), key=lambda x: -x[1])

    # Issue type breakdown
    if "inferred_category" in df.columns:
        cat_counts = df["inferred_category"].value_counts()
    elif "category" in df.columns:
        cat_counts = df["category"].value_counts()
    else:
        cat_counts = pd.Series(dtype=int)

    # Signal vs noise
    noise_mask = df.apply(_is_noise, axis=1)
    noise_count = int(noise_mask.sum())
    real_count = total - noise_count
    noise_pct = noise_count / total if total else 0

    # Recommendations
    recs = _derive_recommendations(df, api_counts, summaries, noise_pct)

    # Agent mismatch rate
    agent_rows: list[tuple] = []
    if not agent_summary.empty and total > 0:
        agent_ticket_counts = df["agent"].value_counts() if "agent" in df.columns else pd.Series(dtype=int)
        for _, row in agent_summary.iterrows():
            agent = str(row["agent"])
            mm_count = int(row["mismatch_count"])
            agent_total = int(agent_ticket_counts.get(agent, 0))
            rate = mm_count / agent_total * 100 if agent_total else 0
            agent_rows.append((agent, mm_count, agent_total, rate))

    # -----------------------------------------------------------------------
    # Build the markdown document
    # -----------------------------------------------------------------------
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"{'#' * level} {text}\n")

    def p(*parts: str) -> None:
        lines.append(" ".join(parts) + "\n")

    def blank() -> None:
        lines.append("\n")

    # Header
    lines.append(f"# ULI Support — February 2026 Recap\n")
    lines.append(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Data: {since_label} – {until_label}_\n")
    blank()
    lines.append("---\n")
    blank()

    # Slide 1: Overview
    h(2, "Slide 1 — Overview")
    lines.append(f"| Metric | Value |\n|--------|-------|\n")
    lines.append(f"| Total tickets | **{total}** |\n")
    lines.append(f"| Date range | {since_label} – {until_label} |\n")
    lines.append(f"| Resolved / Closed | {resolved_count} ({resolved_pct:.0f}%) |\n")
    lines.append(f"| Real lender issues | {real_count} ({(real_count/total*100) if total else 0:.0f}%) |\n")
    lines.append(f"| Automated / Noise | {noise_count} ({noise_pct:.0%}) |\n")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {status} | {count} |\n")
    blank()
    lines.append("---\n")
    blank()

    # Slide 2: Volume by Week
    h(2, "Slide 2 — Volume by Week")
    lines.append("| Week | Tickets |\n|------|---------|\n")
    for label, count in weekly:
        lines.append(f"| {label} | {count} |\n")
    blank()
    lines.append("---\n")
    blank()

    # Slide 3: Issues by API / Service
    h(2, "Slide 3 — Issues by API / Service")
    if api_sorted:
        lines.append("| API / Service | Tickets Mentioning |\n|---------------|-------------------|\n")
        for label, count in api_sorted:
            lines.append(f"| {label} | {count} |\n")
    else:
        lines.append("_No specific API/service keywords detected in ticket subjects or descriptions._\n")
    blank()
    lines.append("---\n")
    blank()

    # Slide 4: Issue Type Breakdown
    h(2, "Slide 4 — Issue Type Breakdown")
    if not cat_counts.empty:
        lines.append("| Category | Count | Share |\n|----------|-------|-------|\n")
        for cat, count in cat_counts.items():
            share = count / total * 100 if total else 0
            lines.append(f"| {cat} | {count} | {share:.0f}% |\n")
    else:
        lines.append("_No category data available._\n")
    blank()
    lines.append("---\n")
    blank()

    # Slide 5: Signal vs Noise
    h(2, "Slide 5 — Signal vs Noise")
    lines.append("| Type | Count | Share |\n|------|-------|-------|\n")
    lines.append(f"| Real lender issues | {real_count} | {(real_count/total*100) if total else 0:.0f}% |\n")
    lines.append(f"| Automated / Noise (alerts, OOO, test, junk) | {noise_count} | {noise_pct:.0%} |\n")
    blank()
    lines.append(
        "_Noise detection: tickets matching keywords `FIRING`, `RESOLVED`, `test ticket`, "
        "`junk`, `out of office`, `OOO`, `auto-reply`._\n"
    )
    blank()
    lines.append("---\n")
    blank()

    # Slide 6: Recurring Clusters
    h(2, "Slide 6 — Recurring Clusters")
    if not summaries.empty:
        lines.append("| Cluster | Size | Top 5 Terms | Theme |\n|---------|------|-------------|-------|\n")
        for _, row in summaries.iterrows():
            cid = int(row["cluster"])
            size = int(row["size"])
            terms_all = str(row.get("top_terms", ""))
            top5 = ", ".join(terms_all.split(", ")[:5])
            # Derive a one-line theme from top terms
            theme = _infer_theme(top5)
            lines.append(f"| {cid} | {size} | {top5} | {theme} |\n")
    else:
        lines.append("_No clusters generated (insufficient ticket volume)._\n")
    blank()
    lines.append("---\n")
    blank()

    # Slide 7: Agent Audit
    h(2, "Slide 7 — Agent Audit")
    if agent_rows:
        lines.append("| Agent | Mismatches | Tickets Handled | Mismatch Rate |\n|-------|------------|-----------------|---------------|\n")
        for agent, mm, total_agent, rate in agent_rows:
            lines.append(f"| {agent} | {mm} | {total_agent} | {rate:.0f}% |\n")
    elif mismatches.empty:
        lines.append("_No category mismatches detected this month._\n")
    else:
        lines.append("_Agent data not available._\n")
    blank()
    lines.append(f"**Overall mismatch rate:** {len(mismatches)}/{total} tickets ({len(mismatches)/total*100 if total else 0:.0f}%)\n")
    blank()
    lines.append("---\n")
    blank()

    # Slide 8: Recommendations
    h(2, "Slide 8 — Recommendations")
    for team, items in recs.items():
        h(3, f"{team}")
        for item in items:
            lines.append(f"- {item}\n")
        blank()
    lines.append("---\n")
    blank()
    lines.append("_End of February 2026 recap. Paste into Claude console or Gamma to build the deck._\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"  output/feb_deck_insights.md written ({total} tickets, {len(summaries)} clusters)")


def _infer_theme(top_terms: str) -> str:
    """Derive a short one-line theme from comma-separated top TF-IDF terms."""
    terms_lower = top_terms.lower()
    if any(t in terms_lower for t in ["firing", "alert", "resolved", "prometheus"]):
        return "Monitoring alert noise"
    if any(t in terms_lower for t in ["pfx", "certificate", "cert", "expired"]):
        return "PFX / certificate expiry"
    if any(t in terms_lower for t in ["penny", "drop", "bank", "account"]):
        return "Penny drop / bank verification"
    if any(t in terms_lower for t in ["pan", "protean", "verification"]):
        return "PAN verification failures"
    if any(t in terms_lower for t in ["consent", "aggregator", "aa"]):
        return "Account Aggregator / consent flow"
    if any(t in terms_lower for t in ["onboard", "registration", "kyc", "activation"]):
        return "Lender onboarding"
    if any(t in terms_lower for t in ["token", "auth", "401", "403", "credential"]):
        return "Authentication / token issues"
    if any(t in terms_lower for t in ["timeout", "latency", "slow", "performance"]):
        return "Performance / latency"
    if any(t in terms_lower for t in ["ifsc", "lookup"]):
        return "IFSC lookup"
    if any(t in terms_lower for t in ["cibil", "bureau", "credit"]):
        return "CIBIL / credit bureau"
    return "Mixed / general issues"
