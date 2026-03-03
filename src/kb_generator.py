"""
KB Generator: generates markdown KB articles from ticket clusters.

Uses a template engine keyed on cluster topic (inferred from top TF-IDF terms
and inferred_category). Zero Claude API calls — the client parameter is
accepted for API compatibility but is never invoked.

For each cluster two articles are written:
  - output/internal/      : detailed, technical, for support agents
  - output/lender-facing/ : simplified, self-serve, for lenders
"""

import os
import re
import textwrap

import anthropic
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

_OUTPUT_DIRS = {
    "internal": "output/internal",
    "lender":   "output/lender-facing",
}

_MAX_SAMPLE  = 8    # ticket subjects to list in the article
_DESC_LIMIT  = 400  # chars of description to quote in the article


# ---------------------------------------------------------------------------
# Topic detection
# ---------------------------------------------------------------------------

_TOPIC_RULES: list[tuple[str, list[str]]] = [
    # Most specific first
    ("pfx_cert",      ["pfx", "certificate", "cert", "expir", "renewal", "opv"]),
    ("alert_noise",   ["firing", "resolved", "alert", "alertmanager", "prometheus", "threshold"]),
    ("penny_drop",    ["penny", "drop", "verifybankaccount", "bank account", "npci"]),
    ("pan_verify",    ["pan", "protean", "pan-protean", "pan verification"]),
    ("account_agg",   ["aggregator", "aaconsent", "aa", "fip", "consent"]),
    ("nesl",          ["nesl"]),
    ("cibil",         ["cibil", "bureau", "credit score"]),
    ("ekcc_land",     ["ekcc", "land record", "khata", "survey", "ekcc"]),
    ("vehicle_hyp",   ["vehicle", "hypothecation", "rc detail", "vahan"]),
    ("ifsc",          ["ifsc", "lookup"]),
    ("onboarding",    ["onboard", "registration", "kyc", "activation", "new lender"]),
    ("auth",          ["token", "auth", "401", "403", "credential", "oauth", "bearer", "jwt"]),
    ("performance",   ["latency", "timeout", "slow", "degradation", "response time", "cid"]),
    ("email_noise",   ["email", "mail", "ooo", "auto-reply", "out of office", "forwarded"]),
    ("test_junk",     ["test", "junk", "dummy", "testing"]),
    ("data_mismatch", ["mismatch", "discrepancy", "wrong value", "incorrect data"]),
    ("config",        ["config", "misconfigur", "parameter", "setting", "environment"]),
    ("general",       []),   # fallback
]


def _infer_topic(top_terms: str, subjects: list[str]) -> str:
    combined = (top_terms + " " + " ".join(subjects)).lower()
    for topic, keywords in _TOPIC_RULES:
        if any(kw in combined for kw in keywords):
            return topic
    return "general"


# ---------------------------------------------------------------------------
# Template content per topic
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, dict] = {

    "alert_noise": {
        "title": "Automated Alertmanager Alerts — FIRING & RESOLVED Noise in Support Queue",
        "summary": (
            "The ULI platform's Alertmanager routes monitoring alerts directly into the "
            "Freshdesk support queue as tickets. These appear as `[FIRING:N]` or `[RESOLVED]` "
            "subjects with a fingerprint hash. They are not raised by lenders and require no "
            "manual resolution — they self-resolve when the monitored condition clears."
        ),
        "root_causes": [
            "Alert flapping: metric briefly crosses the threshold and recovers (insufficient `for:` stabilization window)",
            "Upstream transient failures (NPCI, Protean, state APIs) causing brief error-rate spikes",
            "Pod restarts or rolling deployments triggering short-lived health check failures",
            "Misconfigured alert thresholds set too close to normal operating ranges",
            "False positives during scheduled maintenance windows without alert silencing",
        ],
        "diagnostic": [
            "Check the subject line: `[FIRING:1]` indicates 1 active alert; `[RESOLVED]` means it self-cleared.",
            "Look for a matching `[RESOLVED]` ticket with the same fingerprint hash — if present, the incident is over.",
            "Open the Alertmanager dashboard and search by fingerprint to see the full alert history.",
            "Check the affected service's error rate and latency graphs in Grafana for the alert time window.",
            "If `[FIRING]` and no `[RESOLVED]` after 30 minutes, escalate to the on-call engineering team.",
        ],
        "resolution": [
            "For `[RESOLVED]` alerts: close the ticket immediately with tag `auto-alert-resolved`.",
            "For `[FIRING]` alerts that self-resolve (matching RESOLVED ticket exists): close both, no action needed.",
            "For persistent `[FIRING]` alerts (>30 min, no RESOLVED): create an incident, page on-call engineer.",
            "Engineering action: add or increase `for: 5m` in the alerting rule to suppress flapping.",
            "Engineering action: configure Alertmanager to silence alerts during known maintenance windows.",
        ],
        "escalation": (
            "**L1:** Close RESOLVED alerts immediately. Tag FIRING alerts and monitor for 30 minutes.\n"
            "**L2:** If FIRING persists >30 min with no RESOLVED, create P2 incident and page on-call.\n"
            "**Engineering:** Tune alert `for:` duration; review threshold against p95 baseline."
        ),
        "lender_what": (
            "This is an automated system notification — not a ticket raised by your team. "
            "The ULI platform's monitoring system sends automated `[FIRING]` and `[RESOLVED]` "
            "alerts to the support queue when it detects service anomalies. These typically "
            "self-resolve within minutes."
        ),
        "lender_resolve": [
            "No action is required from you. These alerts are handled by the RBIH engineering team.",
            "If you received a notification in error, please disregard it.",
            "If you are experiencing an actual API issue, raise a separate support ticket describing the problem.",
        ],
        "lender_escalate": (
            "If your integration is actively failing at the same time as a FIRING alert, "
            "raise a new ticket with your trace ID, client ID, and the time of the failure."
        ),
    },

    "penny_drop": {
        "title": "Penny Drop / Bank Account Verification — Failures & Alerts",
        "summary": (
            "Tickets in this cluster relate to failures on the `penny-drop-service` "
            "(`GET /verifybankaccount/{version}/{lang}`). Issues range from upstream NPCI "
            "latency and pod-level degradation to lender-side misuse of API parameters."
        ),
        "root_causes": [
            "NPCI / bank API upstream latency or partial unavailability",
            "Invalid or deprecated `{version}` path parameter passed by the lender",
            "Invalid bank account number or IFSC code in the request payload",
            "OAuth token expired or lacking required scope for this endpoint",
            "Pod-level resource exhaustion (connection pool, memory) on `penny-drop-service`",
            "Alert flapping from response-time threshold breaches during peak load",
        ],
        "diagnostic": [
            "Check if the ticket is an automated Alertmanager alert (subject contains `[FIRING]` or `[RESOLVED]`) — if so, follow the alert noise runbook.",
            "Ask the lender for: trace ID, client/org ID, exact request payload (masked), and timestamp.",
            "Check `penny-drop-service` pod logs for the trace ID — look for upstream timeout or connection errors.",
            "Verify the `{version}` parameter matches the current supported version in the API catalogue.",
            "Check the NPCI status page and internal NPCI circuit-breaker metrics for the time of the failure.",
            "Review Grafana: `penny-drop-service` error rate, p95 latency, and pod restart count.",
        ],
        "resolution": [
            "If upstream NPCI issue: inform the lender of the outage window; no lender-side action needed.",
            "If invalid `{version}`: share the current supported version from the API catalogue with the lender.",
            "If invalid account/IFSC: ask the lender to validate inputs before retrying.",
            "If token issue: direct the lender to re-authenticate and refresh their token.",
            "If pod-level issue: escalate to engineering for pod restart or scaling.",
            "If alert noise: close the ticket per the alert noise runbook.",
        ],
        "escalation": (
            "**L1:** Collect trace ID and initial triage.\n"
            "**L2:** Log investigation, check pod logs, check NPCI status.\n"
            "**Engineering:** Persistent upstream issues, pod restarts, or threshold tuning."
        ),
        "lender_what": (
            "You are seeing an error or degraded response from the ULI Penny Drop API "
            "(`GET /verifybankaccount/{version}/{lang}`), which is used to verify bank account "
            "details by initiating a small credit transaction."
        ),
        "lender_resolve": [
            "Verify the API version in your request URL matches the current version in your onboarding documentation.",
            "Confirm the bank account number and IFSC code are correct and active.",
            "Check that your OAuth token is valid and has not expired — re-authenticate if needed.",
            "Retry with exponential backoff (wait 5s, 15s, 30s before raising a ticket).",
            "If retries fail consistently, collect the trace ID from the error response and raise a support ticket.",
        ],
        "lender_escalate": (
            "Raise a support ticket if failures persist for more than 15 minutes. Include: "
            "your client/org ID, the trace ID from the error response, the time range of failures, "
            "and a sample masked request (remove account numbers before sharing)."
        ),
    },

    "pan_verify": {
        "title": "PAN Verification Failures — pan-protean-service",
        "summary": (
            "This cluster covers failures and alerts on the `pan-protean-service`, which validates "
            "PAN cards via the Protean eGov Technologies backend. Issues include Protean API "
            "downtime, rate limiting, invalid PAN format, and Alertmanager warning/critical alerts."
        ),
        "root_causes": [
            "Protean eGov Technologies API degradation or planned maintenance",
            "Rate limiting by Protean — too many requests per minute from the ULI gateway",
            "Invalid PAN format passed by lender (not matching `[A-Z]{5}[0-9]{4}[A-Z]`)",
            "OAuth token expired or scope insufficient for PAN verification endpoint",
            "Alert flapping on warning/critical thresholds due to Protean latency spikes",
            "Environment misconfiguration (sandbox vs production Protean endpoint)",
        ],
        "diagnostic": [
            "Check if the ticket is an Alertmanager alert — if so, follow the alert noise runbook.",
            "Ask lender for: trace ID, PAN (masked, e.g. ABCXX1234X), timestamp, and error code.",
            "Check `pan-protean-service` logs for the trace ID — look for upstream HTTP status from Protean.",
            "Check internal Protean circuit-breaker status and error rate dashboard.",
            "Validate the PAN format (5 uppercase letters, 4 digits, 1 uppercase letter).",
            "Confirm the lender is calling the correct environment endpoint (sandbox vs production).",
        ],
        "resolution": [
            "If Protean upstream issue: inform the lender; monitor Protean status page for resolution.",
            "If rate limited: engineering to review and adjust rate limit config for the lender's org.",
            "If invalid PAN format: share the correct format specification with the lender.",
            "If token issue: direct the lender to re-authenticate.",
            "If environment mismatch: share the correct production endpoint from the API catalogue.",
        ],
        "escalation": (
            "**L1:** Triage — collect trace ID, confirm alert vs real failure.\n"
            "**L2:** Check Protean status, pod logs, circuit-breaker metrics.\n"
            "**Engineering:** Protean SLA breach, rate limit adjustment, threshold tuning."
        ),
        "lender_what": (
            "The ULI PAN Verification API (`pan-protean-service`) is returning an error or slow "
            "response. This service validates PAN card details via the Protean eGov backend."
        ),
        "lender_resolve": [
            "Confirm the PAN follows the correct format: 5 uppercase letters, 4 digits, 1 uppercase letter (e.g. ABCDE1234F).",
            "Ensure you are calling the production endpoint, not the sandbox URL.",
            "Check that your OAuth token is valid — re-authenticate if it has expired.",
            "Retry with exponential backoff (5s, 15s, 30s) before raising a ticket.",
        ],
        "lender_escalate": (
            "Raise a ticket if failures persist for more than 15 minutes. Include: client/org ID, "
            "masked PAN (replace middle digits, e.g. ABCXX1234X), trace ID, and error code."
        ),
    },

    "pfx_cert": {
        "title": "PFX Certificate Expiry — API Authentication Failure",
        "summary": (
            "PFX (PKCS#12) certificates are used for mutual TLS authentication on several ULI "
            "platform APIs. When a certificate expires or is close to expiry, lenders receive "
            "authentication errors. Alertmanager also generates expiry warning alerts in this cluster."
        ),
        "root_causes": [
            "PFX certificate has expired — API calls are rejected with a TLS handshake error",
            "Certificate expiry warning alert from Alertmanager (typically 30-day pre-expiry)",
            "Lender installed the certificate incorrectly (wrong keystore, wrong alias)",
            "Certificate renewed but not yet deployed to all lender integration environments",
            "Mismatch between the certificate's CN/SAN and the API endpoint hostname",
        ],
        "diagnostic": [
            "Check the ticket subject for Alertmanager expiry alert patterns (e.g. `CertificateExpiringSoon`).",
            "Ask the lender for the certificate's expiry date — `openssl pkcs12 -in cert.pfx -nokeys | openssl x509 -noout -dates`.",
            "Check the ULI certificate inventory spreadsheet for the lender's registered certificate expiry.",
            "If expired: issue a new certificate immediately via the certificate management process.",
            "If warning alert: confirm the lender has initiated the renewal process; set a follow-up reminder.",
        ],
        "resolution": [
            "Generate a new PFX certificate for the lender via the standard certificate issuance process.",
            "Share the new certificate and installation guide with the lender's technical contact.",
            "Confirm the lender has successfully installed and tested the new certificate.",
            "Update the certificate expiry date in the inventory spreadsheet.",
            "Engineering: add the certificate to the automated expiry scanning job with 60-day pre-alert.",
        ],
        "escalation": (
            "**L1:** Confirm expiry, notify lender, initiate certificate issuance.\n"
            "**L2:** Oversee certificate delivery and lender installation.\n"
            "**Engineering:** Automate certificate expiry scanning and proactive renewal workflow."
        ),
        "lender_what": (
            "Your PFX (PKCS#12) certificate used for API authentication with the ULI platform "
            "has expired or is about to expire. API calls will be rejected until a valid certificate "
            "is installed."
        ),
        "lender_resolve": [
            "Check your certificate's expiry date: open the PFX file in your keystore tool or run `openssl pkcs12 -in cert.pfx -nokeys | openssl x509 -noout -dates`.",
            "If expired: contact ULI support immediately to request a new certificate.",
            "If expiring soon: request a renewal at least 7 days before the expiry date.",
            "Once you receive the new certificate, install it in your integration environment and run a test API call to confirm.",
        ],
        "lender_escalate": (
            "Contact support immediately if your certificate has already expired and you are unable "
            "to make API calls. Include your client/org ID and the certificate's current expiry date."
        ),
    },

    "account_agg": {
        "title": "Account Aggregator (AA) Consent Flow Failures — aaconsent Endpoint",
        "summary": (
            "This cluster covers failures on the `account-aggregator-service` "
            "(`/aaconsent/{version}/{lang}`). Issues include consent creation failures, "
            "FIP unavailability, AA handle resolution errors, and Alertmanager critical alerts."
        ),
        "root_causes": [
            "Financial Information Provider (FIP) is offline or not responding to the AA framework",
            "Invalid or malformed AA handle provided by the lender",
            "Consent request expired before the customer approved it",
            "AA ecosystem downtime (Central Registry or specific AA operator)",
            "OAuth token lacks the required scope for the AA consent endpoint",
            "Alert flapping on the aaconsent endpoint due to FIP intermittency",
        ],
        "diagnostic": [
            "Check if the ticket is an Alertmanager alert — if so, follow the alert noise runbook.",
            "Ask the lender for: AA handle, consent request ID, trace ID, and timestamp.",
            "Check `account-aggregator-service` logs for the consent request ID.",
            "Verify the FIP status in the AA Central Registry.",
            "Check whether the customer's consent window (typically 24 hours) has expired.",
        ],
        "resolution": [
            "If FIP offline: inform the lender; monitor FIP status in AA Central Registry.",
            "If invalid AA handle: share the correct AA handle format with the lender.",
            "If consent expired: ask the lender to reinitiate the consent request.",
            "If token scope issue: direct the lender to re-authenticate with correct scopes.",
            "If AA ecosystem downtime: escalate to engineering; monitor AA operator status page.",
        ],
        "escalation": (
            "**L1:** Collect consent request ID and initial triage.\n"
            "**L2:** Check FIP status, service logs, AA Central Registry.\n"
            "**Engineering:** AA ecosystem or service-level failures requiring platform intervention."
        ),
        "lender_what": (
            "The ULI Account Aggregator consent API (`/aaconsent`) is failing. This API is used "
            "to initiate and manage data-sharing consent from your customers via the AA framework."
        ),
        "lender_resolve": [
            "Verify the AA handle (VUA) is in the correct format and belongs to an active AA operator.",
            "Check that the customer has not let the consent request expire — resend if needed.",
            "Confirm your OAuth token has the correct scopes for AA consent operations.",
            "Retry the consent request after a brief wait if the failure appears transient.",
        ],
        "lender_escalate": (
            "Raise a support ticket if the issue persists. Include: your client/org ID, the "
            "consent request ID, the AA handle used (partially masked), trace ID, and error code."
        ),
    },

    "nesl": {
        "title": "NESL Integration Issues",
        "summary": (
            "Tickets in this cluster relate to the NESL (National E-Governance Services Limited) "
            "integration on the ULI platform, covering charge creation, satisfaction, and search failures."
        ),
        "root_causes": [
            "NESL portal downtime or planned maintenance",
            "Invalid or mismatched charge identifiers in the request",
            "Lender entity not registered or not active on NESL",
            "OAuth token scope insufficient for NESL operations",
            "Data format mismatch between lender payload and NESL API schema",
        ],
        "diagnostic": [
            "Ask the lender for: NESL charge ID (if applicable), trace ID, operation type (create/satisfy/search), and timestamp.",
            "Check NESL portal status for any ongoing incidents.",
            "Validate the lender's entity registration status on NESL.",
            "Review `nesl-service` logs for the trace ID.",
        ],
        "resolution": [
            "If NESL portal issue: inform the lender of the outage; retry once NESL is restored.",
            "If invalid charge ID: ask the lender to verify the charge identifier against the NESL portal.",
            "If entity not registered: escalate to onboarding team to complete NESL registration.",
            "If data format issue: share the correct NESL API schema from the integration guide.",
        ],
        "escalation": (
            "**L1/L2:** Collect trace ID, check NESL status, validate entity registration.\n"
            "**Engineering:** Platform-level NESL integration failures or schema changes."
        ),
        "lender_what": (
            "Your request to the ULI NESL integration API has failed. This API handles charge "
            "registration, satisfaction, and search operations with the National E-Governance "
            "Services portal."
        ),
        "lender_resolve": [
            "Confirm your entity is registered and active on the NESL portal.",
            "Verify the charge ID and other identifiers match what is registered on NESL.",
            "Ensure your request payload matches the NESL API schema in your integration documentation.",
            "Retry after confirming the above — NESL portal outages are typically short-lived.",
        ],
        "lender_escalate": (
            "Raise a ticket if the issue persists. Include: client/org ID, NESL charge ID, "
            "operation type, trace ID, and the exact error message."
        ),
    },

    "cibil": {
        "title": "CIBIL / Credit Bureau Report Fetch Failures",
        "summary": (
            "Tickets in this cluster relate to failures fetching CIBIL or other credit bureau "
            "reports via the ULI platform. Issues include bureau API unavailability, consent "
            "validation failures, and identity mismatch errors."
        ),
        "root_causes": [
            "CIBIL / bureau API downtime or rate limiting",
            "Customer consent not obtained or expired before the fetch",
            "Identity details (name, PAN, DOB) provided do not match bureau records",
            "OAuth token expired or lacking the bureau data fetch scope",
            "Invalid report type or bureau code in the request",
        ],
        "diagnostic": [
            "Ask the lender for: consent ID, masked PAN/customer identifier, trace ID, and bureau code.",
            "Check bureau API status and internal circuit-breaker metrics.",
            "Verify the consent record is valid and has not expired.",
            "Check if identity fields match the expected format.",
        ],
        "resolution": [
            "If bureau downtime: inform the lender; retry once bureau API recovers.",
            "If consent issue: lender must re-obtain customer consent.",
            "If identity mismatch: lender to verify input details against their KYC records.",
            "If token issue: direct the lender to re-authenticate.",
        ],
        "escalation": (
            "**L1/L2:** Collect trace ID, verify consent and bureau status.\n"
            "**Engineering:** Persistent bureau API failures or schema changes."
        ),
        "lender_what": (
            "Your request to fetch a credit bureau report via the ULI platform has failed. "
            "This may be due to bureau API unavailability, an expired consent, or a mismatch "
            "in the customer's identity details."
        ),
        "lender_resolve": [
            "Confirm the customer's consent is current and has not expired.",
            "Verify that the identity details (PAN, name, DOB) match your KYC records exactly.",
            "Ensure your OAuth token is valid and has the bureau data fetch scope.",
            "Retry once — transient bureau failures typically resolve within minutes.",
        ],
        "lender_escalate": (
            "Raise a ticket if failures persist. Include: consent ID, masked customer identifier, "
            "bureau code, trace ID, and error message."
        ),
    },

    "ekcc_land": {
        "title": "eKYC / Land Record Data Fetch Issues",
        "summary": (
            "Tickets in this cluster relate to incomplete or incorrect data returned by the "
            "ULI eKCC / land record integration, including missing Survey Numbers, Khata "
            "mismatches, and state portal unavailability."
        ),
        "root_causes": [
            "State land record portal (e.g. AP, Telangana) is offline or undergoing maintenance",
            "One-to-many Khata-to-Survey-Number mapping not handled by the ULI integration layer",
            "State portal API schema change not yet reflected in ULI's parser",
            "Requested Khata/Khatauni does not exist in the state database",
            "Latency or partial response from state portal integration",
        ],
        "diagnostic": [
            "Ask the lender for: state code, Khata number, district/mandal, trace ID, and timestamp.",
            "Check state portal status for the relevant state.",
            "Review `ekcc-service` logs for the trace ID — look for partial responses or timeouts.",
            "Verify the Khata number against the state portal directly if accessible.",
        ],
        "resolution": [
            "If state portal offline: inform the lender; retry once the portal is restored.",
            "If missing Survey Numbers: escalate to engineering to review one-to-many mapping logic.",
            "If schema change: engineering to update the ULI parser for the new state API format.",
            "If Khata not found: ask the lender to verify the Khata number directly with the state portal.",
        ],
        "escalation": (
            "**L1/L2:** Collect Khata details, check state portal status.\n"
            "**Engineering:** Missing records, schema changes, or integration bugs."
        ),
        "lender_what": (
            "Your request to fetch land record data via the ULI platform returned incomplete "
            "or incorrect information. This service integrates with state land record portals "
            "which may be intermittently unavailable."
        ),
        "lender_resolve": [
            "Verify the Khata/Khatauni number directly on the relevant state land portal.",
            "If the state portal is offline, retry after 30–60 minutes.",
            "If data is consistently incomplete, raise a support ticket with the specific Khata number and state.",
        ],
        "lender_escalate": (
            "Raise a ticket with: state code, district, Khata number, trace ID, and a description "
            "of what data is missing or incorrect."
        ),
    },

    "vehicle_hyp": {
        "title": "Vehicle Hypothecation Details API — Partial or Missing Response",
        "summary": (
            "Tickets in this cluster relate to the Vehicle Hypothecation Details API returning "
            "incomplete responses — typically metadata without the encrypted data payload and "
            "security tag, or VAHAN lookup failures."
        ),
        "root_causes": [
            "Encryption/security module not triggered — encrypted payload is missing",
            "OAuth token lacks required scope for the encrypted response tier",
            "Lender calling a staging/sandbox endpoint that returns mock/masked data only",
            "API version mismatch — deprecated version does not support encrypted payload",
            "VAHAN portal unavailability causing lookup failures",
            "Gateway response truncation due to payload size policy",
        ],
        "diagnostic": [
            "Ask the lender for: vehicle registration number (masked), trace ID, API version used, and environment (prod/sandbox).",
            "Confirm the lender is calling the production endpoint, not sandbox.",
            "Check `vehicle-hypothecation-service` logs for the trace ID.",
            "Verify the API version in the request matches the current supported version.",
            "Check VAHAN portal status for any ongoing outages.",
        ],
        "resolution": [
            "If sandbox endpoint: share the correct production endpoint with the lender.",
            "If API version: share the current supported version and migration guide.",
            "If token scope: direct the lender to request the correct OAuth scopes.",
            "If VAHAN offline: inform the lender; retry once VAHAN is restored.",
            "If encryption module issue: escalate to engineering.",
        ],
        "escalation": (
            "**L1/L2:** Triage environment and API version.\n"
            "**Engineering:** Encryption module failures, gateway truncation, or VAHAN integration bugs."
        ),
        "lender_what": (
            "The ULI Vehicle Hypothecation Details API is returning an incomplete response — "
            "you may be receiving only metadata without the encrypted data payload, or the "
            "VAHAN lookup may be failing."
        ),
        "lender_resolve": [
            "Confirm you are calling the production endpoint (not sandbox).",
            "Verify the API version in your request URL matches the latest supported version in your documentation.",
            "Ensure your OAuth token has the required scopes — re-authenticate if needed.",
            "Provide the correct vehicle registration number format (state code + series + number).",
        ],
        "lender_escalate": (
            "Raise a ticket with: client/org ID, masked vehicle registration number, API version used, "
            "environment (prod/sandbox), trace ID, and a description of what is missing in the response."
        ),
    },

    "ifsc": {
        "title": "IFSC Lookup Failures",
        "summary": (
            "Tickets in this cluster relate to IFSC code lookup failures or incorrect branch "
            "details returned by the ULI IFSC service."
        ),
        "root_causes": [
            "IFSC code not found in the ULI IFSC master database (recently issued or decommissioned branch)",
            "Invalid IFSC format passed by the lender",
            "RBI IFSC master database not yet updated with recent branch changes",
            "Typo or encoding error in the IFSC code",
        ],
        "diagnostic": [
            "Ask the lender for: the exact IFSC code and the bank/branch they expect it to resolve to.",
            "Check the IFSC format: 4 uppercase letters + '0' + 6 alphanumeric characters.",
            "Look up the IFSC code in the RBI IFSC master database.",
            "Check when the ULI IFSC database was last refreshed.",
        ],
        "resolution": [
            "If invalid format: share the correct IFSC format with the lender.",
            "If not in database: verify against the RBI master; if valid, trigger an IFSC database refresh.",
            "If decommissioned: inform the lender of the branch merger/closure and provide the replacement IFSC.",
        ],
        "escalation": (
            "**L1/L2:** Validate IFSC format and RBI master lookup.\n"
            "**Engineering:** IFSC database refresh or integration with RBI master update feed."
        ),
        "lender_what": (
            "The ULI IFSC lookup API could not find or return details for the IFSC code you provided."
        ),
        "lender_resolve": [
            "Verify the IFSC code format: 4 uppercase letters, then '0', then 6 alphanumeric characters (e.g. HDFC0001234).",
            "Cross-check the IFSC code against the bank's official website or the RBI IFSC finder.",
            "Confirm the branch has not been merged or closed recently.",
        ],
        "lender_escalate": (
            "Raise a ticket if the IFSC code is valid per RBI records but the ULI lookup fails. "
            "Include the exact IFSC code and the bank/branch details."
        ),
    },

    "onboarding": {
        "title": "Lender Onboarding — Registration, KYC & Activation Issues",
        "summary": (
            "This cluster covers tickets from lenders in the onboarding process — account "
            "registration, KYC completion, environment access, certificate issuance, and "
            "initial API access setup."
        ),
        "root_causes": [
            "KYC documents not yet verified by the RBIH onboarding team",
            "Sandbox/production credentials not yet provisioned after registration",
            "PFX certificate not issued or not correctly installed",
            "Lender trying to access production before completing sandbox testing sign-off",
            "Missing or incorrect entity details in the onboarding form",
        ],
        "diagnostic": [
            "Check the lender's onboarding status in the RBIH onboarding tracker.",
            "Verify KYC document submission is complete.",
            "Check if sandbox credentials have been issued and shared.",
            "Confirm whether the lender has completed and passed sandbox testing.",
        ],
        "resolution": [
            "If KYC pending: notify the onboarding team; share expected timeline with the lender.",
            "If credentials not issued: raise an internal request to provision sandbox access.",
            "If PFX not issued: initiate certificate generation and share with the lender's technical contact.",
            "If production access requested without sandbox sign-off: complete sandbox testing first.",
        ],
        "escalation": (
            "**L1:** Triage onboarding stage and missing steps.\n"
            "**Onboarding team:** KYC verification, credential provisioning, production go-live approval."
        ),
        "lender_what": (
            "Your onboarding to the ULI platform is in progress. You may be waiting for KYC "
            "verification, credential provisioning, or production access approval."
        ),
        "lender_resolve": [
            "Check your email for communications from the RBIH onboarding team.",
            "Ensure all required KYC documents have been submitted via the onboarding portal.",
            "Complete sandbox integration and testing before requesting production access.",
            "If you have not received your sandbox credentials within 3 business days of document submission, raise a support ticket.",
        ],
        "lender_escalate": (
            "Contact support with: your organisation name, LEI/CIN, onboarding request reference number, "
            "and the specific step where you are blocked."
        ),
    },

    "auth": {
        "title": "Authentication & Authorisation Failures — Token, OAuth & API Key Issues",
        "summary": (
            "This cluster covers HTTP 401/403 errors, expired or invalid OAuth tokens, "
            "incorrect API key usage, and scope/permission mismatches on ULI platform APIs."
        ),
        "root_causes": [
            "OAuth access token expired — typically 1-hour validity",
            "Token issued for sandbox being used against production endpoint or vice versa",
            "Insufficient OAuth scopes for the requested API operation",
            "API key rotated but old key still in use by the lender",
            "Client ID / client secret mismatch in token generation request",
            "IP allowlist not updated after lender's outbound IP change",
        ],
        "diagnostic": [
            "Ask the lender for: the HTTP status code (401 vs 403), error message, trace ID, and endpoint called.",
            "Check if the token is expired: decode the JWT and check the `exp` claim.",
            "Verify the client ID matches the lender's registered entity in the ULI admin portal.",
            "Check the token's scope list against the required scopes for the endpoint.",
            "Confirm the lender's outbound IP is on the allowlist.",
        ],
        "resolution": [
            "If token expired: direct the lender to refresh their token using the client credentials flow.",
            "If scope mismatch: update the lender's OAuth client scope in the admin portal.",
            "If wrong environment token: direct the lender to use the correct token endpoint.",
            "If API key rotation: share the new API key securely with the lender's technical contact.",
            "If IP allowlist: update the allowlist with the lender's new outbound IP.",
        ],
        "escalation": (
            "**L1:** Collect HTTP status, error message, and token details.\n"
            "**L2:** Admin portal scope and IP allowlist changes.\n"
            "**Engineering:** OAuth server issues or systemic auth failures."
        ),
        "lender_what": (
            "Your API request to the ULI platform is being rejected with a 401 (Unauthorized) "
            "or 403 (Forbidden) error. This is typically caused by an expired token, incorrect "
            "credentials, or insufficient permissions."
        ),
        "lender_resolve": [
            "Re-authenticate using the client credentials OAuth flow to obtain a fresh access token.",
            "Confirm you are using the correct token endpoint for your environment (sandbox vs production).",
            "Verify your client ID and client secret are correct and have not been rotated.",
            "Check that your access token includes the required scopes for the API you are calling.",
            "If your outbound IP has changed, notify support to update the IP allowlist.",
        ],
        "lender_escalate": (
            "Raise a ticket if you continue to receive 401/403 errors after refreshing your token. "
            "Include: client/org ID, endpoint called, HTTP status code, error message, and trace ID."
        ),
    },

    "performance": {
        "title": "Performance & Latency — Slow Responses & Timeouts",
        "summary": (
            "This cluster covers tickets related to elevated response times, SLA breaches, "
            "and timeout errors on ULI platform APIs. May include both genuine performance "
            "incidents and Alertmanager latency/error-rate alerts."
        ),
        "root_causes": [
            "Upstream dependency latency (NPCI, Protean, bureau APIs, state portals)",
            "Pod-level resource exhaustion (CPU throttling, connection pool saturation)",
            "Database query slowness or lock contention under high load",
            "Alert threshold set too close to normal p95 latency — triggering false positives",
            "Network/ingress layer issues between client and ULI gateway",
            "Concurrent load spike from multiple lenders during peak hours",
        ],
        "diagnostic": [
            "Check if the ticket is an Alertmanager alert — if so, check for a corresponding RESOLVED alert.",
            "Ask the lender for: trace ID / CID, endpoint called, response time observed, and timestamp.",
            "Check Grafana: p95/p99 latency and error rate for the affected service during the time window.",
            "Check upstream dependency status (NPCI, Protean, bureau, state portals).",
            "Review pod CPU/memory metrics and connection pool utilisation.",
        ],
        "resolution": [
            "If upstream latency: inform the lender; no platform-side action unless persistent.",
            "If pod resource issue: engineering to scale pods or adjust resource limits.",
            "If alert threshold misconfiguration: engineering to recalibrate based on baseline p95.",
            "If network issue: engineering to review ingress configuration.",
            "Recommend lender implement client-side timeout of 30s with 3 retries and exponential backoff.",
        ],
        "escalation": (
            "**L1:** Collect trace ID and initial triage.\n"
            "**L2:** Check Grafana, upstream status, pod metrics.\n"
            "**Engineering:** Scaling, threshold tuning, or infrastructure-level intervention."
        ),
        "lender_what": (
            "You are experiencing slow responses or timeout errors from the ULI platform. "
            "This may be caused by high load on the platform or a dependency (e.g. NPCI, Protean) "
            "being slow."
        ),
        "lender_resolve": [
            "Implement a client-side timeout of at least 30 seconds for ULI API calls.",
            "Retry failed requests with exponential backoff: wait 5s, 15s, then 30s between retries.",
            "Avoid sending bursts of concurrent requests — introduce a small delay between calls if batching.",
            "If timeouts persist for more than 10 minutes, raise a support ticket.",
        ],
        "lender_escalate": (
            "Raise a ticket with: client/org ID, affected endpoint, observed response time, "
            "trace ID (from error response), and the time range of the issue."
        ),
    },

    "email_noise": {
        "title": "Email-Originated Noise — Auto-Replies, OOO & Misdirected Mail",
        "summary": (
            "A significant portion of the support queue consists of tickets auto-generated "
            "from email: out-of-office (OOO) replies, auto-acknowledgements, misdirected "
            "notifications, and forwarded alerts. These require no technical action."
        ),
        "root_causes": [
            "Lender or partner email systems sending OOO/auto-reply to the support mailbox",
            "Alertmanager notification emails forwarded to the support address",
            "Internal team communications accidentally CC'ing the support email address",
            "Freshdesk not filtering RFC 3834 auto-reply headers",
            "Bulk notification emails generating one ticket per recipient reply",
        ],
        "diagnostic": [
            "Check the ticket subject for OOO patterns: 'Out of Office', 'Auto-Reply', 'Automatic Reply'.",
            "Check the ticket subject for alert patterns: '[FIRING]', '[RESOLVED]', 'Alertmanager'.",
            "Check if the ticket body contains substantive technical content — if not, it is noise.",
            "Check the 'From' address — internal team addresses or monitoring systems indicate noise.",
        ],
        "resolution": [
            "Close the ticket immediately with tag `email-noise` and status `closed`.",
            "Do not reply — replying to OOO addresses can trigger further auto-replies.",
            "If a lender's domain is consistently generating noise, add their OOO address to the block list.",
            "Engineering: configure Freshdesk email gateway to filter `Auto-Submitted: auto-replied` headers (RFC 3834).",
        ],
        "escalation": (
            "**L1:** Close immediately — no escalation needed.\n"
            "**Engineering:** Freshdesk gateway RFC 3834 header filtering to prevent recurrence."
        ),
        "lender_what": (
            "This article is for internal use. Lenders are not expected to raise tickets about "
            "email noise. If you received this article in error, please raise a new ticket "
            "describing your actual issue."
        ),
        "lender_resolve": [
            "Please raise a new support ticket describing your specific technical issue.",
            "Avoid forwarding automated notifications or OOO replies to the support email address.",
        ],
        "lender_escalate": "Contact support with a clear description of your technical issue.",
    },

    "test_junk": {
        "title": "Test & Junk Ticket Submissions",
        "summary": (
            "This cluster covers tickets submitted by agents or lenders for testing purposes, "
            "dummy submissions, and requests for test data provisioning in the sandbox environment."
        ),
        "root_causes": [
            "Agents testing the ticketing system by submitting dummy tickets",
            "Lenders testing their integration with the support portal",
            "Requests for test PAN numbers, test bank accounts, or sandbox credentials",
            "Training/onboarding sessions generating test tickets",
        ],
        "diagnostic": [
            "Check subject line for keywords: 'test', 'testing', 'dummy', 'ignore', 'junk'.",
            "Check if the ticket body contains placeholder text or obviously fake data.",
            "Check if the submitter is an internal agent or a known test account.",
        ],
        "resolution": [
            "Close the ticket immediately with tag `test-junk`.",
            "For test data requests (test PAN, bank account): share the sandbox test data catalogue with the requester.",
            "Remind agents to use the dedicated test queue for internal testing — not the live support queue.",
        ],
        "escalation": "**L1:** Close immediately — no escalation needed.",
        "lender_what": (
            "If you need test credentials or test data for sandbox integration, please request "
            "them via the onboarding portal rather than raising a support ticket."
        ),
        "lender_resolve": [
            "Use the test data catalogue provided during onboarding for sandbox testing.",
            "For additional sandbox test accounts or data, contact your RBIH onboarding manager.",
        ],
        "lender_escalate": "Contact your onboarding manager for sandbox test data requests.",
    },

    "data_mismatch": {
        "title": "Data Mismatch & Discrepancy Issues",
        "summary": (
            "Tickets in this cluster report incorrect, inconsistent, or mismatched data "
            "returned by ULI platform APIs — values that do not match the lender's source "
            "data or expected API contract."
        ),
        "root_causes": [
            "Source data not yet updated in the upstream provider's database",
            "API response schema change not communicated to the lender",
            "Field mapping bug in the ULI integration layer",
            "Lender consuming a cached/stale response",
            "Timezone or date format mismatch between lender and API",
        ],
        "diagnostic": [
            "Ask the lender for: the specific field with incorrect data, expected value vs actual value, trace ID, and timestamp.",
            "Check the upstream data source for the same record.",
            "Check if the discrepancy is reproducible or intermittent.",
            "Review the ULI API response schema changelog for recent updates.",
        ],
        "resolution": [
            "If upstream data lag: inform the lender; the data should sync within the provider's SLA.",
            "If schema change: share the updated API schema and changelog with the lender.",
            "If integration bug: escalate to engineering with the trace ID and example payload.",
            "If caching: engineering to review cache TTL for the affected endpoint.",
        ],
        "escalation": (
            "**L1/L2:** Collect field details, trace ID, and example payloads.\n"
            "**Engineering:** Field mapping bugs, caching issues, or upstream data pipeline problems."
        ),
        "lender_what": (
            "The data returned by a ULI platform API does not match what you expected — "
            "values may be incorrect, missing, or inconsistent with your source records."
        ),
        "lender_resolve": [
            "Verify the discrepancy is not due to a recent update in your own source data.",
            "Check the ULI API changelog for any recent schema or field changes.",
            "If the issue is reproducible, capture the full API response and raise a support ticket.",
        ],
        "lender_escalate": (
            "Raise a ticket with: client/org ID, affected API endpoint, specific field name, "
            "expected value, actual value received, trace ID, and timestamp."
        ),
    },

    "config": {
        "title": "Configuration & Environment Issues",
        "summary": (
            "Tickets in this cluster relate to misconfigured integration parameters — wrong "
            "environment endpoints, incorrect API versions, parameter errors, and environment "
            "variable mismatches."
        ),
        "root_causes": [
            "Lender using sandbox endpoint in production or vice versa",
            "Incorrect API version in the request URL",
            "Missing or incorrect request headers (Content-Type, Accept, client tokens)",
            "Environment variable pointing to wrong base URL after a deployment",
            "Incorrect TLS/SSL configuration on the lender's HTTP client",
        ],
        "diagnostic": [
            "Ask the lender for: the exact request URL, headers, and error response.",
            "Confirm the lender's environment (sandbox vs production) and cross-check the endpoint.",
            "Verify the API version against the current supported version in the catalogue.",
            "Check required headers against the API specification.",
        ],
        "resolution": [
            "Share the correct endpoint, API version, and required headers from the integration guide.",
            "Confirm the lender has updated all relevant environment variables.",
            "Ask the lender to run a test call after the fix and share the trace ID.",
        ],
        "escalation": (
            "**L1/L2:** Share correct configuration details from integration guide.\n"
            "**Engineering:** If the integration guide is incorrect or out of date."
        ),
        "lender_what": (
            "Your integration with the ULI platform appears to be misconfigured — you may be "
            "using the wrong endpoint, API version, or request headers."
        ),
        "lender_resolve": [
            "Double-check your base URL: sandbox and production URLs are different.",
            "Verify the API version in your request URL matches your integration documentation.",
            "Ensure all required headers are present: Content-Type, Authorization, and any ULI-specific headers.",
            "Review the integration guide shared during onboarding for the correct configuration.",
        ],
        "lender_escalate": (
            "Raise a ticket with: client/org ID, full request URL (mask any secrets), "
            "request headers, HTTP status code, error message, and trace ID."
        ),
    },

    "general": {
        "title": "General ULI Platform Support",
        "summary": (
            "This cluster covers miscellaneous support requests that do not fall into a "
            "specific category — general enquiries, follow-ups, documentation requests, "
            "and mixed-topic tickets."
        ),
        "root_causes": [
            "General integration questions not covered by existing documentation",
            "Follow-up on previously raised tickets",
            "Documentation or API reference requests",
            "Feedback or feature requests routed to the support queue",
        ],
        "diagnostic": [
            "Read the ticket carefully to identify the specific issue or question.",
            "Check existing KB articles and documentation before responding.",
            "If the issue matches another KB article category, re-tag and reassign accordingly.",
        ],
        "resolution": [
            "Respond with the relevant documentation link or KB article.",
            "If a feature request: log in the product backlog tracker and acknowledge to the lender.",
            "If a follow-up on a previous ticket: link the tickets and continue investigation.",
        ],
        "escalation": (
            "**L1:** Handle with documentation or KB reference.\n"
            "**L2/Product:** Feature requests or systemic issues requiring product input."
        ),
        "lender_what": "Please describe your issue in detail so we can assist you.",
        "lender_resolve": [
            "Check the ULI integration guide and API reference documentation.",
            "Search the knowledge base for articles related to your issue.",
            "If you cannot find an answer, raise a support ticket with full details.",
        ],
        "lender_escalate": (
            "Raise a support ticket with a clear description of your issue, your client/org ID, "
            "and any relevant trace IDs or error messages."
        ),
    },
}


# ---------------------------------------------------------------------------
# Article renderers
# ---------------------------------------------------------------------------

def _render_internal(
    title: str,
    tmpl: dict,
    top_terms: str,
    size: int,
    subjects: list[str],
    cluster_id: int,
) -> str:
    root_causes = "\n".join(f"- {rc}" for rc in tmpl["root_causes"])
    diagnostic  = "\n".join(f"{i}. {s}" for i, s in enumerate(tmpl["diagnostic"], 1))
    resolution  = "\n".join(f"{i}. {s}" for i, s in enumerate(tmpl["resolution"], 1))
    subject_list = "\n".join(f"- {s}" for s in subjects[:_MAX_SAMPLE]) if subjects else "_No sample subjects available._"

    return textwrap.dedent(f"""\
        # {title}

        **Cluster:** {cluster_id} | **Volume:** {size} tickets | **Top terms:** {top_terms}

        ---

        ## Issue Summary

        {tmpl['summary']}

        ---

        ## Common Root Causes

        {root_causes}

        ---

        ## Diagnostic Steps

        {diagnostic}

        ---

        ## Resolution

        {resolution}

        ---

        ## Escalation Path

        {tmpl['escalation']}

        ---

        ## Sample Ticket Subjects

        {subject_list}
    """)


def _render_lender(
    title: str,
    tmpl: dict,
    size: int,
) -> str:
    resolve = (
        "\n".join(f"{i}. {s}" for i, s in enumerate(tmpl["lender_resolve"], 1))
        if isinstance(tmpl["lender_resolve"], list)
        else tmpl["lender_resolve"]
    )
    return textwrap.dedent(f"""\
        # {title}

        ## What is this issue?

        {tmpl['lender_what']}

        ---

        ## How to resolve it

        {resolve}

        ---

        ## When to contact support

        {tmpl['lender_escalate']}
    """)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_subjects(df: pd.DataFrame, cluster_id: int, n: int = _MAX_SAMPLE) -> list[str]:
    cluster_df = df[df["cluster"] == cluster_id]
    subjects = cluster_df["subject"].dropna().str.strip()
    subjects = subjects[subjects != ""].head(n).tolist()
    return subjects


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_kb_articles(
    df: pd.DataFrame,
    summaries: pd.DataFrame,
    client: anthropic.Anthropic,  # noqa: ARG001 — accepted for API compat, not used
) -> list[dict]:
    """
    Generate internal and lender-facing KB articles for every cluster.

    Uses a template engine keyed on the cluster's inferred topic — zero
    Claude API calls. The `client` parameter is accepted for API compatibility
    with analyze_tickets.py but is never invoked.

    Args:
        df: DataFrame with 'cluster' column (from clusterer.cluster_tickets()).
        summaries: DataFrame with columns cluster, size, top_terms.
        client: Unused — kept for signature compatibility.

    Returns:
        List of dicts, one per cluster:
            {cluster, size, top_terms, internal_path, lender_path}
    """
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Generating KB articles..."),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} clusters"),
    ) as progress:
        task = progress.add_task("generating", total=len(summaries))

        for _, summary_row in summaries.iterrows():
            cluster_id = int(summary_row["cluster"])
            top_terms  = str(summary_row["top_terms"])
            size       = int(summary_row["size"])

            subjects = _sample_subjects(df, cluster_id)
            topic    = _infer_topic(top_terms, subjects)
            tmpl     = _TEMPLATES.get(topic, _TEMPLATES["general"])
            title    = tmpl["title"]

            internal_article = _render_internal(title, tmpl, top_terms, size, subjects, cluster_id)
            lender_article   = _render_lender(title, tmpl, size)

            internal_path = os.path.join(_OUTPUT_DIRS["internal"], f"cluster_{cluster_id}.md")
            lender_path   = os.path.join(_OUTPUT_DIRS["lender"],   f"cluster_{cluster_id}.md")

            os.makedirs(_OUTPUT_DIRS["internal"], exist_ok=True)
            os.makedirs(_OUTPUT_DIRS["lender"],   exist_ok=True)

            with open(internal_path, "w", encoding="utf-8") as fh:
                fh.write(internal_article)
            with open(lender_path, "w", encoding="utf-8") as fh:
                fh.write(lender_article)

            results.append({
                "cluster":       cluster_id,
                "size":          size,
                "top_terms":     top_terms,
                "internal_path": internal_path,
                "lender_path":   lender_path,
            })

            progress.advance(task)

    return results
