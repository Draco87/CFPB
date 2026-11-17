"""
MCP server: lets AI agents use the Complaint Intelligence system as TOOLS.

MCP (Model Context Protocol) is a standard way for an AI agent (Claude, an IDE assistant, your own agent) to discover
and call outside tools and read outside data. This server is a thin layer over the FastAPI service from Module 7:
every tool simply calls the API, so the models, guardrails, PII masking and request logging all stay in one place.

    agent  --MCP-->  this server  --HTTP-->  FastAPI service  -->  models

Tools (actions the agent can take)
  check_system_health   which parts are ready
  triage_complaint      which of 9 products / teams a complaint belongs to
  score_payout_risk     payout probability, risk band and top 3 reasons
  draft_reply           regulation-cited draft for a human agent, with guardrail warnings (slow, ~30 s)
  get_drift_report      month-by-month monitoring summary: drift alerts, accuracy, calibration
Resources (data the agent can read)
  cfpb://model-cards    what each model is for, NOT for, and its known limits

Run (the API must be running first):   uv run python -m cfpb.mcp_server
Settings: API_URL (default http://127.0.0.1:8000)
"""
import os
from typing import Any

import pandas as pd
import requests
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from cfpb import config

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
PRODUCTS = ["Bank account", "Credit / prepaid card", "Credit reporting", "Debt collection",
            "Money transfer / virtual currency", "Mortgage", "Payday / personal loan", "Student loan", "Vehicle loan"]

server = MCPServer(
    name="cfpb-complaint-intelligence",
    version="0.8.0",
    instructions=(
        "Tools for US consumer-finance complaints (CFPB data). Typical flow: triage_complaint -> score_payout_risk "
        "-> draft_reply. Drafts are for a HUMAN agent to review and edit - never present one as a final reply to a "
        "customer, and always pass on its guardrail warnings. The payout-risk score is for deciding which "
        "complaints to look at first; never use it to deprioritise or reject anyone. Read cfpb://model-cards for "
        "each model's limits."),
)


class APIError(ToolError):
    """An expected failure: the agent receives this message (other exceptions show only a generic error)."""


def _call(endpoint, payload=None, timeout=15):
    """Call the FastAPI service; turn connection problems and API errors into clear messages for the agent."""
    try:
        r = (requests.post(f"{API}{endpoint}", json=payload, timeout=timeout) if payload is not None
             else requests.get(f"{API}{endpoint}", timeout=timeout))
    except requests.RequestException:
        raise APIError(f"The Complaint Intelligence API is not reachable at {API}. "
                       "Start it with: uv run uvicorn cfpb.api:app")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text[:300]
        raise APIError(f"API error {r.status_code} on {endpoint}: {detail}")
    return r.json()


def _complaint(text, product=None, sub_product=None, issue=None, company=None, state=None, tags=None):
    if not text or len(text.strip()) < 20:
        raise APIError("The complaint text must be at least 20 characters.")
    if product is not None and product not in PRODUCTS:
        raise APIError(f"Unknown product '{product}'. Use one of: {', '.join(PRODUCTS)} - or leave it empty.")
    body = {"text": text, "product": product, "sub_product": sub_product, "issue": issue, "company": company,
            "state": state, "tags": tags, "submitted_via": "Web"}
    return {k: v for k, v in body.items() if v is not None}


# ------------------------------------------------------------------ tools
@server.tool()
def check_system_health() -> dict[str, Any]:
    """Check which parts of the system are ready: triage model, risk model, regulation index and the local LLM
    used for drafting. Call this first if another tool fails."""
    return _call("/health", timeout=5)


@server.tool()
def triage_complaint(text: str) -> dict[str, Any]:
    """Decide which of 9 consumer-finance products (and so which team) a complaint is about.

    Args:
        text: the complaint in the customer's own words (at least 20 characters). Card numbers, SSNs, e-mails and
              phone numbers are masked before any model sees them.

    Returns the product, the model's confidence (0-1) and the two next most likely products. If confidence is
    below about 0.6, say the routing is uncertain."""
    return _call("/triage", _complaint(text))


@server.tool()
def score_payout_risk(text: str, product: str | None = None, sub_product: str | None = None,
                      issue: str | None = None, company: str | None = None, state: str | None = None,
                      tags: str | None = None) -> dict[str, Any]:
    """Estimate how likely a complaint is to end with the company paying the customer, and why.

    Args:
        text: the complaint text.
        product: one of the 9 products; leave empty to let the triage model decide.
        sub_product, issue: CFPB sub-product and issue if known - they noticeably change the score, so pass them.
        company: company name as in CFPB data, e.g. "BANK OF AMERICA, NATIONAL ASSOCIATION".
        state: two-letter US state, e.g. "CA".
        tags: "Older American" or "Servicemember" if applicable.

    Returns the payout probability (the average complaint is about 0.013), a risk band ("high" = riskiest 10%,
    review first) and the top 3 reasons. Use it only to prioritise reviews, never to deprioritise anyone."""
    return _call("/risk", _complaint(text, product, sub_product, issue, company, state, tags))


@server.tool()
def draft_reply(text: str, product: str | None = None, issue: str | None = None) -> dict[str, Any]:
    """Draft a reply that cites the federal regulations retrieved for this complaint, checked by guardrails.
    Slow: about 30 seconds (up to a minute the first time).

    Args:
        text: the complaint text.
        product: one of the 9 products; leave empty to let the triage model decide.
        issue: CFPB issue if known.

    Returns the draft, the regulations retrieved, guardrail warnings and needs_careful_review. The draft is for a
    human agent: always show its warnings, and never present it as a final reply. "No warnings" does not mean
    the draft is correct - the checks catch specific mistakes only."""
    return _call("/draft", _complaint(text, product, None, issue), timeout=300)


@server.tool()
def get_drift_report(last_n_months: int = 6) -> dict[str, Any]:
    """Month-by-month monitoring of the deployed models (replay of 2025-26): data drift (PSI), accuracy and
    whether predicted payout rates match actual ones. PSI below 0.1 = stable, above 0.25 = major shift.

    Args:
        last_n_months: how many recent months to include (1-24)."""
    path = config.REPORTS / "m7_monitoring.csv"
    if not path.exists():
        raise APIError("No monitoring results yet. Run: uv run python scripts/m7_03_monitor.py")
    m = pd.read_csv(path)
    keep = ["month", "complaints", "payout_rate", "psi_product", "psi_score", "risk_roc_auc", "cal_rolling"]
    keep += ["triage_macro_f1"] if "triage_macro_f1" in m else []
    alerts = m[(m["psi_product"] > 0.25) | (m["psi_score"] > 0.25)]["month"].tolist()
    recent = m[keep].tail(max(1, min(int(last_n_months), 24))).round(4)
    return {"months_replayed": len(m),
            "drift_alert_months": alerts,
            "months_with_accurate_payout_rate": f"{int(m['cal_rolling'].between(0.8, 1.25).sum())}/"
                                                f"{int(m['cal_rolling'].notna().sum())}",
            "column_guide": {"psi_*": "drift vs reference (<0.1 stable, >0.25 major)",
                             "cal_rolling": "predicted / actual payout rate (1.0 = right)",
                             "risk_roc_auc": "risk ranking quality", "triage_macro_f1": "routing quality"},
            "recent_months": recent.to_dict("records")}


# ------------------------------------------------------------------ resources
@server.resource("cfpb://model-cards", name="model-cards", mime_type="text/markdown",
                 description="What each model is for, what it must not be used for, test results and known limits.")
def model_cards() -> str:
    path = config.ROOT / "docs" / "model_cards.md"
    return path.read_text(encoding="utf-8") if path.exists() else "docs/model_cards.md not found."


if __name__ == "__main__":
    server.run()          # stdio: the agent starts this process and talks to it over stdin/stdout
