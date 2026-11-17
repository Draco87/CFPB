"""
Tests for the MCP server.      uv run pytest -q tests/test_mcp.py

An MCP client talks to the server IN-PROCESS (no subprocess), and the HTTP calls to the API are replaced by a
fake, so these tests need no running API, no models and no GPU. They check what an agent would see: which tools
exist, that arguments reach the API correctly, and that failures come back as clear error messages.
"""
import asyncio

import pytest
import requests
from mcp import Client

from cfpb import mcp_server

TEXT = "Someone made two ATM withdrawals of $200 from my checking account that I did not make."


class FakeResponse:
    def __init__(self, code, body):
        self.status_code, self._body, self.text = code, body, str(body)

    def json(self):
        return self._body


@pytest.fixture
def api(monkeypatch):
    sent = []

    def post(url, json=None, timeout=None):
        sent.append((url.rsplit("/", 1)[-1], json))
        if url.endswith("/triage"):
            return FakeResponse(200, {"product": "Bank account", "confidence": 0.99, "alternatives": []})
        if url.endswith("/risk"):
            return FakeResponse(200, {"product": json.get("product", "Bank account"), "payout_probability": 0.07,
                                      "risk_band": "high", "top_drivers": []})
        return FakeResponse(503, {"detail": "Ollama is not running."})

    def get(url, timeout=None):
        return FakeResponse(200, {"status": "ok", "llm": "ready"})

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", get)
    return sent


def run(coro):
    return asyncio.run(coro)


async def _session(fn):
    async with Client(mcp_server.server) as client:
        return await fn(client)


def test_agent_sees_all_tools_and_the_model_cards(api):
    async def go(c):
        tools = {t.name for t in (await c.list_tools()).tools}
        resources = {str(r.uri) for r in (await c.list_resources()).resources}
        return tools, resources
    tools, resources = run(_session(go))
    assert {"check_system_health", "triage_complaint", "score_payout_risk", "draft_reply",
            "get_drift_report"} <= tools
    assert "cfpb://model-cards" in resources


def test_triage_calls_the_api(api):
    res = run(_session(lambda c: c.call_tool("triage_complaint", {"text": TEXT})))
    assert not res.is_error and res.structured_content["product"] == "Bank account"
    assert api[0] == ("triage", {"text": TEXT, "submitted_via": "Web"})


def test_risk_passes_details_through(api):
    res = run(_session(lambda c: c.call_tool("score_payout_risk", {
        "text": TEXT, "product": "Bank account", "sub_product": "Checking account", "state": "CA"})))
    assert not res.is_error and res.structured_content["risk_band"] == "high"
    assert api[0][1]["sub_product"] == "Checking account" and api[0][1]["state"] == "CA"


def test_bad_input_is_a_clear_error_not_a_crash(api):
    short = run(_session(lambda c: c.call_tool("triage_complaint", {"text": "help"})))
    bad = run(_session(lambda c: c.call_tool("score_payout_risk", {"text": TEXT, "product": "Bananas"})))
    assert short.is_error and "20 characters" in short.content[0].text
    assert bad.is_error and "Unknown product" in bad.content[0].text
    assert api == []                                  # nothing invalid was sent to the API


def test_api_failure_is_explained(api):
    res = run(_session(lambda c: c.call_tool("draft_reply", {"text": TEXT})))
    assert res.is_error and "Ollama is not running" in res.content[0].text


def test_slow_api_is_reported_as_slow_not_as_down(monkeypatch):
    def post(url, json=None, timeout=None):
        raise requests.Timeout()
    monkeypatch.setattr(requests, "post", post)
    res = run(_session(lambda c: c.call_tool("triage_complaint", {"text": TEXT})))
    assert res.is_error and "did not answer" in res.content[0].text and "not reachable" not in res.content[0].text
