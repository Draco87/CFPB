"""
MCP demo: act as an AI agent and use the Complaint Intelligence tools through MCP.   (API must be running)

    uv run python scripts/m8_mcp_demo.py            # triage + risk + drift report (seconds)
    uv run python scripts/m8_mcp_demo.py --draft    # also draft a reply (~30-60 s, needs Ollama)

This starts the MCP server as a separate process and talks to it over stdin/stdout - exactly what Claude Desktop,
an IDE assistant or an agent framework does. It shows what an agent sees: the tool list (with descriptions), the
tool results, and a clear error message when something is wrong.
"""
import argparse
import asyncio
import json
import os
import sys

from mcp import Client
from mcp.client.stdio import StdioServerParameters

COMPLAINT = ("Someone made two ATM withdrawals of $200 from my checking account that I did not make. I reported it "
             "to the bank the same day but they have not investigated and keep telling me to wait.")


def show(title, result):
    print(f"\n=== {title}")
    if result.is_error:
        print("ERROR (as the agent sees it):", result.content[0].text)
    else:
        print(json.dumps(result.structured_content or json.loads(result.content[0].text), indent=2)[:1500])


async def main(draft):
    params = StdioServerParameters(command=sys.executable, args=["-m", "cfpb.mcp_server"], env=dict(os.environ))
    async with Client(params, read_timeout_seconds=320) as agent:
        print("Connected to:", agent.server_info.name, agent.server_info.version)
        print("\nTools the agent can use:")
        for t in (await agent.list_tools()).tools:
            print(f"  - {t.name}: {(t.description or '').strip().splitlines()[0]}")
        print("Resources:", [str(r.uri) for r in (await agent.list_resources()).resources])

        show("check_system_health", await agent.call_tool("check_system_health", {}))
        show("triage_complaint", await agent.call_tool("triage_complaint", {"text": COMPLAINT}))
        show("score_payout_risk", await agent.call_tool("score_payout_risk", {
            "text": COMPLAINT, "company": "BANK OF AMERICA, NATIONAL ASSOCIATION", "state": "CA",
            "sub_product": "Checking account"}))
        show("get_drift_report (last 3 months)", await agent.call_tool("get_drift_report", {"last_n_months": 3}))
        show("a mistake an agent might make (bad product)", await agent.call_tool(
            "score_payout_risk", {"text": COMPLAINT, "product": "Savings"}))
        if draft:
            print("\nDrafting (30-60 s) ...", flush=True)
            show("draft_reply", await agent.call_tool("draft_reply", {"text": COMPLAINT}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    asyncio.run(main(ap.parse_args().draft))
