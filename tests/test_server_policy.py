"""Server-side policy engine of the MCP server + the registry://tools resource (also over real stdio MCP)."""
import asyncio
import json

from core.server_policy import ServerPolicy


def test_policy_allowlist_clamps_validators_quotas():
    p = ServerPolicy()
    assert not p.check("format_disk", {})[0]
    ok, _, args = p.check("search_guidelines", {"query": "x", "k": 99})
    assert ok and args["k"] == 5
    assert not p.check("execute_query", {"sql": "DELETE FROM patients"})[0]
    assert not p.check("web_fetch", {"url": "http://localhost"})[0]
    assert not p.check("shell", {"command": "cat /etc/passwd"})[0]
    assert not p.check("memory_read", {"session_id": "", "limit": 2})[0]
    for _ in range(3):
        assert p.check("pubmed_search", {"query": "stroke", "k": 1})[0]
    assert "quota" in p.check("pubmed_search", {"query": "stroke", "k": 1})[1]


def test_guard_returns_refusal_as_data():
    p = ServerPolicy()

    @p.guard("execute_query")
    def execute_query(sql: str) -> str:
        return "ran"
    assert execute_query(sql="SELECT 1") == "ran"
    out = json.loads(execute_query(sql="DROP TABLE patients"))
    assert out["error"].startswith("refused by server policy") and p.counts["refused"] == 1


def test_registry_lists_tools_with_policies():
    reg = ServerPolicy().registry({"pubmed_search": "PubMed.", "shell": "Shell."})
    row = {r["tool"]: r for r in reg["tools"]}
    assert row["pubmed_search"]["per_run_quota"] == 3 and "evidence_subagent" in row["pubmed_search"]["granted_to"]
    assert row["shell"]["high_risk_controls"]


def test_real_mcp_server_enforces_policy_over_stdio():
    from core.mcp_client import mcp_session

    async def go():
        async with mcp_session() as s:
            names = [t.name for t in (await s.list_tools()).tools]
            assert len(names) == 20 and "pubmed_search" in names
            r = await s.call_tool("execute_query", {"sql": "DROP TABLE patients"})        # client hook bypassed on purpose
            assert "refused by server policy" in r.content[0].text
            r = await s.call_tool("search_guidelines", {"query": "smoking stroke", "k": 50})
            assert len(json.loads(r.content[0].text)) == 5                                # clamped server-side
            r = await s.call_tool("pubmed_search", {"query": "hypertension", "k": 2})      # offline -> real snapshot
            data = json.loads(r.content[0].text)
            assert data["served_from"].startswith("snapshot") and data["results"][0]["pmid"].isdigit()
            reg = json.loads((await s.read_resource("registry://tools")).contents[0].text)
            assert len(reg["tools"]) == 20 and reg["counters"]["refused"] >= 1
    asyncio.run(go())
