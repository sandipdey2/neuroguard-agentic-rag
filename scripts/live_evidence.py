"""Evidence that MCP is real and the workflow is agentic:  python scripts/live_evidence.py [out.json]

1. MCP: spawns the server exactly as the app does (core/mcp_client.server_params), records the initialize
   handshake (protocol version, serverInfo, capabilities), the child PID and parent PID from /proc, and confirms
   no secret variable names reached the child's environment.
2. LLM tool choice: binds the Risk Analyst's MCP tools to gpt-4o-mini with bind_tools and records the
   tool_calls the model itself returned (finish_reason == "tool_calls"), then executes them over MCP.
3. Full run: one case through the whole graph; records every orchestrator decision (LLM choice, reason,
   code override) and every MCP tool call with the hook decision.
No secrets are printed or written.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import httpx  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_mcp_adapters.tools import load_mcp_tools  # noqa: E402
from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from core.config import get_llm, load_contract  # noqa: E402
from core.mcp_client import SERVER, server_params  # noqa: E402
from core.pipeline import run_case  # noqa: E402

P = {"age": 67, "gender": "Male", "hypertension": 0, "heart_disease": 1, "ever_married": "Yes", "work_type": "Private",
     "Residence_type": "Urban", "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked"}
SECRET_NAMES = ("OPENAI_API_KEY", "OPENAI_API_BASE", "NG_RECEIPT_KEY", "GH_TOKEN", "GITHUB_TOKEN")


def children():
    me, out = os.getpid(), []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            if SERVER.name in cmd and ppid == me:
                env_names = {kv.split(b"=", 1)[0].decode() for kv in (d / "environ").read_bytes().split(b"\0") if b"=" in kv}
                out.append({"pid": int(d.name), "ppid": ppid, "cmdline": cmd.strip().replace(str(ROOT), "<repo>"),
                            "secret_vars_in_child_env": sorted(n for n in SECRET_NAMES if n in env_names)})
        except (OSError, ValueError, IndexError):
            pass
    return out


async def mcp_and_tool_choice():
    ev = {}
    async with stdio_client(server_params()) as (r, w):
        async with ClientSession(r, w) as s:
            t0 = time.perf_counter()
            init = await s.initialize()
            ev["handshake"] = {"protocolVersion": init.protocolVersion, "serverInfo": init.serverInfo.model_dump(),
                               "capabilities": sorted(k for k, v in init.capabilities.model_dump().items() if v),
                               "ms": round((time.perf_counter() - t0) * 1000)}
            ev["server_process"] = children()
            ev["client_pid"] = os.getpid()
            ev["tools_list"] = [t.name for t in (await s.list_tools()).tools]
            tools = {t.name: t for t in await load_mcp_tools(s)}
            allowed = load_contract()["agents"]["risk_analyst"]["tools"]
            async with httpx.AsyncClient(timeout=60) as hc:
                llm = get_llm(http_async_client=hc).bind_tools([tools[n] for n in allowed])
                ai = await llm.ainvoke([SystemMessage("You are the Risk Analyst agent. Call predict_risk and explain_factors with the profile values "
                                                      "exactly as given, then cohort_stats."),
                                        HumanMessage(f"PATIENT PROFILE: {json.dumps(P)}")])
            ev["llm_tool_choice"] = {"bound_tools": allowed, "model": ai.response_metadata.get("model_name"),
                                     "finish_reason": ai.response_metadata.get("finish_reason"),
                                     "openai_response_id_prefix": str(ai.response_metadata.get("id") or "")[:14],
                                     "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in ai.tool_calls]}
            res = []
            for tc in ai.tool_calls:
                out = await tools[tc["name"]].ainvoke(tc["args"])
                txt = out[0] if isinstance(out, tuple) else out
                txt = txt if isinstance(txt, str) else json.dumps(txt, default=str)
                res.append({"tool": tc["name"], "mcp_result_preview": txt[:220]})
            ev["llm_tool_choice"]["executed_over_mcp"] = res
    ev["server_process_after_close"] = children()
    return ev


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval" / "logs" / f"live_evidence_py{sys.version_info.major}{sys.version_info.minor}.json"
    ev = {"python": sys.version.split()[0], "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    ev.update(asyncio.run(mcp_and_tool_choice()))
    r = run_case(P, "Explain this patient's stroke risk and what to discuss at the next visit.",
                 "Follow-up visit. Denies weakness or speech problems.", session_id="evidence", consent=False)
    ev["full_run"] = {"status": r["status"], "latency_s": r["latency_s"], "llm_calls": r["llm_calls"], "tool_calls": r["tool_calls"],
                      "mcp_tools_listed": r.get("mcp_tools_listed"), "route": r.get("route"),
                      "orchestrator_decisions": [{k: t.get(k) for k in ("turn", "eligible", "next", "skill", "reason", "override")}
                                                 for t in r["trace"] if t.get("type") == "route"],
                      "mcp_tool_calls": [{k: t.get(k) for k in ("agent", "tool", "args", "decision", "hook_reason", "ms")}
                                         for t in r["trace"] if t.get("type") == "mcp_tool_call"],
                      "resources_and_prompts": [t.get("uri") or t.get("prompt") for t in r["trace"] if t.get("type") in ("mcp_resource_read", "mcp_prompt_get")],
                      "skills_loaded": [t["skill"] for t in r["trace"] if t.get("type") == "skill_loaded"],
                      "gate": next((t for t in r["trace"] if t.get("type") == "gate"), None),
                      "receipt_signature_prefix": r["receipt"]["signature"][:16],
                      "plan": [s_["agent"] for s_ in (r.get("plan") or {}).get("steps", [])],
                      "react_steps": [{k: t.get(k) for k in ("agent", "step", "thought", "action")} for t in r["trace"] if t.get("type") == "react_step"],
                      "evidence_sufficiency": [{k: t.get(k) for k in ("round", "score", "missing", "action")} for t in r["trace"] if t.get("type") == "evidence_sufficiency"],
                      "self_critique": r.get("self_critique"), "confidence": r.get("confidence"), "cost": r.get("cost"),
                      "decision_kinds": [d["kind"] for d in r.get("decisions", [])], "spans": len(r.get("spans", [])),
                      "episode_stored": (r.get("episode") or {}).get("stored")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ev, indent=1, default=str))
    print(json.dumps({k: ev[k] for k in ("python", "handshake", "server_process", "client_pid", "llm_tool_choice", "server_process_after_close")}, indent=1, default=str))
    fr = ev["full_run"]
    print("full run:", fr["status"], fr["latency_s"], "s | llm", fr["llm_calls"], "| tools", fr["tool_calls"], "| route", fr["route"])
    print("  plan", fr["plan"], "| react steps", len(fr["react_steps"]), "| sufficiency", fr["evidence_sufficiency"],
          "| confidence", (fr["confidence"] or {}).get("score"), "| cost $", (fr["cost"] or {}).get("cost_usd"), "| decisions", len(fr["decision_kinds"]))
    for d in fr["orchestrator_decisions"]:
        print(f"  turn {d['turn']}: eligible={d['eligible']} -> {d['next']} ({d['skill']}) override={d['override']}")
    print("wrote", out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
