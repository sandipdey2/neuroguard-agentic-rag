"""Agent as a tool: the Evidence Researcher sub-agent wrapped as a LangChain tool another agent can bind and call.

``make_evidence_agent_tool`` returns a ``StructuredTool`` named ``consult_evidence_researcher``. Calling it runs a
complete, bounded sub-agent: its own ReAct loop (gpt-4o-mini + bind_tools) over the MCP tools granted to
``evidence_subagent`` in harness_contract.json (search_guidelines, graph_traverse, pubmed_search), every call through
the same pre-tool-use hook, budget and audit chain as the parent run, then a structured answer whose citations are
kept only if a tool really returned that excerpt id. The parent agent (the chat follow-up agent, core/chat.py)
sees only the tool's JSON result - the sub-agent's context window stays separate (context isolation).

Delegation is still traced as A2A: the call opens a Task on the run's MessageBus (followup_agent -> evidence_subagent)
and completes it with a ``cited_findings`` artifact, so the A2A log shows agent-as-tool calls like any other delegation.
"""
from __future__ import annotations

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .context import truncate
from .guardrails import redact

SUB_SYSTEM = ("You are the Evidence Researcher sub-agent of NeuroGuard (decision support, not a medical device). Answer the "
              "QUESTION with evidence: call search_guidelines (CDC/NHLBI excerpts, cite as G-xx) for each topic, optionally "
              "graph_traverse with factor names, and at most one pubmed_search for peer-reviewed literature metadata. Before "
              "every tool call write one short line starting with 'Thought:'. Never name medicines or doses and never "
              "diagnose. Only excerpts returned by search_guidelines or graph_traverse may be cited.")


class ConsultArgs(BaseModel):
    question: str = Field(description="The evidence question to research, e.g. 'how does high blood pressure affect stroke risk'")
    factors: list[str] = Field(default_factory=list, description="Risk-factor names: hypertension, glucose, bmi, smoking, heart_disease, age, tia, lifestyle")


class SubFinding(BaseModel):
    claim: str = Field(description="One short sentence supported by the excerpt")
    citation_id: str = Field(description="Excerpt id such as G-03")


class SubOut(BaseModel):
    findings: list[SubFinding] = Field(default_factory=list)


def _text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(x.get("text", "") if isinstance(x, dict) else getattr(x, "text", str(x)) for x in result)
    return getattr(result, "text", None) or str(result)


def returned_ids(outputs: list[str]) -> dict:
    ids = {}
    for out in outputs:
        try:
            data = json.loads(out)
        except (json.JSONDecodeError, TypeError):
            continue
        rows = (data.get("chunks") or data.get("results") or []) if isinstance(data, dict) else data
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict) and str(r.get("id", "")).startswith("G-"):
                ids[r["id"]] = r
    return ids


def make_evidence_agent_tool(llm, tool_map: dict, ctx, sink: dict, max_rounds: int | None = None,
                             parent: str = "followup_agent") -> StructuredTool:
    """``sink`` collects side outputs for the parent harness: trace, evidence_log, retrieved excerpts, literature."""
    max_rounds = max_rounds or ctx.contract.get("agent_as_tool", {}).get("max_tool_rounds", 3)
    for k in ("trace", "evidence_log", "literature"):
        sink.setdefault(k, [])
    sink.setdefault("retrieved", {})

    async def consult(question: str, factors: list[str] | None = None) -> str:
        t0 = time.perf_counter()
        factors = list(factors or [])
        task = ctx.bus.new_task(parent, "evidence_subagent", question, data={"factors": factors}, skill="guideline-citation")
        ctx.audit.log(parent, "agent_as_tool_call", "consult_evidence_researcher", "allow", factors=factors)
        allowed = [tool_map[n] for n in ctx.perms.allowed_tools("evidence_subagent") if n in tool_map]
        bound = llm.bind_tools(allowed) if allowed else llm
        msgs = [SystemMessage(SUB_SYSTEM), HumanMessage(f"QUESTION: {question}\nFACTORS: {factors}")]
        outputs = []
        for step in range(1, max_rounds + 1):
            if not ctx.budget_left():
                break
            w = time.time()
            ai = await bound.ainvoke(msgs)
            ctx.record_llm("evidence_subagent", ai, w, time.time())
            msgs.append(ai)
            thought = ai.content.strip() if isinstance(ai.content, str) else ""
            if not ai.tool_calls:
                break
            for tc in ai.tool_calls:
                d = ctx.perms.pre_tool_use("evidence_subagent", tc["name"], tc["args"])
                w1 = time.time()
                if not d.allow or tc["name"] not in tool_map:
                    out = json.dumps({"error": f"blocked by pre-tool-use hook: {d.reason}"})
                else:
                    try:
                        out, _ = redact(_text(await tool_map[tc["name"]].ainvoke(d.args)))
                    except Exception as exc:
                        out = json.dumps({"error": str(exc)[:300]})
                ctx.record_tool_span("evidence_subagent", tc["name"], w1, time.time(), "allow" if d.allow else "deny")
                outputs.append(out)
                sink["evidence_log"].append({"agent": "evidence_subagent", "tool": tc["name"], "args": d.args, "output": out, "allowed": d.allow})
                if tc["name"] == "pubmed_search" and d.allow:
                    try:
                        sink["literature"] += json.loads(out).get("results", [])
                    except (json.JSONDecodeError, AttributeError):
                        pass
                obs, _ = truncate(out, ctx.contract["context"]["max_tool_output_tokens"])
                msgs.append(ToolMessage(content=obs, tool_call_id=tc["id"]))
                sink["trace"].append({"agent": "evidence_subagent", "type": "react_step", "step": step,
                                      "thought": thought[:300] or "(no thought text returned with the tool call)",
                                      "action": f"{tc['name']}({json.dumps(d.args)[:140]})", "observation": obs[:200],
                                      "decision": "allow" if d.allow else "deny"})
        msgs.append(HumanMessage("Return your findings now, using only the tool results above."))
        w = time.time()
        res = await llm.with_structured_output(SubOut, include_raw=True).ainvoke(msgs)
        ctx.record_llm("evidence_subagent", res["raw"], w, time.time())
        parsed = res.get("parsed") or SubOut()
        ids = returned_ids(outputs)
        kept = [f.model_dump() for f in parsed.findings if f.citation_id in ids]
        dropped = len(parsed.findings) - len(kept)
        for f in kept:
            sink["retrieved"][f["citation_id"]] = ids[f["citation_id"]]
        art = {"findings": kept, "dropped_unretrieved": dropped,
               "literature": [{k: x.get(k) for k in ("pmid", "title", "journal", "year", "url")} for x in sink["literature"][:3]]}
        ctx.bus.complete(task, "evidence_subagent", "cited_findings", art)
        sink["trace"].append({"agent": "evidence_subagent", "type": "agent_output", "output": art, "ms": round((time.perf_counter() - t0) * 1000)})
        return json.dumps(art)

    return StructuredTool.from_function(
        coroutine=consult, name="consult_evidence_researcher", args_schema=ConsultArgs,
        description=("Delegate an evidence question to the Evidence Researcher sub-agent. It searches CDC/NHLBI guideline "
                     "excerpts (and PubMed metadata) and returns findings with G-xx citation ids you may cite."))
