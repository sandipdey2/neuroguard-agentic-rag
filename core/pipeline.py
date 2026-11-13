"""Single entry point used by the Streamlit app, the scripts and the evaluation notebook.

run_case -> input guardrails (no LLM) -> fixed run contract -> MCP server subprocess -> LangGraph agents ->
finalize gate -> signed receipt. All async work runs on ONE long-lived background event loop, and every run
creates and closes its own httpx client inside that loop, so Streamlit reruns on Python 3.13/3.14 never touch
a closed loop ("Event loop is closed").
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path

import httpx
from langchain_mcp_adapters.tools import load_mcp_tools

from . import episodes
from .agents import build_graph
from .config import get_llm
from .data import validate_profile
from .guardrails import check_input
from .harness import RunContext, sha, sign_receipt
from .observability import metrics, otlp_json
from .mcp_client import SERVER, mcp_session

SERVER_SHA = hashlib.sha256(Path(SERVER).read_bytes()).hexdigest()


def _receipt(ctx: RunContext, inputs: dict, result: dict) -> dict:
    final = result.get("final") or {}
    v = final.get("verification") or {}
    return sign_receipt({
        "run_id": ctx.run_id, "product": "NeuroGuard AI", "contract_version": ctx.contract["contract_version"],
        "contract_sha256": ctx.contract_sha256, "model": ctx.contract["model"], "temperature": ctx.contract["temperature"],
        "mcp_server_sha256": SERVER_SHA, "inputs_sha256": sha(inputs), "report_sha256": sha(result.get("report") or ""),
        "trace_sha256": sha(result.get("trace", [])), "audit_chain_head": ctx.audit.entries[-1]["hash"] if ctx.audit.entries else None,
        "audit_chain_valid": ctx.audit.verify_chain(), "status": result["status"], "checks": v.get("checks", {}),
        "failed_checks": v.get("failed", []), "needs_human_approval": final.get("needs_human_approval", False),
        "approval": None, "counts": {"llm_calls": ctx.llm_calls, "tokens": ctx.tokens, "tool_calls": ctx.perms.tool_calls,
                                     "spans": len(ctx.spans), "decisions": len(ctx.decisions)},
        "cost_usd": round(ctx.cost_usd, 6), "confidence": (final.get("confidence") or {}).get("score"),
        "confidence_gate": (final.get("confidence") or {}).get("gate"), "decision_log_sha256": sha(ctx.decisions),
        "latency_s": result.get("latency_s"), "rag_mode": result.get("rag_mode")})


RAG_MODES = ("classical", "agentic", "agentic_search")


async def arun_case(profile: dict, question: str, note: str = "", audience: str = "clinician",
                    session_id: str = "anonymous", consent: bool = False, llm=None, progress=None, limiter=None,
                    rag_mode: str | None = None) -> dict:
    """``progress`` (optional callable) receives each new trace event as the graph runs - used for live UI updates.
    ``limiter`` (optional, batch mode) is a shared token-bucket rate limiter awaited before every LLM call."""
    t0 = time.perf_counter()
    ctx = RunContext(session_id=session_id, consent=consent, limiter=limiter)        # contract fixed before any LLM token
    rag_mode = rag_mode or ctx.contract.get("rag_modes", {}).get("default", "agentic")
    if rag_mode not in RAG_MODES:
        raise ValueError(f"rag_mode must be one of {RAG_MODES}")
    ctx.perms.rag_mode = rag_mode                                                    # per-run least-privilege tool set
    gate = check_input(question, note)
    ctx.audit.log("input_guardrail", "check_input", "question+note", "allow" if gate["allowed"] else "block",
                  reason=gate["reason"], phi_redacted=gate["phi_redacted"])
    try:
        prof = validate_profile(profile)
    except ValueError as exc:
        gate = {**gate, "allowed": False, "reason": "invalid_profile", "detail": str(exc)}
        prof = None
    inputs = {"profile": prof, "question": gate["question"], "note": gate["note"], "audience": audience, "rag_mode": rag_mode}
    base = {"rag_mode": rag_mode, "run_id": ctx.run_id, "inputs": inputs, "input_guardrail": gate, "contract_sha256": ctx.contract_sha256}
    if not gate["allowed"]:
        ctx.decide("input_guardrail", "block", gate["reason"], "input guardrail blocked the request before any LLM call")
        res = {**base, "status": "blocked", "trace": [{"agent": "input_guardrail", "type": "guardrail", **gate}],
               "latency_s": round(time.perf_counter() - t0, 2), "tool_calls": 0, "llm_calls": 0, "a2a": [], "audit": ctx.audit.entries,
               "decisions": ctx.decisions, "spans": [], "cost": ctx.cost_summary(), "metrics": metrics([])}
        res["receipt"] = _receipt(ctx, inputs, res)
        return res
    try:
        if llm is None:
            async with httpx.AsyncClient(timeout=httpx.Timeout(ctx.contract["llm_timeout_s"])) as http_client:
                state = await asyncio.wait_for(_run_graph(inputs, gate, ctx, get_llm(http_async_client=http_client), progress),
                                               timeout=ctx.contract["budget"]["wall_clock_timeout_s"] + 30)
        else:
            state = await asyncio.wait_for(_run_graph(inputs, gate, ctx, llm, progress), timeout=ctx.contract["budget"]["wall_clock_timeout_s"] + 30)
        status = state["final"]["status"]
    except asyncio.TimeoutError:
        state, status = {"trace": [{"agent": "harness", "type": "timeout"}]}, "timeout"
        ctx.audit.log("harness", "timeout", "run", "stop")
    res = {**base, **{k: v for k, v in state.items() if k not in ("task",)}, "status": status,
           "report": state.get("draft", ""), "tool_calls": ctx.perms.tool_calls, "llm_calls": ctx.llm_calls, "tokens": ctx.tokens,
           "latency_s": round(time.perf_counter() - t0, 2), "a2a": ctx.bus.log, "audit": ctx.audit.entries,
           "decisions": ctx.decisions, "spans": ctx.spans, "cost": ctx.cost_summary(), "metrics": metrics(ctx.spans),
           "confidence": (state.get("final") or {}).get("confidence"), "episode": (state.get("final") or {}).get("episode")}
    res["otlp"] = otlp_json(ctx.run_id, ctx.spans, ctx.started, time.time())
    res["receipt"] = _receipt(ctx, inputs, res)
    return res


async def _run_graph(inputs: dict, gate: dict, ctx: RunContext, llm, progress=None) -> dict:
    async with mcp_session() as session:
        listed = await session.list_tools()
        tools = await load_mcp_tools(session)
        ctx.audit.log("harness", "mcp_initialize", "neuroguard", "ok", tools=[t.name for t in listed.tools])
        graph = build_graph(llm, tools, session, ctx)
        init = {"rag_mode": inputs["rag_mode"], "question": inputs["question"], "note": inputs["note"], "audience": inputs["audience"], "profile": inputs["profile"],
                "turns": 0, "visits": {}, "revisions": 0, "pending_review": False, "retrieved": {}, "evidence_log": [],
                "trace": [{"agent": "input_guardrail", "type": "guardrail", **gate}], "route": [], "self_critique": {}}
        state, seen = None, 0
        # astream(values) yields the full state after every node; the last value equals ainvoke's result.
        async for state in graph.astream(init, config={"recursion_limit": 80}, stream_mode="values"):
            tr = state.get("trace", [])
            if progress is not None:
                for ev in tr[seen:]:
                    try:
                        progress(ev)
                    except Exception:      # a UI callback must never break a run
                        pass
            seen = len(tr)
        state["mcp_tools_listed"] = [t.name for t in listed.tools]
        return state


def approve(result: dict, decision: str, note: str = "") -> dict:
    """Record a clinician decision on a finalized draft and re-sign the receipt (the old signature is chained in)."""
    from .harness import sign_receipt as _sign
    r = {k: v for k, v in result["receipt"].items() if k not in ("signature", "signed_at", "key_id", "algorithm")}
    r["approval"] = {"decision": decision, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": note[:200],
                     "previous_signature": result["receipt"]["signature"]}
    try:      # feed the clinician decision back into episodic memory (learning loop); never blocks the approval
        episodes.set_clinician_decision(result["run_id"], decision)
    except Exception:
        pass
    return _sign(r)


# ----------------------------------------------------------------------------- one persistent event loop
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    with _lock:
        if _loop is None or _loop.is_closed() or _thread is None or not _thread.is_alive():
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(target=_loop.run_forever, name="neuroguard-asyncio", daemon=True)
            _thread.start()
        return _loop


def start_case(profile: dict, question: str, note: str = "", audience: str = "clinician", session_id: str = "anonymous",
               consent: bool = False, llm=None, progress=None, rag_mode: str | None = None):
    """Schedule a run on the background loop and return a concurrent.futures.Future (the UI polls it for live progress)."""
    return asyncio.run_coroutine_threadsafe(arun_case(profile, question, note, audience, session_id, consent, llm, progress,
                                                      rag_mode=rag_mode),
                                            _background_loop())


def run_case(profile: dict, question: str, note: str = "", audience: str = "clinician", session_id: str = "anonymous",
             consent: bool = False, llm=None, timeout: float = 400, progress=None, rag_mode: str | None = None) -> dict:
    return start_case(profile, question, note, audience, session_id, consent, llm, progress, rag_mode).result(timeout=timeout)


def describe(ev: dict) -> str | None:
    """One-line human description of a trace event for the live progress panel (None = not shown)."""
    t, a = ev.get("type"), ev.get("agent")
    if t == "route":
        return f"🧭 Orchestrator (turn {ev['turn']}) -> **{ev['next']}**" + (f" with skill `{ev['skill']}`" if ev.get("skill") else "") + \
            (f" - override: {ev['override']}" if ev.get("override") else f" - {ev.get('reason', '')}")
    if t == "mcp_tool_call":
        return f"🔧 {a} -> MCP `{ev['tool']}` ({ev['decision']}, {ev['ms']} ms)"
    if t == "mcp_resource_read":
        return f"📄 {a} read MCP resource `{ev['uri']}`"
    if t == "mcp_prompt_get":
        return f"🧾 {a} fetched MCP prompt `{ev['prompt']}`"
    if t == "skill_loaded":
        return f"📘 {a} loaded skill `{ev['skill']}`"
    if t == "review":
        return f"🔍 Safety review: **{ev['verdict']}**" + (f" (failed: {', '.join(ev['failed_checks'])})" if ev.get("failed_checks") else "")
    if t == "gate":
        return f"🚦 Finalize gate: **{ev['status']}**" + (f" - confidence {ev['confidence']} ({ev['confidence_gate']})" if ev.get("confidence") is not None else "")
    if t == "plan":
        return f"🗺️ Planner: {' -> '.join(st['agent'] for st in ev['steps'])}" + (f" ({len(ev['learned_hints'])} learned hint(s))" if ev.get("learned_hints") else "")
    if t == "react_step":
        return f"💭 {a} thought: {ev['thought'][:140]}"
    if t == "evidence_sufficiency":
        return f"📚 Evidence round {ev['round']}: coverage {ev['score']}" + (f" - missing {ev['missing']} -> re-retrieve" if ev.get("missing") else " - sufficient")
    if t == "self_critique":
        return f"🪞 Writer self-critique: {'revised' if ev['revised'] else 'kept'} ({len(ev['code_issues'])} check issue(s), {len(ev['llm_problems'])} LLM-flagged)"
    if t == "context_window":
        top = sorted(ev["slots"], key=lambda r: -r["tokens"])[:3]
        return (f"🪟 {a} context window ({ev.get('rag_mode') or 'agent'}): {ev['total_tokens']} tokens in {len(ev['slots'])} slots - "
                + ", ".join(f"{r['slot'].split(' ')[0]} {r['tokens']}" for r in top))
    if t == "source_check":
        return f"🧩 {a} source check: not yet consulted {ev['missing_sources']} -> asked to use them"
    if t == "warning":
        return f"⚠️ {a}: {ev.get('detail')}"
    return None


def dumps(result: dict) -> str:
    return json.dumps(result, default=str, indent=1)
