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

from .agents import build_graph
from .config import get_llm
from .data import validate_profile
from .guardrails import check_input
from .harness import RunContext, sha, sign_receipt
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
        "approval": None, "counts": {"llm_calls": ctx.llm_calls, "tokens": ctx.tokens, "tool_calls": ctx.perms.tool_calls},
        "latency_s": result.get("latency_s")})


async def arun_case(profile: dict, question: str, note: str = "", audience: str = "clinician",
                    session_id: str = "anonymous", consent: bool = False, llm=None) -> dict:
    t0 = time.perf_counter()
    ctx = RunContext(session_id=session_id, consent=consent)        # contract fixed before any LLM token
    gate = check_input(question, note)
    ctx.audit.log("input_guardrail", "check_input", "question+note", "allow" if gate["allowed"] else "block",
                  reason=gate["reason"], phi_redacted=gate["phi_redacted"])
    try:
        prof = validate_profile(profile)
    except ValueError as exc:
        gate = {**gate, "allowed": False, "reason": "invalid_profile", "detail": str(exc)}
        prof = None
    inputs = {"profile": prof, "question": gate["question"], "note": gate["note"], "audience": audience}
    base = {"run_id": ctx.run_id, "inputs": inputs, "input_guardrail": gate, "contract_sha256": ctx.contract_sha256}
    if not gate["allowed"]:
        res = {**base, "status": "blocked", "trace": [{"agent": "input_guardrail", "type": "guardrail", **gate}],
               "latency_s": round(time.perf_counter() - t0, 2), "tool_calls": 0, "llm_calls": 0, "a2a": [], "audit": ctx.audit.entries}
        res["receipt"] = _receipt(ctx, inputs, res)
        return res
    try:
        if llm is None:
            async with httpx.AsyncClient(timeout=httpx.Timeout(ctx.contract["llm_timeout_s"])) as http_client:
                state = await asyncio.wait_for(_run_graph(inputs, gate, ctx, get_llm(http_async_client=http_client)),
                                               timeout=ctx.contract["budget"]["wall_clock_timeout_s"] + 30)
        else:
            state = await asyncio.wait_for(_run_graph(inputs, gate, ctx, llm), timeout=ctx.contract["budget"]["wall_clock_timeout_s"] + 30)
        status = state["final"]["status"]
    except asyncio.TimeoutError:
        state, status = {"trace": [{"agent": "harness", "type": "timeout"}]}, "timeout"
        ctx.audit.log("harness", "timeout", "run", "stop")
    res = {**base, **{k: v for k, v in state.items() if k not in ("task",)}, "status": status,
           "report": state.get("draft", ""), "tool_calls": ctx.perms.tool_calls, "llm_calls": ctx.llm_calls, "tokens": ctx.tokens,
           "latency_s": round(time.perf_counter() - t0, 2), "a2a": ctx.bus.log, "audit": ctx.audit.entries}
    res["receipt"] = _receipt(ctx, inputs, res)
    return res


async def _run_graph(inputs: dict, gate: dict, ctx: RunContext, llm) -> dict:
    async with mcp_session() as session:
        listed = await session.list_tools()
        tools = await load_mcp_tools(session)
        ctx.audit.log("harness", "mcp_initialize", "neuroguard", "ok", tools=[t.name for t in listed.tools])
        graph = build_graph(llm, tools, session, ctx)
        init = {"question": inputs["question"], "note": inputs["note"], "audience": inputs["audience"], "profile": inputs["profile"],
                "turns": 0, "visits": {}, "revisions": 0, "pending_review": False, "retrieved": {}, "evidence_log": [],
                "trace": [{"agent": "input_guardrail", "type": "guardrail", **gate}], "route": []}
        state = await graph.ainvoke(init, config={"recursion_limit": 80})
        state["mcp_tools_listed"] = [t.name for t in listed.tools]
        return state


def approve(result: dict, decision: str, note: str = "") -> dict:
    """Record a clinician decision on a finalized draft and re-sign the receipt (the old signature is chained in)."""
    from .harness import sign_receipt as _sign
    r = {k: v for k, v in result["receipt"].items() if k not in ("signature", "signed_at", "key_id", "algorithm")}
    r["approval"] = {"decision": decision, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": note[:200],
                     "previous_signature": result["receipt"]["signature"]}
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


def run_case(profile: dict, question: str, note: str = "", audience: str = "clinician", session_id: str = "anonymous",
             consent: bool = False, llm=None, timeout: float = 400) -> dict:
    fut = asyncio.run_coroutine_threadsafe(arun_case(profile, question, note, audience, session_id, consent, llm), _background_loop())
    return fut.result(timeout=timeout)


def dumps(result: dict) -> str:
    return json.dumps(result, default=str, indent=1)
