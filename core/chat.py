"""Chat interface back end: run a case from a chat message, then answer follow-up questions with conversation memory.

``handle_message`` is shared by the Streamlit chat tab, the HTTP API and the Slack bot:
* a case request ("assess row 9046", or "age=67 gender=Male hypertension=1 glucose=228 ...") runs the full multi-agent
  pipeline (core/pipeline.py) and replies with a de-identified summary + receipt;
* anything else is a follow-up about the latest case: the follow-up agent (gpt-4o-mini, bind_tools) gets the case
  summary + the recalled conversation turns and may call ``consult_evidence_researcher`` - the Evidence Researcher
  sub-agent wrapped as a tool (core/agent_tools.py). Its answer passes the same input guardrails before the LLM and
  deterministic output checks after it (no dosing, no diagnosis, citations must have been retrieved, disclaimer).
Every follow-up has its own RunContext: contract hash, budget, audit chain, A2A log and an HMAC-signed receipt.
"""
from __future__ import annotations

import asyncio
import json
import re
import time

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools

from .agent_tools import make_evidence_agent_tool
from .chat_memory import get_memory
from .config import get_llm
from .guardrails import check_input
from .harness import DIAGNOSIS_PATTERNS, DOSING_PATTERNS, RunContext, sha, sign_receipt
from .mcp_client import mcp_session

FOLLOWUP_SYSTEM = ("You are the NeuroGuard follow-up assistant for a stroke-risk decision-support tool (not a medical device). "
                   "Answer the user's follow-up question about the CASE SUMMARY in at most 120 words. Use the CONVERSATION so far for "
                   "context (e.g. 'it', 'that factor'). For any guideline evidence, call consult_evidence_researcher and cite only the "
                   "G-xx ids it returns, like [G-03]. Never name medicines or doses, never say the patient has had or is having a "
                   "stroke, never invent numbers: only numbers from the CASE SUMMARY or the tool result.")
DISCLAIMER = "Decision support only - not a medical device and not a diagnosis."
FIELDS = {"age": float, "gender": str, "hypertension": int, "heart_disease": int, "ever_married": str, "work_type": str,
          "residence_type": str, "avg_glucose_level": float, "glucose": float, "bmi": float, "smoking_status": str, "smoking": str}
DEFAULTS = {"gender": "Female", "hypertension": 0, "heart_disease": 0, "ever_married": "Yes", "work_type": "Private",
            "Residence_type": "Urban", "avg_glucose_level": 100.0, "bmi": None, "smoking_status": "never smoked"}
SMOKING = {"never": "never smoked", "former": "formerly smoked", "formerly": "formerly smoked", "smokes": "smokes",
           "current": "smokes", "unknown": "Unknown"}


# ----------------------------------------------------------------------------- case requests
def parse_case_request(text: str, rows=None) -> dict | None:
    """'row 9046' (a real row of the public dataset) or key=value pairs -> profile dict; None when not a case request."""
    t = (text or "").strip()
    m = re.search(r"\brow\s*(?:id\s*)?#?\s*(\d{1,6})\b", t, re.I)
    if m:
        if rows is None:
            from .data import load_stroke
            rows, _ = load_stroke()
        hit = rows[rows["id"] == int(m.group(1))]
        if hit.empty:
            raise ValueError(f"dataset row id {m.group(1)} not found")
        r = hit.iloc[0]
        import math
        return {"age": float(r.age), "gender": r.gender, "hypertension": int(r.hypertension), "heart_disease": int(r.heart_disease),
                "ever_married": r.ever_married, "work_type": r.work_type, "Residence_type": r.Residence_type,
                "avg_glucose_level": float(r.avg_glucose_level), "bmi": None if (r.bmi is None or (isinstance(r.bmi, float) and math.isnan(r.bmi))) else float(r.bmi),
                "smoking_status": r.smoking_status}
    pairs = dict((k.lower(), v.strip().strip('"')) for k, v in re.findall(r"(\w+)\s*=\s*(\"[^\"]*\"|[^\s,;]+)", t))
    if "age" not in pairs:
        return None
    prof = dict(DEFAULTS)
    for k, v in pairs.items():
        if k not in FIELDS:
            continue
        key = {"glucose": "avg_glucose_level", "residence_type": "Residence_type", "smoking": "smoking_status"}.get(k, k)
        if key == "smoking_status":
            v = SMOKING.get(v.lower(), v)
        elif key == "gender":
            v = v.capitalize()
        prof[key] = FIELDS[k](v) if FIELDS[k] is not str else v
    return prof


def case_summary(result: dict) -> dict:
    """De-identified summary of a pipeline result (what the chat, API and Slack bot show and what follow-ups see)."""
    risk = result.get("risk") or {}
    final = result.get("final") or {}
    ev = (result.get("evidence") or {}).get("findings", [])
    return {"run_id": result.get("run_id"), "status": result.get("status"), "risk_percent": risk.get("risk_percent"),
            "risk_band": risk.get("risk_band"), "top_factors": (result.get("analyst") or {}).get("top_factors", [])[:5],
            "cohort_comparison": (result.get("analyst") or {}).get("cohort_comparison"),
            "findings": [{"claim": f["claim"], "citation_id": f["citation_id"]} for f in ev][:8],
            "failed_checks": (final.get("verification") or {}).get("failed", []),
            "needs_human_approval": final.get("needs_human_approval", False), "approval_reasons": final.get("approval_reasons", []),
            "confidence": (result.get("confidence") or {}).get("score"), "latency_s": result.get("latency_s"),
            "cost_usd": (result.get("cost") or {}).get("cost_usd"), "receipt_signature": (result.get("receipt") or {}).get("signature"),
            "guardrail": (result.get("input_guardrail") or {}).get("reason")}


def summary_text(s: dict) -> str:
    if s["status"] == "blocked":
        return f"Request blocked by the input guardrail ({s['guardrail']}). Receipt {str(s['receipt_signature'])[:16]}..."
    lines = [f"**Status:** {s['status']} · **model risk {s['risk_percent']}% ({s['risk_band']})** · confidence {s['confidence']}"]
    if s["top_factors"]:
        lines.append("**Main factors:** " + "; ".join(map(str, s["top_factors"][:4])))
    lines += [f"- {f['claim']} [{f['citation_id']}]" for f in s["findings"][:4]]
    if s["needs_human_approval"]:
        lines.append("Clinician sign-off required: " + ", ".join(s["approval_reasons"]))
    if s["failed_checks"]:
        lines.append("Failed checks: " + ", ".join(s["failed_checks"]))
    lines.append(f"Run `{str(s['run_id'])[:8]}` · {s['latency_s']} s · ${s['cost_usd']} · receipt `{str(s['receipt_signature'])[:16]}...`")
    lines.append(f"_{DISCLAIMER}_")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- follow-ups
def output_checks(answer: str, retrieved: dict) -> dict:
    low = answer.lower()
    cited = sorted(set(re.findall(r"\[(G-\d{2})\]", answer)))
    return {"no_dosing_advice": not any(re.search(p, low) for p in DOSING_PATTERNS),
            "no_diagnosis_claim": not any(re.search(p, low) for p in DIAGNOSIS_PATTERNS),
            "citations_were_retrieved": all(c in retrieved for c in cited), "cited": cited}


async def afollowup(question: str, case: dict | None, session_id: str, consent: bool = False, llm=None) -> dict:
    t0 = time.perf_counter()
    mem = get_memory()
    gate = check_input(question)
    ctx = RunContext(session_id=session_id, consent=consent)
    if not gate["allowed"]:
        answer = gate["detail"]
        mem.add(session_id, "user", question, consent)
        mem.add(session_id, "assistant", answer, consent)
        body = {"kind": "followup", "run_id": ctx.run_id, "contract_sha256": ctx.contract_sha256, "status": "blocked",
                "guardrail": gate["reason"], "question_sha256": sha(gate["question"]), "answer_sha256": sha(answer)}
        return {"status": "blocked", "answer": answer, "guardrail": gate["reason"], "receipt": sign_receipt(body), "llm_calls": 0}
    history = mem.as_context(session_id)
    n_hist = len(mem.recall(session_id))
    sink: dict = {}
    async def run(model):
        async with mcp_session() as session:
            tools = {t.name: t for t in await load_mcp_tools(session)}
            consult = make_evidence_agent_tool(model, tools, ctx, sink)
            bound = model.bind_tools([consult])
            msgs = [SystemMessage(FOLLOWUP_SYSTEM),
                    HumanMessage(f"CASE SUMMARY (de-identified): {json.dumps(case or {'note': 'no case has been run yet'}, default=str)[:2500]}\n"
                                 f"CONVERSATION (recalled, PHI-redacted):\n{history}\n\nFOLLOW-UP QUESTION: {gate['question']}")]
            for _ in range(3):
                w = time.time()
                ai: AIMessage = await bound.ainvoke(msgs)
                ctx.record_llm("followup_agent", ai, w, time.time())
                msgs.append(ai)
                if not ai.tool_calls:
                    return ai.content if isinstance(ai.content, str) else str(ai.content)
                for tc in ai.tool_calls:
                    d = ctx.perms.pre_tool_use("followup_agent", tc["name"], tc["args"])
                    out = await consult.ainvoke(d.args) if d.allow and tc["name"] == consult.name else json.dumps({"error": d.reason})
                    msgs.append(ToolMessage(content=out, tool_call_id=tc["id"]))
            msgs.append(HumanMessage("Answer now in at most 120 words using the tool results above."))
            ai = await model.ainvoke(msgs)
            ctx.record_llm("followup_agent", ai)
            return ai.content
    if llm is None:
        async with httpx.AsyncClient(timeout=httpx.Timeout(ctx.contract["llm_timeout_s"])) as hc:
            answer = await asyncio.wait_for(run(get_llm(http_async_client=hc)), timeout=180)
    else:
        answer = await asyncio.wait_for(run(llm), timeout=180)
    checks = output_checks(answer, sink.get("retrieved", {}))
    status = "answered" if all(v for k, v in checks.items() if k != "cited") else "withheld"
    if status == "withheld":
        failed = [k for k, v in checks.items() if k != "cited" and not v]
        answer = f"The draft answer failed the output checks ({', '.join(failed)}) and was withheld. Please ask your clinician."
    if DISCLAIMER.lower() not in answer.lower():
        answer = answer.rstrip() + f"\n\n_{DISCLAIMER}_"
    mem.add(session_id, "user", question, consent, run_id=ctx.run_id)
    mem.add(session_id, "assistant", answer, consent, run_id=ctx.run_id)
    body = {"kind": "followup", "run_id": ctx.run_id, "case_run_id": (case or {}).get("run_id"), "contract_sha256": ctx.contract_sha256,
            "model": ctx.contract["model"], "temperature": ctx.contract["temperature"], "top_p": ctx.contract.get("top_p"),
            "status": status, "checks": checks, "question_sha256": sha(gate["question"]), "answer_sha256": sha(answer),
            "audit_chain_head": ctx.audit.entries[-1]["hash"], "audit_chain_valid": ctx.audit.verify_chain(),
            "counts": {"llm_calls": ctx.llm_calls, "tokens": ctx.tokens, "tool_calls": ctx.perms.tool_calls},
            "cost_usd": round(ctx.cost_usd, 6), "history_turns_recalled": n_hist}
    return {"status": status, "answer": answer, "checks": checks, "citations": checks["cited"], "retrieved": sink.get("retrieved", {}),
            "literature": sink.get("literature", [])[:3], "trace": sink.get("trace", []), "a2a": ctx.bus.log, "audit": ctx.audit.entries,
            "llm_calls": ctx.llm_calls, "tool_calls": ctx.perms.tool_calls, "cost_usd": round(ctx.cost_usd, 6),
            "latency_s": round(time.perf_counter() - t0, 2), "receipt": sign_receipt(body), "run_id": ctx.run_id}


def followup(question: str, case: dict | None, session_id: str, consent: bool = False, llm=None, timeout: float = 200) -> dict:
    from .pipeline import _background_loop
    return asyncio.run_coroutine_threadsafe(afollowup(question, case, session_id, consent, llm), _background_loop()).result(timeout=timeout)


def handle_message(text: str, session_id: str, case: dict | None = None, consent: bool = False, run=None, follow=None,
                   rag_mode: str | None = None) -> dict:
    """One chat turn -> {"kind": "case"|"followup"|"error", "reply": markdown, "case": summary?, "result": raw?}."""
    try:
        prof = parse_case_request(text)
    except ValueError as exc:
        return {"kind": "error", "reply": str(exc)}
    if prof is not None:
        if run is None:
            from .pipeline import run_case as run
        mem = get_memory()
        mem.add(session_id, "user", text, consent)
        res = run(prof, "Explain this patient's stroke risk and what to discuss at the next visit.", "", "clinician",
                  session_id, consent, rag_mode=rag_mode)
        s = case_summary(res)
        reply = summary_text(s)
        mem.add(session_id, "assistant", reply, consent, run_id=res.get("run_id"))
        return {"kind": "case", "reply": reply, "case": s, "result": res}
    out = (follow or followup)(text, case, session_id, consent)
    return {"kind": "followup", "reply": out["answer"], "followup": out}
