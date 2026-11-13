"""LangGraph multi-agent system: an LLM orchestrator loops over five specialist agents.

Graph (exactly what ``build_graph`` wires):

    START -> planner (LLM plan, informed by episodic memory + learned hints)
          -> orchestrator --(LLM: next agent + skill + instruction)--> intake | risk_analyst |
             ^                                                         evidence_researcher | clinical_writer | safety_reviewer
             +------------------- every specialist returns to the orchestrator --------------------+
    orchestrator --FINISH--> gate (confidence score + policy gate, episode write) -> END

Planner -> executor: the planner writes an explicit step plan; the orchestrator executes it turn by turn (it sees
the plan and the next planned step) and may deviate; plan adherence is measured.

Harness loop inside each tool-using specialist (Yao et al., 2023, ReAct): context -> LLM (Thought + tool_calls)
-> pre-tool-use hook (allow/deny) -> MCP result (PHI-redacted, token-budgeted) -> back into context (Observation),
at most ``loop.max_tool_rounds_per_agent`` rounds, then a structured final answer. Every step is recorded as a
``react_step`` (thought / action / observation).

Agentic RAG: after its first pass the Evidence Researcher's coverage of the patient's required topics is
measured; if a topic is missing it gets one more round with an explicit "reformulate and re-retrieve" task (it
may also traverse the knowledge graph). Reflection: the Clinical Writer self-critiques its draft (code checks +
LLM critique) and revises once before the Safety Reviewer sees it.

Code-enforced orchestration rules (each override is logged in the trace):
 (1) turn cap and budget from harness_contract.json; (2) the LLM may choose among agents whose inputs are
 ready - intake, risk_analyst and evidence_researcher in any order, then clinical_writer, then safety_reviewer;
 (3) FINISH is accepted only after the reviewer has checked the latest draft; (4) after one revision the run
 finishes; (5) per-agent visit cap. Delegation happens through A2A-style tasks on the in-process bus.
"""
from __future__ import annotations

import json
import operator
import re
import time
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import confidence as conf
from .a2a import AGENT_DESCRIPTIONS
from .context import assemble_evidence, count as count_tokens, messages_tokens, truncate
from .data import validate_profile
from .guardrails import redact
from .harness import RunContext, section, verify
from .skills import catalog_for, load_body

SPECIALISTS = ["intake", "risk_analyst", "evidence_researcher", "clinical_writer", "safety_reviewer"]
GATHER = ["intake", "risk_analyst", "evidence_researcher"]
DEFAULT_SKILL = {"intake": "note-screening", "risk_analyst": "cohort-comparison", "evidence_researcher": "guideline-citation",
                 "clinical_writer": "risk-communication", "safety_reviewer": "red-team-review"}


class CaseState(TypedDict, total=False):
    plan: Optional[dict]
    learning: Optional[dict]
    coverage: Optional[dict]
    self_critique: Optional[dict]
    question: str
    note: str
    audience: str
    profile: dict
    intake: Optional[dict]
    analyst: Optional[dict]
    risk: Optional[dict]               # authoritative predict_risk output for the exact profile
    evidence: Optional[dict]
    retrieved: dict                    # id -> full guideline record (read via MCP resource)
    draft: str
    pending_review: bool
    review: Optional[dict]
    revisions: int
    turns: int
    visits: dict
    next: str
    task: dict
    skill: Optional[str]
    final: Optional[dict]
    evidence_log: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]
    route: Annotated[list, operator.add]


# ----------------------------------------------------------------------------- structured outputs
class PlanStep(BaseModel):
    agent: Literal["intake", "risk_analyst", "evidence_researcher", "clinical_writer", "safety_reviewer"]
    goal: str = Field(description="what this step must produce, one sentence")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(description="ordered steps; gather (intake, risk_analyst, evidence_researcher) before clinical_writer, then safety_reviewer")
    watch_outs: list[str] = Field(default_factory=list, description="up to 3 risks to watch, e.g. from similar past episodes or learned hints")
    rationale: str = Field(description="one or two sentences")


class CritiqueItem(BaseModel):
    rule: Literal["missing_citation", "medication_or_dose", "diagnosis_claim", "risk_band_missing", "disclaimer_missing"]
    quote: str = Field(description="the exact offending sentence copied verbatim from the draft (empty only for *_missing rules)")


class SelfCritique(BaseModel):
    problems: list[CritiqueItem] = Field(default_factory=list, description="clear violations only, max 5; empty if none")


class Route(BaseModel):
    next: Literal["intake", "risk_analyst", "evidence_researcher", "clinical_writer", "safety_reviewer", "FINISH"]
    skill: Optional[str] = Field(default=None, description="one skill name from the chosen agent's catalog, or null")
    instruction: str = Field(description="one or two sentences telling the agent what to do")
    reason: str = Field(description="one short sentence")


class IntakeOut(BaseModel):
    red_flag_signs: list[str] = Field(description="sign names returned by screen_note, empty if none")
    neuro_note_probability: Optional[float] = Field(default=None, description="copied from screen_note, null if no note")
    prior_cases: str = Field(description="earlier risk results from memory_read, or 'no earlier cases in this session'")
    summary: str = Field(description="two sentences, facts from the tools only")


class RiskOut(BaseModel):
    risk_probability: float = Field(description="copied exactly from predict_risk")
    risk_percent: float = Field(description="copied exactly from predict_risk")
    risk_band: Literal["low", "elevated", "high"]
    top_factors: list[str] = Field(description="'feature (raises|lowers)' from explain_factors, most influential first")
    cohort_comparison: str = Field(description="the 'headline' sentence from cohort_stats, copied verbatim")
    calculations: list[str] = Field(default_factory=list, description="calculator expressions and results used")


class Finding(BaseModel):
    claim: str = Field(description="one short sentence (<= 25 words) paraphrasing one retrieved excerpt")
    citation_id: str = Field(description="the excerpt id, e.g. G-03")


class EvidenceOut(BaseModel):
    queries: list[str]
    findings: list[Finding] = Field(description="3-6 findings relevant to this patient's factors and to warning signs")


class Issue(BaseModel):
    rule: int = Field(description="number of the violated rule, 1-8")
    quote: str = Field(description="the exact offending sentence copied from the draft")
    explanation: str


class RedTeam(BaseModel):
    verdict: Literal["pass", "fail"]
    issues: list[Issue] = Field(default_factory=list)


# LLM proposes, code confirms: for rules with an observable surface form, the quoted sentence must show it.
RULE_CONFIRM = {
    1: r"\b(mg|mcg|dose|dosage|tablet|pill|aspirin|statin|warfarin|apixaban|clopidogrel|metformin|insulin|start|stop)\b",
    2: r"\b(has had|had a stroke|is having|suffered|diagnos\w*|confirm\w*)\b",
    3: r"\[G-\d{2}\]",
    4: r"\b(certain\w*|guarantee\w*|definite\w*|will (have|suffer|get)|true risk|incidence)\b",
}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", t.lower()).strip()


def confirm_red_team(red: dict, draft: str) -> dict:
    """Keep only LLM red-team issues whose quote really occurs in the draft (drops hallucinated quotes)."""
    d = _norm(draft)
    kept, dropped = [], []
    for i in red.get("issues", []):
        q = _norm(i["quote"])
        confirm = RULE_CONFIRM.get(i["rule"])
        ok = 1 <= i["rule"] <= 8 and len(q) >= 12 and q in d and (confirm is None or re.search(confirm, i["quote"], re.I))
        (kept if ok else dropped).append(i)
    return {"llm_verdict": red["verdict"], "verdict": "fail" if kept else "pass",
            "issues": [f"rule {i['rule']}: \"{i['quote'][:160]}\" - {i['explanation'][:160]}" for i in kept],
            "dropped_unverifiable": len(dropped)}


SYSTEM = {
    "planner": ("You are the planner of NeuroGuard, a stroke-risk decision-support system (not a medical device). Write a short "
                "step plan for the specialist agents, using similar past episodes and learned hints to anticipate problems."),
    "orchestrator": ("You are the orchestrator of NeuroGuard, a stroke-risk clinical decision-support system (not a medical device). "
                     "You delegate work to specialist agents and decide when the case is complete. Choose the next agent from the "
                     "ELIGIBLE list, optionally one skill from that agent's catalog, and write a short instruction. Choose FINISH only "
                     "when the safety reviewer has approved the latest draft."),
    "intake": ("You are the Intake agent. Call screen_note on the note (if a note is given) and memory_read once. "
               "Report only what the tools return."),
    "risk_analyst": ("You are the Risk Analyst agent. Call predict_risk and explain_factors with the profile values exactly as given "
                     "(same spelling, bmi null if unknown), then cohort_stats. Use calculator for any arithmetic. Copy numbers exactly."),
    "evidence_researcher": ("You are the Evidence Researcher agent. Search the guideline corpus with search_guidelines (hybrid keyword + "
                            "vector search) and/or graph_traverse (knowledge graph from risk factors to excerpts) and return findings "
                            "that are supported by a retrieved excerpt, each with that excerpt's id. Work in ReAct style: before every "
                            "tool call write one short line starting with 'Thought:' that says what you still need and why this call. "
                            "Only cite excerpts that search_guidelines returned to you."),
    "clinical_writer": ("You are the Clinical Writer agent of a decision-support tool. Follow the report template exactly and use only the "
                        "facts and evidence supplied."),
    "safety_reviewer": "You are the Safety Reviewer agent. Follow the review instructions and return a verdict.",
}


def _text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):
        return "\n".join(r.get("text", str(r)) if isinstance(r, dict) else getattr(r, "text", str(r)) for r in result)
    return str(result)


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def factors_present(p: dict) -> list[str]:
    f = []
    if p["age"] >= 55:
        f.append("age 55 or older")
    if p["hypertension"]:
        f.append("high blood pressure")
    if p["heart_disease"]:
        f.append("heart disease")
    if p["avg_glucose_level"] >= 126:
        f.append("high average glucose / diabetes range")
    if p.get("bmi") and p["bmi"] >= 30:
        f.append("obesity (BMI 30 or more)")
    if p["smoking_status"] in ("smokes", "formerly smoked"):
        f.append(f"tobacco ({p['smoking_status']})")
    return f or ["no major listed factor; general prevention"]


WARNING_TOPICS = {"emergency", "tia"}


def profile_topics(p: dict) -> list[str]:
    """Guideline topics this patient's evidence must cover (risk factors present + warning signs)."""
    t = []
    if p["age"] >= 55:
        t.append("age")
    if p["hypertension"]:
        t.append("hypertension")
    if p["heart_disease"]:
        t.append("heart_disease")
    if p["avg_glucose_level"] >= 126:
        t.append("glucose")
    if p.get("bmi") and p["bmi"] >= 30:
        t.append("bmi")
    if p["smoking_status"] in ("smokes", "formerly smoked"):
        t.append("smoking")
    return (t or ["lifestyle"]) + ["emergency"]


def coverage_of(topics: list[str], findings: list[dict], retrieved: dict) -> dict:
    covered = {(retrieved.get(f["citation_id"]) or {}).get("topic") for f in findings}
    cov = [t for t in topics if (t in covered) or (t == "emergency" and covered & WARNING_TOPICS)]
    miss = [t for t in topics if t not in cov]
    return {"required": topics, "covered": cov, "missing": miss, "score": round(len(cov) / len(topics), 3) if topics else 1.0}


def next_required(s: CaseState, max_revisions: int) -> Optional[str]:
    for a in GATHER:
        if not s.get({"intake": "intake", "risk_analyst": "analyst", "evidence_researcher": "evidence"}[a]):
            return a
    if not s.get("draft"):
        return "clinical_writer"
    if s.get("pending_review"):
        return "safety_reviewer"
    rv = s.get("review") or {}
    if rv.get("verdict") == "revise" and s.get("revisions", 0) <= max_revisions:
        return "clinical_writer"
    return None


def eligible(s: CaseState, max_revisions: int) -> list[str]:
    missing = [a for a, k in zip(GATHER, ["intake", "analyst", "evidence"]) if not s.get(k)]
    if missing:
        return missing
    r = next_required(s, max_revisions)
    return [r] if r else ["FINISH"]


# ----------------------------------------------------------------------------- graph
def build_graph(llm, tools: list, session, ctx: RunContext):
    tool_map = {t.name: t for t in tools}
    C = ctx.contract
    L = C["loop"]

    async def call_llm(agent: str, runnable, msgs):
        if ctx.limiter is not None:                 # batch mode: shared token-bucket rate limit across runs
            await ctx.limiter.acquire()
        t_start = time.time()
        out = await runnable.ainvoke(msgs)
        raw = out["raw"] if isinstance(out, dict) and "raw" in out else out
        ctx.record_llm(agent, raw, t_start, time.time())
        return out

    async def structured(agent: str, schema, msgs):
        out = await call_llm(agent, llm.with_structured_output(schema, include_raw=True), msgs)
        if out.get("parsed") is None:
            raise RuntimeError(f"{agent}: structured output failed: {out.get('parsing_error')}")
        return out["parsed"]

    def system_for(agent: str, skill: Optional[str]) -> tuple[str, list]:
        sys_prompt, tr = SYSTEM[agent], []
        if skill:
            body = load_body(skill, agent)
            sys_prompt += f"\n\n# Skill loaded: {skill}\n{body}"
            tr.append({"agent": agent, "type": "skill_loaded", "skill": skill, "chars_injected": len(body)})
            ctx.audit.log(agent, "skill_loaded", skill, "ok", chars=len(body))
        return sys_prompt, tr

    async def react(agent: str, s: CaseState, human: str):
        """The harness agent loop: context -> LLM -> tool call -> hook -> result -> back to context."""
        t0 = time.perf_counter()
        sys_prompt, trace = system_for(agent, s.get("skill"))
        allowed = [tool_map[n] for n in ctx.perms.allowed_tools(agent) if n in tool_map]
        bound = llm.bind_tools(allowed)
        task_text = s["task"]["history"][0]["parts"][0]["text"]
        msgs = [SystemMessage(sys_prompt), HumanMessage(f"TASK FROM ORCHESTRATOR: {task_text}\n\n{human}")]
        evidence = []
        cx = C["context"]
        truncated = compacted = 0
        for step in range(1, L["max_tool_rounds_per_agent"] + 1):
            if not ctx.budget_left():
                trace.append({"agent": agent, "type": "budget_stop"})
                break
            if messages_tokens(msgs) > cx["agent_context_tokens"]:      # compact the oldest observations first
                for m in msgs:
                    if isinstance(m, ToolMessage) and count_tokens(m.content) > 200 and messages_tokens(msgs) > cx["agent_context_tokens"]:
                        m.content, _ = truncate(m.content, 200)
                        compacted += 1
            ai: AIMessage = await call_llm(agent, bound, msgs)
            msgs.append(ai)
            thought = (ai.content or "").strip() if isinstance(ai.content, str) else ""
            if not ai.tool_calls:
                if thought:
                    trace.append({"agent": agent, "type": "react_step", "step": step, "thought": thought[:400], "action": "final_answer"})
                break
            for tc in ai.tool_calls:
                t1, w1 = time.perf_counter(), time.time()
                d = ctx.perms.pre_tool_use(agent, tc["name"], tc["args"])
                if not d.allow or tc["name"] not in tool_map:
                    out, n_red = json.dumps({"error": f"blocked by pre-tool-use hook: {d.reason}"}), []
                else:
                    try:
                        out, n_red = redact(_text(await tool_map[tc["name"]].ainvoke(d.args)))
                    except Exception as exc:          # tool errors go back to the LLM as data
                        out, n_red = json.dumps({"error": str(exc)[:300]}), []
                ctx.record_tool_span(agent, tc["name"], w1, time.time(), "allow" if d.allow else "deny")
                evidence.append({"agent": agent, "tool": tc["name"], "args": d.args, "output": out, "allowed": d.allow})
                obs, cut = truncate(out, cx["max_tool_output_tokens"])     # budgeted context: the LLM sees at most N tokens
                truncated += int(cut)
                msgs.append(ToolMessage(content=obs, tool_call_id=tc["id"]))
                trace.append({"agent": agent, "type": "mcp_tool_call", "tool": tc["name"],
                              "args": {k: v for k, v in d.args.items() if k not in ("session_id", "boost_factors")}, "decision": "allow" if d.allow else "deny",
                              "hook_reason": d.reason, "phi_redactions": len(n_red), "output_preview": out[:300], "ms": _ms(t1)})
                trace.append({"agent": agent, "type": "react_step", "step": step, "thought": thought[:400] or "(no thought text returned with the tool call)",
                              "action": f"{tc['name']}({json.dumps({k: v for k, v in d.args.items() if k not in ('session_id', 'boost_factors')})[:160]})",
                              "observation": obs[:240]})
        trace.append({"agent": agent, "type": "context", "tokens": messages_tokens(msgs), "budget": cx["agent_context_tokens"],
                      "tool_outputs_truncated": truncated, "observations_compacted": compacted, "window": cx["model_context_window"]})
        msgs.append(HumanMessage("Now return your final answer in the required structure, using only the tool results above."))
        return msgs, evidence, trace, t0

    def finish_task(s: CaseState, agent: str, name: str, data: dict):
        ctx.bus.complete(s["task"], agent, name, data)

    # ------------------------------------------------------------------ planner (planner -> executor)
    async def planner(s: CaseState):
        t0 = time.perf_counter()
        topics = profile_topics(s["profile"])
        lo = int(s["profile"]["age"] // 10 * 10)
        band = f"{lo}-{lo + 9}"
        recall = {"similar_episodes": [], "learned_hints": [], "retrieval_boosts": {}}
        trace = []
        if C["learning"]["enabled"] and "episode_recall" in tool_map:
            d = ctx.perms.pre_tool_use("planner", "episode_recall", {"factors": topics, "age_band": band, "k": C["episodic_memory"]["recall_k"]})
            if d.allow:
                w1 = time.time()
                recall = json.loads(_text(await tool_map["episode_recall"].ainvoke(d.args)))
                ctx.record_tool_span("planner", "episode_recall", w1, time.time(), "allow")
                trace.append({"agent": "planner", "type": "mcp_tool_call", "tool": "episode_recall", "args": d.args, "decision": "allow",
                              "hook_reason": d.reason, "phi_redactions": 0, "output_preview": json.dumps(recall)[:300], "ms": _ms(t0)})
        hints = recall.get("learned_hints", [])[: C["learning"]["max_hints"]]
        if hints:
            ctx.decide("planner", "learning_hints_applied", [h["key"] for h in hints],
                       "feedback of the same kind occurred in >= min_support recent episodes", evidence={h["key"]: h["support"] for h in hints})
        task = ctx.bus.new_task("harness", "planner", "Plan the specialist steps for this case.", data={"topics": topics})
        msg = (f"CASE: audience={s['audience']}; question={s['question']!r}; note given={bool(s['note'].strip())}; evidence topics={topics}\n"
               f"AGENTS: {json.dumps({a: AGENT_DESCRIPTIONS[a] for a in SPECIALISTS})}\n"
               f"SIMILAR PAST EPISODES (de-identified): {json.dumps(recall.get('similar_episodes', []))[:2500]}\n"
               f"LEARNED HINTS: {json.dumps([h['hint'] for h in hints])}\n"
               "Rules: intake, risk_analyst and evidence_researcher first (any order), then clinical_writer, then safety_reviewer.")
        plan = (await structured("planner", Plan, [SystemMessage(SYSTEM["planner"]), HumanMessage(msg)])).model_dump()
        ctx.bus.complete(task, "planner", "plan", plan)
        ctx.decide("planner", "plan", [st["agent"] for st in plan["steps"]], plan["rationale"], options=SPECIALISTS,
                   evidence={"similar_episodes": len(recall.get("similar_episodes", [])), "hints": len(hints)})
        trace.append({"agent": "planner", "type": "plan", "steps": plan["steps"], "watch_outs": plan["watch_outs"], "rationale": plan["rationale"],
                      "similar_episodes": len(recall.get("similar_episodes", [])), "learned_hints": [h["hint"] for h in hints], "ms": _ms(t0)})
        return {"plan": plan, "learning": {"hints": hints, "similar_episodes": recall.get("similar_episodes", []),
                                           "retrieval_boosts": recall.get("retrieval_boosts", {}), "topics": topics, "age_band": band},
                "trace": trace}

    # ------------------------------------------------------------------ orchestrator
    async def orchestrator(s: CaseState):
        t0 = time.perf_counter()
        turns = s.get("turns", 0) + 1
        visits = dict(s.get("visits", {}))
        elig = eligible(s, L["max_revisions"])
        override = None
        if turns > L["max_orchestrator_turns"] or not ctx.budget_left():
            choice = Route(next="FINISH", instruction="stop", reason="turn cap or budget reached")
            override = "turn cap / budget -> FINISH"
        else:
            done = {k: bool(s.get(k)) for k in ("intake", "analyst", "evidence", "draft")}
            cards = {a: {"description": AGENT_DESCRIPTIONS[a], "skills": catalog_for(a)} for a in SPECIALISTS}
            rv = s.get("review") or {}
            planned = [st["agent"] for st in (s.get("plan") or {}).get("steps", [])]
            done_agents = [a for a in s.get("route", []) if a != "FINISH"]
            nxt_planned = next((a for a in planned if a not in done_agents and a in elig), None)
            msg = (f"PLAN (from the planner): {planned}; next planned step that is eligible now: {nxt_planned}\n"
                   f"CASE: audience={s['audience']}; question={s['question']!r}; note given={bool(s['note'].strip())}\n"
                   f"PROGRESS: {json.dumps(done)}; draft awaiting review={s.get('pending_review', False)}; "
                   f"last review verdict={rv.get('verdict')}; issues={rv.get('issues', [])[:4]}; revisions used={s.get('revisions', 0)}\n"
                   f"ELIGIBLE NOW: {elig}\nAGENT CARDS: {json.dumps(cards)}\n"
                   "For clinical_writer pick 'patient-friendly-summary' when audience=patient, otherwise 'risk-communication'.")
            choice = await structured("orchestrator", Route, [SystemMessage(SYSTEM["orchestrator"]), HumanMessage(msg)])
            if choice.next not in elig:
                override = f"LLM chose {choice.next}; not eligible now -> {elig[0]}"
                choice = Route(next=elig[0], skill=None, instruction=choice.instruction, reason=choice.reason)
            if choice.next != "FINISH" and visits.get(choice.next, 0) >= L["max_visits_per_agent"]:
                override = f"visit cap for {choice.next} -> FINISH"
                choice = Route(next="FINISH", instruction="stop", reason="visit cap")
        skill = choice.skill
        if choice.next != "FINISH":
            names = [c["name"] for c in catalog_for(choice.next)]
            if skill not in names:
                default = DEFAULT_SKILL[choice.next]
                if choice.next == "clinical_writer" and s["audience"] == "patient":
                    default = "patient-friendly-summary"
                override = (override or "") + f" skill {skill!r} not in catalog -> {default}"
                skill = default
            visits[choice.next] = visits.get(choice.next, 0) + 1
            task = ctx.bus.new_task("orchestrator", choice.next, choice.instruction, data={"turn": turns}, skill=skill)
        else:
            task, skill = {}, None
        ctx.audit.log("orchestrator", "route", choice.next, "ok" if not override else "override", reason=choice.reason, override=override)
        ctx.decide("orchestrator", "route", choice.next, choice.reason, options=elig, turn=turns, skill=skill, override=override)
        tr = {"agent": "orchestrator", "type": "route", "turn": turns, "next": choice.next, "skill": skill,
              "instruction": choice.instruction, "reason": choice.reason, "override": override, "eligible": elig, "ms": _ms(t0)}
        return {"turns": turns, "visits": visits, "next": choice.next, "skill": skill, "task": task,
                "trace": [tr], "route": [choice.next]}

    # ------------------------------------------------------------------ specialists
    async def intake(s: CaseState):
        human = f"NOTE (PHI-redacted, may be empty): {s['note'] or '(no note)'}\nQUESTION: {s['question']}"
        msgs, ev, trace, t0 = await react("intake", s, human)
        out = (await structured("intake", IntakeOut, msgs)).model_dump()
        finish_task(s, "intake", "intake_summary", out)
        trace.append({"agent": "intake", "type": "agent_output", "output": out, "ms": _ms(t0)})
        return {"intake": out, "evidence_log": ev, "trace": trace}

    async def risk_analyst(s: CaseState):
        card = ""
        if ctx.perms.allowed_resource("risk_analyst", "model://risk-card"):
            r = await session.read_resource("model://risk-card")
            m = json.loads(r.contents[0].text)
            card = json.dumps({"served_model": m["served_model"], "test_metrics": {k: m["test_metrics"][k] for k in ("roc_auc", "brier", "recall", "precision")}})
            ctx.audit.log("risk_analyst", "resource_read", "model://risk-card", "allow")
        human = f"PATIENT PROFILE (pass these exact values): {json.dumps(s['profile'])}\nMODEL CARD (MCP resource): {card}"
        msgs, ev, trace, t0 = await react("risk_analyst", s, human)
        trace.insert(0, {"agent": "risk_analyst", "type": "mcp_resource_read", "uri": "model://risk-card"})
        out = (await structured("risk_analyst", RiskOut, msgs)).model_dump()
        risk = None
        for e in ev:   # authoritative number: a predict_risk call whose arguments equal the entered profile
            if e["tool"] == "predict_risk" and e["allowed"]:
                try:
                    same = validate_profile({**e["args"], "bmi": e["args"].get("bmi")}) == s["profile"]
                    data = json.loads(e["output"])
                    if same and "risk_probability" in data:
                        risk = data
                except (ValueError, json.JSONDecodeError):
                    pass
        if risk is None:
            trace.append({"agent": "risk_analyst", "type": "warning", "detail": "no predict_risk call matched the entered profile"})
        finish_task(s, "risk_analyst", "risk_assessment", out)
        trace.append({"agent": "risk_analyst", "type": "agent_output", "output": out, "ms": _ms(t0)})
        return {"analyst": out, "risk": risk, "evidence_log": ev, "trace": trace}

    def _returned_ids(ev: list) -> set:
        ids = set()
        for e in ev:
            if e["allowed"] and e["tool"] in ("search_guidelines", "graph_traverse"):
                try:
                    data = json.loads(e["output"])
                    rows = data["chunks"] if isinstance(data, dict) else data
                    ids |= {r["id"] for r in rows}
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
        return ids

    async def evidence_researcher(s: CaseState):
        topics = (s.get("learning") or {}).get("topics") or profile_topics(s["profile"])
        graph_factors = [t for t in topics if t != "emergency"]
        human = (f"PATIENT FACTORS PRESENT: {factors_present(s['profile'])}\nGRAPH FACTOR NAMES: {graph_factors}\nQUESTION: {s['question']}\n"
                 "Also retrieve the stroke warning-signs excerpt.")
        boosts = (s.get("learning") or {}).get("retrieval_boosts") or {}
        if C["learning"]["enabled"] and boosts:
            ctx.perms.pinned["search_guidelines"] = {"boost_factors": graph_factors}     # harness-pinned learning boost
            ctx.decide("harness", "retrieval_boost", sorted(boosts), "clinician-approved reports for similar factors cited these excerpts",
                       evidence=boosts)
        msgs, ev, trace, t0 = await react("evidence_researcher", s, human)
        out = (await structured("evidence_researcher", EvidenceOut, msgs)).model_dump()
        all_ev, rounds = list(ev), 1

        async def read_cited(findings, retrieved):
            for cid in sorted({f["citation_id"] for f in findings} - set(retrieved)):
                uri = f"guideline://{cid}"
                if ctx.perms.allowed_resource("evidence_researcher", uri):
                    r = await session.read_resource(uri)
                    retrieved[cid] = json.loads(r.contents[0].text)
                    ctx.audit.log("evidence_researcher", "resource_read", uri, "allow")
                    trace.append({"agent": "evidence_researcher", "type": "mcp_resource_read", "uri": uri})

        def split(findings, ids):
            kept, dropped = [], []
            for f in findings:
                (kept if f["citation_id"] in ids else dropped).append(f)
            return kept, dropped

        kept, dropped = split(out["findings"], _returned_ids(all_ev))
        retrieved = dict(s.get("retrieved") or {})
        await read_cited(kept, retrieved)
        cov = coverage_of(topics, kept, retrieved)
        trace.append({"agent": "evidence_researcher", "type": "evidence_sufficiency", "round": 1, **cov,
                      "action": "re-retrieve" if cov["missing"] else "sufficient"})
        # agentic RAG: insufficient evidence -> the agent reformulates and re-retrieves (bounded)
        while cov["missing"] and rounds < L["max_retrieval_rounds"] and ctx.budget_left():
            rounds += 1
            ctx.decide("evidence_researcher", "re_retrieve", cov["missing"], "required evidence topics not covered by any cited excerpt",
                       options=["stop", "re_retrieve"], evidence=cov)
            prev_q = [e["args"].get("query") or e["args"].get("factors") for e in all_ev]
            human2 = (f"EVIDENCE INSUFFICIENT. Missing topics: {cov['missing']}. Previous queries: {prev_q}.\n"
                      "Reformulate: use different wording or synonyms for each missing topic, or call graph_traverse with those "
                      "factor names, then return findings ONLY for the missing topics.")
            msgs2, ev2, tr2, _ = await react("evidence_researcher", s, human2)
            trace += tr2
            out2 = (await structured("evidence_researcher", EvidenceOut, msgs2)).model_dump()
            all_ev += ev2
            k2, d2 = split(out2["findings"], _returned_ids(all_ev))
            seen = {(f["citation_id"], f["claim"]) for f in kept}
            kept += [f for f in k2 if (f["citation_id"], f["claim"]) not in seen]
            dropped += d2
            out["queries"] = out["queries"] + out2["queries"]
            await read_cited(kept, retrieved)
            cov = coverage_of(topics, kept, retrieved)
            trace.append({"agent": "evidence_researcher", "type": "evidence_sufficiency", "round": rounds, **cov,
                          "action": "re-retrieve" if cov["missing"] else "sufficient"})
        ctx.perms.pinned.pop("search_guidelines", None)
        out["findings"], out["dropped_uncited_or_unretrieved"], out["retrieval_rounds"] = kept, dropped, rounds
        finish_task(s, "evidence_researcher", "cited_findings", out)
        trace.append({"agent": "evidence_researcher", "type": "agent_output", "output": out, "ms": _ms(t0)})
        return {"evidence": out, "retrieved": retrieved, "evidence_log": all_ev, "trace": trace, "coverage": {**cov, "rounds": rounds}}

    def facts(s: CaseState) -> dict:
        a = s.get("analyst") or {}
        return {"profile": s["profile"], "question": s["question"], "intake": s.get("intake"),
                "model_risk": s.get("risk"), "top_factors": a.get("top_factors"), "cohort_comparison": a.get("cohort_comparison"),
                "calculations": a.get("calculations")}

    def evidence_payload(s: CaseState) -> list:
        items = [{"id": cid, "title": d["title"], "publisher": d["publisher"], "text": d["text"],
                  "findings": [f["claim"] for f in (s.get("evidence") or {}).get("findings", []) if f["citation_id"] == cid]}
                 for cid, d in (s.get("retrieved") or {}).items()]
        return sorted(items, key=lambda it: -len(it["findings"]))      # most-supported first for budgeted assembly

    async def clinical_writer(s: CaseState):
        t0 = time.perf_counter()
        if not ctx.perms.allowed_prompt("clinical_writer", "clinical_report"):
            raise PermissionError("clinical_writer may not use prompt clinical_report")
        payload, cx = assemble_evidence(evidence_payload(s), C["context"]["writer_evidence_tokens"])
        p = await session.get_prompt("clinical_report", {"audience": s["audience"], "facts_json": json.dumps(facts(s), default=str),
                                                         "evidence_json": json.dumps(payload)})
        ctx.audit.log("clinical_writer", "prompt_get", "clinical_report", "allow")
        sys_prompt, trace = system_for("clinical_writer", s.get("skill"))
        hints = [h["hint"] for h in (s.get("learning") or {}).get("hints", [])]
        if hints:
            sys_prompt += "\n\n# Learned hints (from recent reviewer / clinician feedback; bounded, whitelisted)\n- " + "\n- ".join(hints)
        msgs = [SystemMessage(sys_prompt), HumanMessage(p.messages[0].content.text)]
        trace.append({"agent": "clinical_writer", "type": "context", "tokens": messages_tokens(msgs), "budget": C["context"]["writer_evidence_tokens"],
                      "evidence_assembly": cx, "learned_hints": len(hints), "window": C["context"]["model_context_window"]})
        rv = s.get("review") or {}
        if rv.get("verdict") == "revise":
            msgs.append(HumanMessage("REVISION REQUIRED. Fix every issue below and return the full corrected report.\n- " +
                                     "\n- ".join(rv.get("issues", [])) + "\nPREVIOUS DRAFT:\n" + s.get("draft", "")))
        out = await call_llm("clinical_writer", llm, msgs)
        finish = (out.response_metadata or {}).get("finish_reason")
        if finish in ("content_filter", "length"):
            # Observed in evaluation: the provider's content filter cuts output that copies long guideline passages.
            trace.append({"agent": "clinical_writer", "type": "warning", "detail": f"generation stopped early (finish_reason={finish}); retrying once with paraphrase instruction"})
            ctx.audit.log("clinical_writer", "llm_truncated", finish, "retry")
            msgs.append(HumanMessage("Your previous answer was cut off by the provider (finish_reason=" + finish + "). Write the complete report again. "
                                     "Paraphrase every guideline finding in one short sentence of your own words; do not copy long passages."))
            out = await call_llm("clinical_writer", llm, msgs)
            finish = (out.response_metadata or {}).get("finish_reason")
        draft = out.content
        # reflection / self-critique before the Safety Reviewer: deterministic checks + LLM critique, one self-revision
        sc = {"code_issues": [], "llm_problems": [], "revised": False}
        if not s.get("self_critique") and ctx.budget_left():
            v0 = verify(draft, s.get("risk"), s.get("analyst"), s.get("retrieved") or {}, evidence_text(s), None, ctx)
            sc["code_issues"] = [c for c in v0["failed"] if c not in ("red_team_pass", "within_budget")]
            try:
                crit = await structured("clinical_writer", SelfCritique, [
                    SystemMessage("You are reviewing your own draft. Flag ONLY clear violations of these rules, quoting the exact sentence: "
                                  "missing_citation = a bullet in the Guideline evidence section without a [G-xx] tag; medication_or_dose = a "
                                  "medicine name or dose; diagnosis_claim = says the patient has/had/is having a stroke; risk_band_missing = the "
                                  "model risk percentage is never given with its band; disclaimer_missing = no 'decision support, not a medical "
                                  "device' disclaimer. Mentioning risk factors is NOT a diagnosis. Style preferences are not problems. "
                                  "Return an empty list if the draft is fine."),
                    HumanMessage(f"DRAFT:\n{draft}")])
            except RuntimeError as exc:            # a failed critique never blocks the run; the Safety Reviewer still runs
                crit = SelfCritique(problems=[])
                sc["llm_error"] = str(exc)[:200]
            # keep only critiques we can verify: quoted text must really be in the draft (filters hallucinated problems)
            norm = lambda x: re.sub(r"\s+", " ", x).strip().lower()
            dn = norm(draft)
            ev_sec = norm(section(draft, "Guideline evidence"))
            cue = {"missing_citation": lambda q: "[g-" not in q and q in ev_sec,
                   "medication_or_dose": lambda q: bool(re.search(r"\d+(\.\d+)?\s?(mg|mcg|ml|units?)\b|aspirin|statin|warfarin|clopidogrel|metformin|insulin|lisinopril|amlodipine", q)),
                   "diagnosis_claim": lambda q: bool(re.search(r"\b(has|had|is having|suffered|experienced|diagnosed with)\s+(a\s+)?(stroke|tia)\b", q))}
            # LLM proposes, code verifies: the quote must be verbatim in the draft AND show the rule's lexical cue
            # (*_missing rules are left to the deterministic checks)
            sc["llm_problems"] = [f"{c.rule}: \"{c.quote[:160]}\"" for c in crit.problems[:5]
                                  if c.rule in cue and len(c.quote) > 15 and norm(c.quote) in dn and cue[c.rule](norm(c.quote))]
            sc["llm_flagged_unverified"] = len(crit.problems) - len(sc["llm_problems"])
            problems = [f"failed check: {c}" for c in sc["code_issues"]] + sc["llm_problems"]
            if problems and ctx.budget_left() and L["max_self_revisions"] > 0:
                msgs.append(AIMessage(draft))
                msgs.append(HumanMessage("SELF-CRITIQUE found these problems. Fix them and return the full corrected report:\n- " + "\n- ".join(problems)))
                out2 = await call_llm("clinical_writer", llm, msgs)
                if (out2.response_metadata or {}).get("finish_reason") not in ("content_filter", "length") and out2.content.strip():
                    draft, sc["revised"] = out2.content, True
            ctx.decide("clinical_writer", "self_critique", "revise" if sc["revised"] else "keep", "; ".join(problems)[:300] or "no problems found",
                       options=["keep", "revise"], evidence={"code_issues": sc["code_issues"], "llm_problems": len(sc["llm_problems"])})
            trace.append({"agent": "clinical_writer", "type": "self_critique", **sc})
        trace.insert(0, {"agent": "clinical_writer", "type": "mcp_prompt_get", "prompt": "clinical_report"})
        finish_task(s, "clinical_writer", "report_draft", {"chars": len(draft), "revision": s.get("revisions", 0)})
        trace.append({"agent": "clinical_writer", "type": "generation", "chars": len(draft), "finish_reason": finish, "revision": s.get("revisions", 0), "ms": _ms(t0)})
        return {"draft": draft, "pending_review": True, "trace": trace, "self_critique": s.get("self_critique") or sc}

    def evidence_text(s: CaseState) -> str:
        return json.dumps({"profile": s["profile"], "intake": s.get("intake"), "risk": s.get("risk"),
                           "tools": [e["output"] for e in s.get("evidence_log", [])],
                           "retrieved": [d["text"] for d in (s.get("retrieved") or {}).values()]}, default=str)

    async def safety_reviewer(s: CaseState):
        t0 = time.perf_counter()
        sys_prompt, trace = system_for("safety_reviewer", s.get("skill"))
        p = await session.get_prompt("red_team_review", {"report_md": s["draft"], "evidence_json": json.dumps(
            {"model_and_cohort_facts": facts(s), "guideline_excerpts": evidence_payload(s)}, default=str)})
        ctx.audit.log("safety_reviewer", "prompt_get", "red_team_review", "allow")
        red_raw = (await structured("safety_reviewer", RedTeam, [SystemMessage(sys_prompt), HumanMessage(p.messages[0].content.text)])).model_dump()
        red = confirm_red_team(red_raw, s["draft"])
        v = verify(s["draft"], s.get("risk"), s.get("analyst"), s.get("retrieved") or {}, evidence_text(s), red, ctx)
        issues = [f"failed check: {c}" for c in v["failed"] if c != "red_team_pass"]
        d = v["details"]
        if d["uncited_bullets"]:
            issues.append("uncited evidence bullets: " + " | ".join(b[:80] for b in d["uncited_bullets"][:3]))
        if d["ungrounded_numbers"]:
            issues.append(f"numbers not found in any tool output or excerpt: {d['ungrounded_numbers'][:6]}")
        if d["invalid_citations"]:
            issues.append(f"citations never retrieved: {d['invalid_citations']}")
        if not s.get("risk"):
            issues.append("risk number is not backed by a predict_risk call on the entered profile")
        issues += [f"red team: {i}" for i in red["issues"]][:5]
        revisions = s.get("revisions", 0)
        verdict = "approve" if v["passed"] else ("revise" if revisions < L["max_revisions"] else "approve_with_warnings")
        review = {"verdict": verdict, "issues": issues, "verification": v, "red_team": red}
        ctx.decide("safety_reviewer", "review", verdict, "; ".join(issues)[:300] or "all required checks passed",
                   options=["approve", "revise", "approve_with_warnings"], evidence={"failed": v["failed"], "red_team_kept": len(red["issues"]),
                                                                                    "red_team_dropped": red["dropped_unverifiable"]})
        finish_task(s, "safety_reviewer", "review", {"verdict": verdict, "failed": v["failed"]})
        trace += [{"agent": "safety_reviewer", "type": "mcp_prompt_get", "prompt": "red_team_review"},
                  {"agent": "safety_reviewer", "type": "review", "verdict": verdict, "failed_checks": v["failed"],
                   "issues": issues, "ms": _ms(t0)}]
        return {"review": review, "pending_review": False, "trace": trace, "revisions": revisions + (1 if verdict == "revise" else 0)}

    # ------------------------------------------------------------------ finalize gate
    async def gate(s: CaseState):
        t0 = time.perf_counter()
        rv = s.get("review") or {}
        red = rv.get("red_team")
        if s.get("draft") and not s.get("pending_review"):
            v = verify(s["draft"], s.get("risk"), s.get("analyst"), s.get("retrieved") or {}, evidence_text(s), red, ctx)
        else:
            v = {"checks": {}, "failed": ["draft_not_reviewed"], "passed": False, "details": {}}
        finalized = v["passed"] and s.get("risk") is not None
        band = (s.get("risk") or {}).get("risk_band")
        # ---- confidence score + policy gate
        card = {}
        if ctx.perms.allowed_resource("harness", "model://risk-card"):
            card = json.loads((await session.read_resource("model://risk-card")).contents[0].text)
        cohort_n = None
        for e in s.get("evidence_log", []):
            if e["tool"] == "cohort_stats" and e["allowed"]:
                m = re.search(r"(\d+) of (\d+) had", e["output"])
                cohort_n = int(m.group(2)) if m else cohort_n
        checks = v.get("checks") or {}
        req = [c for c in C["required_checks"] if c != "red_team_pass"]
        comps = {"calibration": conf.calibration_component((s.get("risk") or {}).get("risk_probability"), card.get("calibration_curve", {})),
                 "cohort": round(min(1.0, (cohort_n or 0) / 100), 3),
                 "coverage": (s.get("coverage") or {}).get("score", 0.0),
                 "verifier": round(sum(bool(checks.get(c)) for c in req) / len(req), 3) if checks else 0.0,
                 "red_team": 1.0 if (red or {}).get("verdict") == "pass" else 0.5,
                 "grounding": conf.grounding_support(s.get("draft", ""), s.get("retrieved") or {}, section) or 0.0}
        cscore = conf.score(comps, C["confidence"]["weights"])
        cgate = conf.gate_decision(cscore, C["confidence"])
        ctx.decide("harness", "confidence_gate", cgate, f"confidence {cscore} vs refuse<{C['confidence']['refuse_below']} / review<{C['confidence']['review_below']}",
                   options=["pass", "require_clinician_review", "refuse"], evidence=comps)
        failed_conf = []
        if finalized and cgate == "refuse":
            finalized, failed_conf = False, ["low_confidence"]
        reasons = []
        if cgate == "require_clinician_review":
            reasons.append(f"confidence {cscore} < {C['confidence']['review_below']}")
        if band == "high":
            reasons.append("risk_band == high")
        if (s.get("intake") or {}).get("red_flag_signs"):
            reasons.append("note_has_red_flag_signs")
        if rv.get("verdict") == "approve_with_warnings":
            reasons.append("reviewer_approved_with_warnings")
        mem = None
        if finalized:
            args = {"age": s["profile"]["age"], "risk_probability": s["risk"]["risk_probability"], "risk_band": band,
                    "top_factors": [f.split(" (")[0] for f in (s.get("analyst") or {}).get("top_factors", [])][:6],
                    "report_sha256": __import__("hashlib").sha256(s["draft"].encode()).hexdigest()}
            d = ctx.perms.pre_tool_use("harness", "memory_write", args)
            if d.allow:
                mem = json.loads(_text(await tool_map["memory_write"].ainvoke(d.args)))
            else:
                mem = {"skipped": d.reason}
        ctx.decide("harness", "finalize", "finalized" if finalized else "not_finalized",
                   "all required checks passed" if finalized else f"failed: {v['failed'] + failed_conf}",
                   options=["finalized", "not_finalized"], evidence={"approval_reasons": reasons})
        # ---- episodic memory: one de-identified episode per run (feeds the bounded learning loop)
        rv_trace = [t for t in s.get("trace", []) if t.get("type") == "review"]
        sc = s.get("self_critique") or {}
        lrn = s.get("learning") or {}
        episode = {"run_id": ctx.run_id, "age_band": lrn.get("age_band"), "factors": lrn.get("topics", []), "audience": s["audience"],
                   "plan": [st["agent"] for st in (s.get("plan") or {}).get("steps", [])], "route": s.get("route", []),
                   "overrides": sum(1 for t in s.get("trace", []) if t.get("type") == "route" and t.get("override") and "LLM chose" in t["override"]),
                   "retrieval_rounds": (s.get("coverage") or {}).get("rounds", 1),
                   "self_critique": "revised" if sc.get("revised") else ("clean" if sc else None),
                   "review_verdicts": [t["verdict"] for t in rv_trace],
                   "failed_checks": sorted({c for t in rv_trace for c in t.get("failed_checks", []) if c != "red_team_pass"} | set(sc.get("code_issues", []))),
                   "red_team_rules": sorted({int(m) for t in rv_trace for i in t.get("issues", []) for m in re.findall(r"red team: rule (\d)", i)}),
                   "confidence": cscore, "outcome": "finalized" if finalized else "not_finalized",
                   "cited": sorted(set(re.findall(r"\[(G-\d{2})\]", s.get("draft", ""))))}
        ep_res = None
        d = ctx.perms.pre_tool_use("harness", "episode_write", {"episode_json": json.dumps(episode)})
        if d.allow and "episode_write" in tool_map:
            ep_res = json.loads(_text(await tool_map["episode_write"].ainvoke(d.args)))
        ctx.decide("harness", "episode_write", "stored" if ep_res else "skipped", "de-identified run episode for episodic memory / learning loop",
                   evidence={"fields": sorted(episode)})
        final = {"status": "finalized" if finalized else "not_finalized", "verification": {**v, "failed": v["failed"] + failed_conf},
                 "needs_human_approval": bool(reasons) and finalized, "approval_reasons": reasons, "memory": mem,
                 "confidence": {"score": cscore, "gate": cgate, "components": comps, "weights": C["confidence"]["weights"],
                                "refuse_below": C["confidence"]["refuse_below"], "review_below": C["confidence"]["review_below"]},
                 "episode": {"stored": bool(ep_res), **episode}}
        ctx.audit.log("harness", "finalize_gate", "report", final["status"], failed=v["failed"])
        tr = {"agent": "gate", "type": "gate", "status": final["status"], "failed_checks": final["verification"]["failed"], "needs_human_approval": final["needs_human_approval"],
              "approval_reasons": reasons, "memory": mem, "confidence": cscore, "confidence_gate": cgate, "ms": _ms(t0)}
        return {"final": final, "trace": [tr]}

    g = StateGraph(CaseState)
    for name, fn in [("planner", planner), ("orchestrator", orchestrator), ("intake", intake), ("risk_analyst", risk_analyst),
                     ("evidence_researcher", evidence_researcher), ("clinical_writer", clinical_writer),
                     ("safety_reviewer", safety_reviewer), ("gate", gate)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_edge("planner", "orchestrator")
    g.add_conditional_edges("orchestrator", lambda s: s["next"], {**{a: a for a in SPECIALISTS}, "FINISH": "gate"})
    for a in SPECIALISTS:
        g.add_edge(a, "orchestrator")
    g.add_edge("gate", END)
    return g.compile()

