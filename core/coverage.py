"""Agentic-AI knowledge-graph coverage: every node -> where it lives in code -> live evidence from a run.

Used by the Streamlit "Knowledge-graph coverage" tab, docs/KNOWLEDGE_GRAPH_COVERAGE.md and the notebook.
``status`` is honest: "full" = implemented and exercised on every run; "partial" = implemented but limited
(the limitation is written in ``note``).
"""
from __future__ import annotations

import json

NODES = [
    # group, node, status, where, how, note
    ("Foundations", "LLM reasoning", "full", "core/agents.py::build_graph", "gpt-4o-mini (temperature 0) reasons in 7 LLM roles: planner, orchestrator, 3 gatherers, writer, reviewer", ""),
    ("Foundations", "Prompt patterns", "full", "core/agents.py::SYSTEM, mcp_server prompts, skills/*", "role system prompts, ReAct 'Thought:' format, MCP prompt templates, progressive-disclosure skills, structured output", ""),
    ("Foundations", "Tool use / function calling", "full", "core/agents.py::react, mcp_server (11 tools)", "bind_tools function calling to real MCP tools over stdio, every call through the pre-tool-use hook", ""),
    ("Foundations", "Embeddings & vectors", "full", "core/vectors.py::DenseEncoder, core/rag.py::search", "24-dim LSA dense vectors + TF-IDF sparse, weighted hybrid fusion; notebook compares OpenAI text-embedding-3-small", "MCP server uses local LSA (it holds no API key by design)"),
    ("Foundations", "Context windows", "full", "core/context.py, core/agents.py::react", "tiktoken o200k_base counts, tool output truncation, observation compaction, budgeted writer evidence", ""),
    ("Retrieval & Memory", "Classical RAG", "full", "core/rag.py::GuidelineIndex.search", "30 public guideline excerpts, top-k retrieval, citations verified against retrieved text", ""),
    ("Retrieval & Memory", "Graph RAG", "full", "core/graph_rag.py::traverse, MCP graph_traverse", "networkx graph of factors, conditions and chunks; weighted multi-hop traversal returns paths", ""),
    ("Retrieval & Memory", "Agentic RAG", "full", "core/agents.py::evidence_researcher", "agent picks tools/queries, code measures topic coverage, missing topics trigger a reformulated re-retrieval round", ""),
    ("Retrieval & Memory", "Short-term memory", "full", "core/agents.py::CaseState", "LangGraph state carries messages, findings, plan and trace across nodes in one run", ""),
    ("Retrieval & Memory", "Long-term memory", "full", "core/memory.py, MCP memory_read/memory_write", "consent-gated, de-identified SQLite case summaries per session", ""),
    ("Retrieval & Memory", "Episodic memory", "full", "core/episodes.py, MCP episode_recall/episode_write", "one de-identified episode per run (plan, route, verdicts, failures, outcome, clinician decision); Jaccard recall", ""),
    ("Agent patterns", "ReAct loop", "full", "core/agents.py::react", "Thought -> Action (MCP tool) -> Observation steps recorded as react_step events", "gpt-4o-mini often returns a tool call without thought text; missing thoughts are recorded as missing, never invented"),
    ("Agent patterns", "Planner -> executor", "full", "core/agents.py::planner, orchestrator", "planner LLM writes a structured plan from similar episodes; orchestrator executes it under code guards", ""),
    ("Agent patterns", "Multi-agent orchestration", "full", "core/agents.py::build_graph, core/a2a.py", "LangGraph supervisor + 5 specialists + A2A task bus with agent cards", ""),
    ("Agent patterns", "Context graph traversal", "full", "core/graph_rag.py::traverse", "hop-by-hop expansion factor -> chunk / factor -> condition <- chunk with path explanations", ""),
    ("Agent patterns", "Tool selection & routing", "full", "core/agents.py::orchestrator, core/permissions.py", "LLM routes and picks skills/tools; contract allow-lists, arg clamps and pinned args constrain it", ""),
    ("Agent patterns", "Reflection / self-critique", "full", "core/agents.py::clinical_writer, safety_reviewer", "writer self-critique (code checks + quote-verified LLM critique) then independent reviewer + red team", ""),
    ("Production & governance", "Confidence scoring", "full", "core/confidence.py, gate", "weighted calibration, cohort size, coverage, verifier, red team, grounding", "heuristic score, not a calibrated probability of correctness"),
    ("Production & governance", "Policy gates", "full", "core/guardrails.py, core/permissions.py, gate", "input guardrails, pre-tool-use hook, finalize gate, confidence gate, human approval", ""),
    ("Production & governance", "Audit trail design", "full", "core/harness.py::AuditLog, sign_receipt", "SHA-256 hash-chained audit log + HMAC-signed receipt", ""),
    ("Production & governance", "Decision reasoning logs", "full", "core/harness.py::RunContext.decide", "one structured entry per decision: actor, kind, options, chosen, rationale, evidence; hashed into the receipt", ""),
    ("Production & governance", "Eval & observability", "full", "evaluation.ipynb, core/observability.py", "notebook metrics + per-call spans exported as OTLP-JSON", "no live OTLP collector is attached on Streamlit Cloud"),
    ("Production & governance", "Cost / token economics", "full", "core/harness.py::price, cost_summary", "per-call tokens priced with public list rates, per-agent cost, budgets in the contract", "list price; Vocareum billing may differ"),
    ("Frontier", "Memory & learning loops", "partial", "core/episodes.py::learned_hints, retrieval_boosts", "recurring reviewer/clinician feedback becomes whitelisted hints and bounded retrieval boosts", "bounded and reversible by design; no weight updates"),
    ("Frontier", "Self-improving agents", "partial", "core/episodes.py + planner + writer", "agents read their own past failures before acting; effect shown in notebook demo", "prompt-level only; no autonomous code or prompt rewriting"),
    ("Frontier", "Multi-modal reasoning", "partial", "core/multimodal.py", "lab image -> gpt-4o-mini vision structured values; PDF -> pypdf text + regex", "3 fields only, no OCR for scanned PDFs, small rendered test set"),
    ("Frontier", "Agentic workflows at scale", "partial", "core/batch.py", "cohort batch with semaphore concurrency and shared token-bucket rate limit", "single process, max 6 cases, no distributed queue"),
]


def _count(trace, t):
    return sum(1 for e in trace if e.get("type") == t)


def evidence(node: str, r: dict | None) -> str:
    """Live evidence for one node from a pipeline result (empty when no run yet)."""
    if not r or r.get("status") == "blocked":
        return ""
    tr = r.get("trace", [])
    ev = r.get("evidence_log", [])
    tools = [e["tool"] for e in ev]
    cov = r.get("coverage") or {}
    conf = r.get("confidence") or {}
    cost = r.get("cost") or {}
    ctxs = [e for e in tr if e.get("type") == "context"]
    m = {
        "LLM reasoning": f"{r.get('llm_calls')} LLM calls, {r.get('tokens')} tokens",
        "Prompt patterns": f"{_count(tr, 'skill_loaded')} skills loaded, {_count(tr, 'mcp_prompt_get')} MCP prompts, {_count(tr, 'react_step')} ReAct steps",
        "Tool use / function calling": f"{len(ev)} MCP tool calls: {', '.join(sorted(set(tools)))}",
        "Embeddings & vectors": f"{tools.count('search_guidelines')} hybrid searches (sparse+dense)",
        "Context windows": f"{len(ctxs)} context checks, max {max((c.get('tokens', 0) for c in ctxs), default=0)} tokens of 128k; "
                           f"{sum(c.get('tool_outputs_truncated', 0) for c in ctxs)} tool outputs truncated",
        "Classical RAG": f"{len(r.get('retrieved') or {})} excerpts retrieved and read",
        "Graph RAG": f"{tools.count('graph_traverse')} graph traversals",
        "Agentic RAG": f"{cov.get('rounds', 1)} retrieval round(s), coverage {cov.get('score')}" + (f", missing {cov['missing']}" if cov.get("missing") else ""),
        "Short-term memory": f"{len(r.get('trace', []))} state trace events across {len([x for x in r.get('route', []) if x != 'FINISH'])} agent turns",
        "Long-term memory": "memory: " + json.dumps((r.get('final') or {}).get('memory'))[:140],
        "Episodic memory": f"episode stored={bool((r.get('episode') or {}).get('stored'))}; {len((r.get('learning') or {}).get('similar_episodes', []))} similar recalled",
        "ReAct loop": f"{_count(tr, 'react_step')} thought/action/observation steps",
        "Planner -> executor": "plan: " + " -> ".join(s["agent"] for s in (r.get("plan") or {}).get("steps", [])),
        "Multi-agent orchestration": "route: " + " -> ".join(r.get("route", [])) + f"; {len(r.get('a2a', []))} A2A messages",
        "Context graph traversal": next((c["path"] for e in ev if e["tool"] == "graph_traverse" for c in _chunks(e)[:1]), "no traversal this run"),
        "Tool selection & routing": f"{_count(tr, 'route')} routing decisions, {sum(1 for e in tr if e.get('type') == 'route' and e.get('override'))} code overrides",
        "Reflection / self-critique": f"self-critique: {'revised' if (r.get('self_critique') or {}).get('revised') else 'kept'}; reviewer: "
                                      + ", ".join(e["verdict"] for e in tr if e.get("type") == "review"),
        "Confidence scoring": f"confidence {conf.get('score')} -> {conf.get('gate')}",
        "Policy gates": f"status {r.get('status')}; failed {(r.get('receipt') or {}).get('failed_checks')}; approval reasons {(r.get('final') or {}).get('approval_reasons')}",
        "Audit trail design": f"{len(r.get('audit', []))} chained audit entries, chain valid={(r.get('receipt') or {}).get('audit_chain_valid')}",
        "Decision reasoning logs": f"{len(r.get('decisions', []))} decisions: " + ", ".join(sorted({d['kind'] for d in r.get('decisions', [])})),
        "Eval & observability": f"{len(r.get('spans', []))} spans (OTLP-JSON export available)",
        "Cost / token economics": f"${cost.get('cost_usd')} ({cost.get('input_tokens')} in / {cost.get('output_tokens')} out tokens)",
        "Memory & learning loops": f"{len((r.get('learning') or {}).get('hints', []))} learned hint(s) applied; {len((r.get('learning') or {}).get('retrieval_boosts', {}))} boosted excerpts",
        "Self-improving agents": f"planner watch-outs: {len((r.get('plan') or {}).get('watch_outs', []))}",
        "Multi-modal reasoning": "used" if "Lab report (" in ((r.get("inputs") or {}).get("note") or "") else "not used in this run (upload a lab report)",
        "Agentic workflows at scale": "see the Cohort batch tab",
    }
    return m.get(node, "")


def _chunks(e):
    import json
    try:
        return json.loads(e["output"]).get("chunks", [])
    except Exception:
        return []


def coverage_rows(result: dict | None = None) -> list[dict]:
    return [{"group": g, "node": n, "status": st, "where": w, "how": h, "limits": note, "live_evidence": evidence(n, result)}
            for g, n, st, w, h, note in NODES]
