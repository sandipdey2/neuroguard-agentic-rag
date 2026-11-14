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
    ("Foundations", "Tool use / function calling", "full", "core/agents.py::react, mcp_server (20 tools)", "bind_tools function calling to real MCP tools over stdio, every call through the pre-tool-use hook", ""),
    ("Foundations", "Embeddings & vectors", "full", "core/vectors.py::DenseEncoder, core/vector_store.py, core/rag.py::search", "24-dim LSA dense vectors stored in a persisted FAISS index + TF-IDF sparse, weighted hybrid fusion; notebook compares OpenAI text-embedding-3-small", "MCP server uses local LSA (it holds no API key by design)"),
    ("Foundations", "Context windows", "full", "core/context.py, core/agents.py::react", "tiktoken o200k_base counts, tool output truncation, observation compaction, budgeted writer evidence", ""),
    ("Retrieval & Memory", "Classical RAG", "full", "core/rag.py::GuidelineIndex.search, core/agents.py::classical_retrieve", "selectable mode 'classical': a fixed pipeline calls MCP semantic_search once per required topic BEFORE the LLM; the docs are injected into the context (system prompt + user message + docs); one LLM call, no tool calls", "no re-retrieval loop in this mode, by design"),
    ("Retrieval & Memory", "Graph RAG", "full", "core/graph_rag.py::traverse, MCP graph_traverse", "networkx graph of factors, conditions and chunks; weighted multi-hop traversal returns paths", ""),
    ("Retrieval & Memory", "Agentic RAG", "full", "core/agents.py::evidence_researcher, agentic_slots", "default mode: the LLM issues search_guidelines / graph_traverse calls itself; context window assembled from slots (system prompt, skills YAML front matter, tool list, memory file, history, user message, tool calls, tool results); missing topics trigger a re-retrieval round", ""),
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
    ("RAG modes & agentic search", "Agentic search: file search", "full", "core/context_tools.py::search_files, MCP search_files", "keyword search over the read-only knowledge/ folder (Plan.md, cohort schema, guideline titles) and skills/*/SKILL.md", ""),
    ("RAG modes & agentic search", "Skill-loading tool", "full", "MCP load_skill, core/permissions.py", "LLM loads a full SKILL.md body on demand; the hook only allows skills in the agent's own catalog", ""),
    ("RAG modes & agentic search", "Database tools", "full", "core/context_tools.py::execute_query, MCP semantic_search", "semantic_search over guideline vectors + execute_query: one read-only SELECT over the cohort (mode=ro, SQLite authorizer, 50 rows, 2 s)", ""),
    ("RAG modes & agentic search", "Web search (allowlisted)", "partial", "core/context_tools.py::web_search, web_fetch", "live https fetch of CDC / NIH / WHO pages only; redirects and DNS answers re-checked (SSRF guard); permission-gated, logged, budgeted; offline -> local excerpt", "not a general search engine: ranks the corpus' known source pages; web text is context, never a citation"),
    ("RAG modes & agentic search", "Memory tool", "full", "MCP get_memory", "this session's de-identified cases + learned hints + episode stats; session_id pinned by the hook", ""),
    ("RAG modes & agentic search", "Shell tool (sandboxed)", "full", "core/context_tools.py::run_shell, MCP shell", "grep / wc / head / ls via argv (no shell interpreter), metacharacters, absolute and parent paths refused, cwd knowledge/, empty env, 3 s timeout, 8 KB cap, 4 calls per run", "allowlist sandbox, not an OS container"),
    ("RAG modes & agentic search", "Context-window inspector", "full", "core/agents.py::context_window_event, app.py 'Context window' tab", "every agent's context window listed slot by slot with tiktoken o200k_base counts; bar chart in the UI, context_window events in the trace", ""),
    ("Frontier", "Memory & learning loops", "partial", "core/episodes.py::learned_hints, retrieval_boosts", "recurring reviewer/clinician feedback becomes whitelisted hints and bounded retrieval boosts", "bounded and reversible by design; no weight updates"),
    ("Frontier", "Self-improving agents", "partial", "core/episodes.py + planner + writer", "agents read their own past failures before acting; effect shown in notebook demo", "prompt-level only; no autonomous code or prompt rewriting"),
    ("Frontier", "Multi-modal reasoning", "partial", "core/multimodal.py", "lab image -> gpt-4o-mini vision structured values; PDF -> pypdf text + regex", "3 fields only, no OCR for scanned PDFs, small rendered test set"),
    ("Frontier", "Agentic workflows at scale", "partial", "core/batch.py, core/job_queue.py, core/worker.py", "cohort batch with semaphore concurrency and shared token-bucket rate limit; v2.3: durable job queue (SQLite / PostgreSQL / Redis) consumed by N workers", "multi-worker path implemented and tested on one machine; not load-tested on a cluster"),
]

# "How to Build an AI Agent" diagram (8 boxes, 37 items) -> implementation. Item labels follow the v2.2 audit of that diagram.
AGENT_BUILD = [
    # box, item, status, where, how, note
    ("1 Purpose", "Problem / goal", "full", "README.md, knowledge/Plan.md, module_summary.md s.1", "stroke-risk decision support for clinicians: cited, checked report from a profile + note", ""),
    ("1 Purpose", "Target users", "full", "README.md, docs/GUMROAD_LISTING.md", "clinicians (report audience 'clinician') and patients ('patient-friendly-summary' skill)", ""),
    ("1 Purpose", "Scope & non-goals", "full", "core/guardrails.py, README disclaimer", "not a medical device: emergency stop, dosing refusal, out-of-scope refusal before any LLM call", ""),
    ("1 Purpose", "Success criteria", "full", "harness_contract.json success_criteria, scripts/success_report.py, eval/success_criteria.md", "11 numeric targets declared before the v2.3 runs; the report compares observed vs target", ""),
    ("2 Prompts", "Role / persona", "full", "core/agents.py::SYSTEM", "one role prompt per agent (planner, orchestrator, 5 specialists, follow-up agent, evidence sub-agent)", ""),
    ("2 Prompts", "Instructions", "full", "core/agents.py::SYSTEM, mcp_server prompts", "step instructions + MCP prompt templates clinical_report / red_team_review", ""),
    ("2 Prompts", "Constraints / guardrails", "full", "prompts + core/harness.py::verify", "no doses, no diagnosis, cite only retrieved ids; re-checked by code", ""),
    ("2 Prompts", "Output format", "full", "core/agents.py pydantic schemas", "with_structured_output (Plan, Route, RiskOut, EvidenceOut, RedTeam, SubOut) + fixed report headings", ""),
    ("2 Prompts", "Examples / skills", "full", "skills/*/SKILL.md, core/skills.py", "progressive-disclosure skills with worked rules; load_skill tool", ""),
    ("3 LLM", "Model choice", "full", "harness_contract.json model, core/config.py::get_llm", "gpt-4o-mini via the OpenAI-compatible Vocareum endpoint (only model used)", ""),
    ("3 LLM", "Context window", "full", "core/context.py, app 'Context window' tab", "tiktoken o200k_base budgets out of 128k; per-slot inspector", ""),
    ("3 LLM", "Cost", "full", "core/harness.py::price, cost_summary", "per-call list-price costing; every receipt carries cost_usd", ""),
    ("3 LLM", "Latency", "full", "core/observability.py, receipts latency_s", "per-call spans (OTLP-JSON), run latency, wall-clock budget 300 s", ""),
    ("3 LLM", "Temperature", "full", "harness_contract.json temperature=0", "deterministic decoding; written into the signed receipt", ""),
    ("3 LLM", "Top-p", "full", "harness_contract.json top_p=1.0, core/config.py::get_llm", "passed to ChatOpenAI and written into every receipt (v2.3)", ""),
    ("4 Tools", "Web & data APIs", "full", "core/pubmed.py (MCP pubmed_search), core/context_tools.py::web_fetch", "NCBI PubMed E-utilities (esearch + esummary, keyless, 3 req/s, cached, real snapshot fallback) + allowlisted CDC/NIH/WHO pages", "PubMed returns metadata only; reports cite only G-xx excerpts"),
    ("4 Tools", "Custom tools", "full", "mcp_server/neuroguard_mcp_server.py", "predict_risk, explain_factors, cohort_stats, screen_note, calculator, graph_traverse, ...", ""),
    ("4 Tools", "MCP", "full", "official mcp SDK (FastMCP) over stdio", "20 tools, 6 resources + 1 template, 2 prompts; server-side policy engine", ""),
    ("4 Tools", "AI agent as a tool", "full", "core/agent_tools.py::make_evidence_agent_tool", "Evidence Researcher sub-agent wrapped as LangChain StructuredTool consult_evidence_researcher; bound by the follow-up agent; A2A-traced", ""),
    ("5 Memory", "Working memory", "full", "core/agents.py::CaseState", "LangGraph state across nodes in one run", ""),
    ("5 Memory", "Episodic memory", "full", "core/episodes.py + core/chat_memory.py", "run episodes for the learning loop + de-identified conversation turns recalled into follow-ups (consent-gated, bounded)", ""),
    ("5 Memory", "Vector database", "full", "core/vector_store.py (FAISS IndexFlatIP, persisted)", "guideline embeddings in a persisted FAISS index; semantic_search / hybrid search read it; numpy fallback", ""),
    ("5 Memory", "SQL database", "full", "core/db.py, core/memory.py", "SQLite by default, PostgreSQL via DATABASE_URL (tested on a real PostgreSQL 16 server)", ""),
    ("5 Memory", "File storage", "full", "knowledge/, data/runtime/, skills/", "read-only knowledge files (search_files / shell), model cache, FAISS index files", ""),
    ("6 Orchestration", "Framework (LangGraph)", "full", "core/agents.py::build_graph", "StateGraph planner -> orchestrator <-> specialists -> gate", ""),
    ("6 Orchestration", "Triggers", "full", "api/main.py (POST /v1/cases, /v1/webhooks/case), scripts/scheduled_batch.py", "UI, HTTP API, HMAC-signed webhook and a cron-ready scheduled batch CLI", ""),
    ("6 Orchestration", "Message queue", "full", "core/job_queue.py, core/worker.py", "durable SQLite (WAL) job queue with leases + retries; PostgreSQL (SKIP LOCKED) or Redis backends; A2A events published to it", ""),
    ("6 Orchestration", "Agent2Agent", "full", "core/a2a.py", "agent cards + task/message/artifact on an in-process bus; events mirrored to the queue", "in-process, not a networked A2A server"),
    ("6 Orchestration", "Error handling", "full", "core/agents.py::call_llm, core/worker.py, core/cache.py", "gateway retries, tool errors as data, timeouts, job retry + lease recovery, cache / DB fallbacks", ""),
    ("7 Interface", "Chat interface", "full", "app.py 'Chat' tab, core/chat.py", "st.chat_input / st.chat_message: run cases and ask follow-ups with conversation memory", ""),
    ("7 Interface", "Web app", "full", "app.py (Streamlit Community Cloud)", "live demo with 7 tabs", ""),
    ("7 Interface", "API endpoint", "full", "api/main.py (FastAPI)", "POST /v1/cases, GET /v1/cases/{id}, GET /v1/receipts/{id}/verify, webhook, chat, health; API-key auth, CORS, OpenAPI /docs", "Streamlit Cloud cannot host it: run it locally with uvicorn (docs/DEPLOYMENT.md)"),
    ("7 Interface", "Slack / Discord bot", "out of scope", "-", "removed in v2.4 for capstone simplicity; the Streamlit Chat tab and POST /v1/chat cover chat access", "not shipped"),
    ("8 Testing", "Unit tests", "full", "tests/ (pytest), ci/github-actions-ci.yml", "harness, guardrails, permissions, sandbox, SQL authorizer, RAG/vector store, queue, cache, DB, API, chat memory, agent-as-tool, config templates; CI workflow for 3.13 + 3.14", "the CI workflow ships as ci/github-actions-ci.yml; it runs once copied to .github/workflows/ (the push token had no workflow scope)"),
    ("8 Testing", "Latency tests", "full", "evaluation.ipynb, eval/logs/e2e_v23.json", "per-run latency in every eval row and receipt; median vs target in success criteria", ""),
    ("8 Testing", "Quality metrics", "full", "evaluation.ipynb, eval/summary.json", "attack block rate, finalized rate, citation coverage, faithfulness, hit@3, AUC, Brier", ""),
    ("8 Testing", "Improvement loop", "full", "core/episodes.py, docs/IMPROVEMENTS.md", "episodes -> whitelisted hints + bounded retrieval boosts; roadmap from failures", ""),
]

# "LangGraph + MCP Architecture" reference diagram -> implementation.
LANGGRAPH_MCP = [
    ("Client", "Web UI", "full", "app.py", "Streamlit app (assess form, chat tab, traces); responsive CSS for narrow screens", ""),
    ("Client", "Mobile App", "out of scope", "-", "removed in v2.4 for capstone simplicity; clients use the Streamlit web UI or the REST API", "no mobile app or mobile client docs are shipped"),
    ("Client", "API Request", "full", "api/main.py", "JSON over HTTPS with X-API-Key; webhook with HMAC signature", ""),
    ("API Gateway", "FastAPI", "full", "api/main.py", "FastAPI app: auth, validation (pydantic), CORS, queue hand-off, receipt verification", ""),
    ("LangGraph Agent System", "Planner Node", "full", "core/agents.py::planner", "LLM plan from episodic memory (NODE_ROLES planner)", ""),
    ("LangGraph Agent System", "Executor Node", "full", "core/agents.py::orchestrator + intake / risk_analyst / evidence_researcher / clinical_writer", "orchestrator executes the plan by delegating to specialists (NODE_ROLES executor)", ""),
    ("LangGraph Agent System", "Validator Node", "full", "core/agents.py::safety_reviewer + gate", "12-check verifier + red team + finalize/confidence gate (NODE_ROLES validator)", ""),
    ("LangGraph Agent System", "Agent State Machine", "full", "core/agents.py::CaseState, NODE_ROLES; result['state_machine']", "typed LangGraph state with phase = planner | executor | validator; executed phase sequence returned per run", ""),
    ("MCP Server", "Tool Registry", "full", "mcp_server: tools/list + resource registry://tools", "every tool with granted agents, clamps, quotas and high-risk controls", ""),
    ("MCP Server", "Memory Access", "full", "MCP memory_read / memory_write / get_memory / episode_recall / episode_write", "session-pinned, consent-gated memory tools over SQLite or PostgreSQL", ""),
    ("MCP Server", "Policy Engine", "full", "core/server_policy.py::ServerPolicy.guard (server) + core/permissions.py (client hook)", "server re-checks allowlist, clamps, validators and per-run quotas from the contract on every tools/call", ""),
    ("Data & Memory Layer", "Redis (cache)", "full", "core/cache.py", "REDIS_URL -> Redis cache for retrieval, PubMed and opt-in LLM responses; in-process fallback (tested on real Redis 6.2 + fakeredis)", "optional; Streamlit Cloud uses the in-process cache"),
    ("Data & Memory Layer", "PostgreSQL (database)", "full", "core/db.py", "DATABASE_URL -> memory, episodes, chat turns and job queue in PostgreSQL (tested on a real PostgreSQL 16.2 server); SQLite default", "optional; Streamlit Cloud uses SQLite"),
    ("Data & Memory Layer", "Vector Store (embeddings)", "full", "core/vector_store.py", "persisted FAISS index of the guideline embeddings", ""),
    ("Deployment & LLM Infrastructure", "Docker / Compose", "out of scope", "-", "removed in v2.4 for capstone simplicity; the app runs on Streamlit Community Cloud and the API locally with uvicorn", "no container files are shipped"),
    ("Deployment & LLM Infrastructure", "Kubernetes", "out of scope", "-", "removed in v2.4 for capstone simplicity", "no cluster manifests are shipped"),
    ("Deployment & LLM Infrastructure", "vLLM", "out of scope", "-", "removed in v2.4 for capstone simplicity; the only LLM is gpt-4o-mini via the Vocareum OpenAI-compatible endpoint", "no self-hosted model profile is shipped"),
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
        "Classical RAG": f"mode {r.get('rag_mode')}: " + (f"{tools.count('semantic_search')} fixed-pipeline semantic_search calls before the LLM" if r.get("rag_mode") == "classical" else "run with RAG mode = classical to see the fixed pipeline") + f"; {len(r.get('retrieved') or {})} excerpts read",
        "Agentic search: file search": _tool_ev(ev, ("search_files",)),
        "Skill-loading tool": _tool_ev(ev, ("load_skill",)),
        "Database tools": _tool_ev(ev, ("semantic_search", "execute_query")),
        "Web search (allowlisted)": _tool_ev(ev, ("web_search", "web_fetch")),
        "Memory tool": _tool_ev(ev, ("get_memory",)),
        "Shell tool (sandboxed)": _tool_ev(ev, ("shell",)),
        "Context-window inspector": "; ".join(f"{e['agent']} {e['total_tokens']}" for e in tr if e.get("type") == "context_window"
                                             and e["agent"] == "evidence_researcher") + " tokens (Evidence Researcher)",
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
        "Agentic workflows at scale": "see the Cohort batch tab; API jobs go through the durable queue",
    }
    return m.get(node, "")


def _tool_ev(ev, names):
    hits = [e for e in ev if e["tool"] in names]
    if not hits:
        return "not used in this run (select RAG mode = agentic search)"
    return f"{len(hits)} call(s): " + ", ".join(f"{e['tool']}({'allow' if e['allowed'] else 'deny'})" for e in hits[:6])


def _chunks(e):
    import json
    try:
        return json.loads(e["output"]).get("chunks", [])
    except Exception:
        return []


def agent_build_rows() -> list[dict]:
    return [{"box": b, "item": i, "status": st, "where": w, "how": h, "limits": n} for b, i, st, w, h, n in AGENT_BUILD]


def langgraph_mcp_rows() -> list[dict]:
    return [{"layer": l_, "component": c, "status": st, "where": w, "how": h, "limits": n} for l_, c, st, w, h, n in LANGGRAPH_MCP]


def coverage_rows(result: dict | None = None) -> list[dict]:
    return [{"group": g, "node": n, "status": st, "where": w, "how": h, "limits": note, "live_evidence": evidence(n, result)}
            for g, n, st, w, h, note in NODES]
