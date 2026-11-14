# Changelog

**Repository:** https://github.com/sandipdey2/neuroguard-agentic-rag · **Live demo:** https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/

## 2.4.0 - 2026-11-14
Simplified for the capstone presentation. No change to the agents, prompts, MCP server, harness contract (still v1.3.0, same hash) or evaluation numbers.
- Removed the chat-platform bots, the mobile-client guide, the self-hosted-model profile and all container and cluster deployment files (`bots/`, `deploy/`, container build/compose files, `Procfile`, `requirements-bots.txt`, `docs/MOBILE_CLIENT.md`, `eval/logs/deploy_validation.log`) with their tests.
- Deployment is now two paths only: Streamlit Community Cloud (live demo) and `uvicorn api.main:app` locally (`docs/DEPLOYMENT.md`).
- Coverage re-mapped honestly: "How to Build an AI Agent" 37 items = 36 full, 1 out of scope; LangGraph + MCP 17 components = 13 full, 4 out of scope. Architecture diagram redrawn without the removed boxes.
- Chat tests moved to `tests/test_chat.py`; config-template checks in `tests/test_config.py`. pytest: 75 tests (73 run and pass; 2 real-server backend tests skip without `NG_TEST_PG_URL` / `NG_TEST_REDIS_URL`).
- API version 2.4.0; product zip `neuroguard-ai-2.4.0.zip`.

## 2.3.0 - 2026-11-14
Closes the 12 gaps of the "How to Build an AI Agent" audit (37/37 items now mapped as full) and maps a LangGraph + MCP reference architecture.
- **top_p** (1.0) in the contract, `get_llm` and the signed receipt. Contract v1.3.0.
- **Success criteria**: 11 targets in `harness_contract.json`; `scripts/success_report.py` writes `eval/success_criteria.json/.md` (11/11 met).
- **PubMed E-utilities** MCP tool `pubmed_search` (keyless, fixed host, 3 req/s, cache, then live, then a snapshot of real responses); literature is context, never a `[G-xx]` citation.
- **Agent as a tool**: `consult_evidence_researcher` (`core/agent_tools.py`) wraps an evidence sub-agent ReAct loop; used by the chat follow-up agent and traced on the A2A bus.
- **Conversation memory** (`core/chat_memory.py`): PHI-redacted, bounded, persisted only with consent; **💬 Chat** tab in the app (`st.chat_input`) and `core/chat.py`.
- **Vector DB**: FAISS `IndexFlatIP` persisted with a corpus fingerprint (`core/vector_store.py`), used by hybrid search.
- **Triggers + durable queue**: FastAPI (`api/main.py`: `/v1/cases`, `/v1/receipts/{id}/verify`, `/v1/chat`, HMAC `/v1/webhooks/case`, `/v1/events/{topic}`, CORS, API key), `scripts/scheduled_batch.py` (cron), `core/job_queue.py` (SQLite WAL / PostgreSQL SKIP LOCKED / Redis) with leases and retries, `core/worker.py`; A2A events are published to the queue.
- **Server-side MCP policy engine** (`core/server_policy.py`): allowlist, clamps, validators, quotas and session pin re-checked inside the server for all 20 tools; tool registry resource `registry://tools`. MCP server: 20 tools, 6 resources + 1 template, 2 prompts.
- **Optional backends**: Redis cache (`REDIS_URL`, `core/cache.py`) and PostgreSQL (`DATABASE_URL`, `core/db.py`) with SQLite / in-process fallbacks.
- Chat-platform bot and deployment packaging (removed again in 2.4.0).
- **LangGraph state machine phases**: planner, then executor (orchestrator + specialists), then validator (safety reviewer + gate) (`NODE_ROLES`, `phase` in `CaseState`).
- **Tests**: 79 pytest tests (`tests/`), CI workflow file `ci/github-actions-ci.yml` (Python 3.13 + 3.14).
- Responsive Streamlit layout; a blocked request no longer stops the remaining tabs from rendering.
- Live e2e (gpt-4o-mini): pipeline finalized in 59.6 s (USD 0.0057); API agentic search finalized in 69.0 s (USD 0.0077); webhook classical finalized in 52.4 s (USD 0.0038).

## 2.2.0 - 2026-11-13
Three selectable RAG modes plus agentic-search context tools; knowledge-graph coverage now 34 nodes (29 full, 5 partial).
- **Classical RAG**: a fixed pipeline calls MCP `semantic_search` once per required topic before the LLM; docs are injected (system + user + docs).
- **Agentic RAG** (default): the Evidence Researcher's context window is assembled from named slots (system prompt, skills YAML front matter, tool list, memory file, history, user message, tool calls, tool results).
- **Agentic search**: 8 new MCP tools - `semantic_search`, `search_files` (read-only `knowledge/` incl. `Plan.md`), `load_skill`, `execute_query` (read-only SQL over the cohort), `web_search` / `web_fetch` (CDC/NIH/WHO allowlist, SSRF guard, offline fallback), `get_memory`, `shell` (grep/wc/head/ls argv sandbox). MCP server: 19 tools.
- **Hooks**: per-run tool set by RAG mode; argument checks for shell, URL and SQL in the pre-tool-use hook and again in the server; skills limited to the agent's catalog; `get_memory` session pinned; web (3) and shell (4) call budgets. One "source check" nudge in agentic-search mode.
- **Context-window inspector**: `context_window` trace events with tiktoken counts per slot; new result tab "Context window"; sidebar RAG-mode selector; `rag_mode` in the signed receipt.
- LLM calls retried up to twice when the gateway answers "timed out".
- Notebook run3: s.6b red team (44/44 shell/SSRF/SQL attacks blocked by hook and server, 5/5 positive controls allowed), s.17 three-mode comparison (hit@3 1.0 / 0.933 / 0.929; latency 71.3 / 113.8 / 89.5 s; tokens 21,272 / 25,023 / 49,867).
- Regressions vs 2.1.0: one eval case (N1) hit the 330 s timeout, so finalized 1.0 -> 0.875; the LLM no longer chose `graph_traverse` (0.625 -> 0.0) and re-retrieval was not triggered (0.25 -> 0.0). Faithfulness improved 0.977 -> 1.0.
- Contract v1.2.0.

## 2.1.0 - 2026-11-13
Every node of the agentic-AI knowledge graph is now implemented and mapped to code (`docs/KNOWLEDGE_GRAPH_COVERAGE.md`,
app tab "Knowledge-graph coverage"): 27 nodes, 23 full, 4 partial.
- **Planner -> executor**: planner node recalls similar episodes (MCP `episode_recall`) and writes a structured plan the orchestrator executes.
- **ReAct traces**: thought / action / observation steps (`react_step`), tool-output truncation and context compaction (`core/context.py`, tiktoken).
- **Embeddings & hybrid retrieval**: 24-d LSA dense vectors (`core/vectors.py`) fused with TF-IDF; `search_guidelines(mode=sparse|dense|hybrid)`.
- **Graph RAG**: networkx knowledge graph (49 nodes, 144 edges, `core/graph_rag.py`), MCP tool `graph_traverse` and resource `graph://knowledge-graph`.
- **Agentic RAG**: code-measured topic coverage; missing topics trigger one reformulated re-retrieval round.
- **Reflection**: writer self-critique (deterministic checks + quote-verified LLM critique) and one self-revision before the Safety Reviewer.
- **Confidence scoring + policy gate** (`core/confidence.py`): refuse < 0.4, clinician review < 0.7.
- **Decision reasoning log**, **OTLP-JSON spans** (`core/observability.py`) and **per-call cost** at list price; all hashed / summarized in the receipt.
- **Episodic memory + bounded learning loop** (`core/episodes.py`, MCP `episode_write`, harness-only): whitelisted hints, retrieval boosts capped at 0.06, reset button; clinician approve/reject is written back.
- **Multi-modal intake (partial)**: lab report image (gpt-4o-mini vision) or text PDF (pypdf) -> validated glucose / BMI / BP (`core/multimodal.py`).
- **Batch at scale (partial)**: `core/batch.py` semaphore + shared token-bucket limiter; app tab "Cohort batch".
- Contract v1.1.0 (budgets 36 LLM / 32 tool calls / 110k tokens / 300 s); MCP server now 11 tools, 5 resources + 1 template, 2 prompts.
- Re-executed `evaluation.ipynb`: route choices accepted 0.82 -> 0.98, paraphrase hit@3 sparse 0.875 vs hybrid 1.0; latency 56.3 -> 87.2 s,
  LLM calls 17.6 -> 20.4 (cost of the planner, re-retrieval and self-critique); word-overlap support 0.612 -> 0.55; LLM-judged faithfulness 1.0 -> 0.977 (one mis-cited bullet of 44).
- New dependencies: networkx 3.7, pypdf 7.1.0.

## 2.0.1 - 2026-11-13
- Live progress panel: orchestrator decisions, MCP tool calls, resource/prompt reads, review and gate stream into the UI
  while a run is in progress (`core/pipeline.start_case` / `describe`, LangGraph `astream`); AppTest asserts it.
- `scripts/live_evidence.py`: MCP handshake, server PID and secret-free environment, LLM-chosen `tool_calls`, and
  orchestrator decisions captured to `eval/logs/live_evidence_py313.json` / `_py314.json`.
- `scripts/fairness_check.py`: preliminary subgroup audit (sex, age band, residence) -> `eval/fairness_preliminary.json`.
- Architecture diagram regenerated from code (names/counts read from the MCP server, contract and skills); fixed
  mismatches (resource count, missing skills/receipt/prompt paths, caps, red-team-review skill, data source, links).
- Repository and live-demo links in every document; `docs/IMPROVEMENTS.md` roadmap; About tab corrected
  ("4 resources + 1 resource template"). Evaluation numbers unchanged.

## 2.0.0 - 2026-11-13
Complete rebuild of the earlier single-file demo.
- Real MCP server (official `mcp` SDK, stdio subprocess): 8 tools, 4 resources + 1 resource template, 2 prompts.
- LangGraph LLM orchestrator with five specialist agents, A2A-style agent cards and task/message/artifact bus.
- Harness: `harness_contract.json` (turns, budgets, permissions, required checks), pre-tool-use hook, invariant
  verifier, LLM red-team pass with quote confirmation, finalize gate, HMAC-SHA256 receipts, hash-chained audit log.
- Skills with progressive disclosure (`skills/*/SKILL.md`).
- Public data only: Stroke Prediction Dataset, MTSamples (CC0), 30 verbatim CDC/NHLBI excerpts.
- Calibrated logistic regression (compared with random forest and gradient boosting), cohort statistics,
  BE FAST red-flag rules and an MTSamples note classifier.
- Consent-based SQLite case memory with a clear button.
- Streamlit UI with trace, A2A, MCP, citations, checks, receipt, approval and PDF/Markdown downloads.
- Executed `evaluation.ipynb`, architecture diagram, `module_summary` paper, buyer docs and `scripts/build_dist.py`.
- Removed: simulated agents, word-overlap "RAG", "MCP-style proxy", hand-weighted score, unpinned requirements.

## 1.0.0
Initial single-file Streamlit demo (superseded).
