# Changelog

**Repository:** https://github.com/sandipdey2/neuroguard-agentic-rag · **Live demo:** https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/

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
