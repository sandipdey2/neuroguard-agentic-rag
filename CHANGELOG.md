# Changelog

**Repository:** https://github.com/sandipdey2/neuroguard-agentic-rag · **Live demo:** https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/

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
