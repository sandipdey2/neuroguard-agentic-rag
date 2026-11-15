# Changelog

**Repository:** https://github.com/sandipdey2/neuroguard-agentic-rag · **Live demo:** https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/

## 2.6.1 - 2026-11-15
ruff and mypy now run in CI. No change to agents, MCP, the harness contract or evaluation numbers.
- The lint and type steps from `docs/CI_LINT_STEP.md` were added to `.github/workflows/ci.yml` (commit b8fb7f7). Run https://github.com/sandipdey2/neuroguard-agentic-rag/actions/runs/43707597000 passed on Python 3.13 and 3.14: pytest, MCP server over stdio, Lint (ruff) and Types (mypy) all succeeded.
- Docs updated so C03 Quality on the Lauren Tan sheet is fully covered. `docs/CI_LINT_STEP.md` is kept as a reference noting the steps are applied.
- API version 2.6.1; product zip neuroguard-ai-2.6.1.zip.

## 2.6.0 - 2026-11-15
Closes the gaps found when the project was checked against Lauren Tan's "Agent Operating System" sheet. The safety gate is not loosened: all 12 required checks, budgets, thresholds and the red team are unchanged. Contract v1.4.0.
- **Gardener (was missing):** MCP prompt `garden_review` plus `scripts/garden.py`. The script reads only de-identified episodes (episodic-memory table, `eval/raw_results.json`, `eval/logs/*.json`; no notes, questions or report text). It clusters repeated failed checks, red-team rules, reviewer revisions, writer self-revisions, re-retrievals, orchestrator overrides, clinician rejections and non-finalized outcomes (support >= 2). For each pattern it writes the recurring pattern, support count, episode ids, a proposed constraint (check / tool / hint) and a validation plan to `eval/gardening_queue.md` and `eval/logs/garden_report.json`. `--llm` asks gpt-4o-mini through `garden_review`, and code drops proposals with unknown counts or loosening wording. `--trial [--live]` tries ONE existing whitelisted hint (`NG_TRIAL_HINT`, HINT_LIBRARY keys only) on ONE `eval/cases.json` profile. It only proposes: it never edits code, prompts, the contract or checks.
- **Git commit on the receipt (E01):** `git_sha` (`git rev-parse HEAD`) and `git_dirty` (`git status --porcelain`, tracked files) are inside the HMAC-signed payload. When git is unavailable they fall back to `GIT_SHA` / `NG_GIT_SHA`, then `unknown` (`git_dirty` null).
- **PASS / FAIL / NOT_RUN (BLOCKED, E05):** `verify()` returns `check_status`, `unrun_checks` and `required` alongside the backward-compatible `checks`, `failed` and `passed`. A check is NOT_RUN when its input never existed: no `predict_risk` result, no Risk Analyst summary, an empty draft (so `no_dosing_advice` cannot silently pass), red team not run, or a contract check the verifier does not compute. `release_decision` blocks any required NOT_RUN as `blocked_unrun` with a reason naming the check. A `passed=True` flag cannot override it. The writer pre-check (`require_red_team=False`) records `red_team_pass` as NOT_RUN and not required. The receipt carries `check_status`, `unrun_checks` and `block_reason`. The app's Checks & receipt tab, the gate error message, the Markdown export, the verify span (`checks.not_run`) and the Monitoring failed-check chart (`NOT_RUN:<check>`) show it.
- **Lint + types:** `ruff==0.17.5` and `mypy==2.4.1` in `requirements-dev.txt`, with `ruff.toml` and `mypy.ini` (pragmatic scope: harness, pipeline, episodes, config, permissions, confidence, monitoring, garden). Fixed the real findings: 4 unused imports, an f-string without placeholders, an unused variable, trailing whitespace, a possible `None` base URL in `get_credentials`, and Optional handling in `verify()` and `_run_graph`. The CI step was documented in `docs/CI_LINT_STEP.md` (the automation token cannot edit workflows) and was pasted in 2.6.1.
- Architecture diagram regenerated (title v2.6.0, 3 prompts, NOT_RUN gate, git commit on the receipt).
- Tests: 113 pytest tests (111 run and pass; 2 real-server backend tests skip). New tests cover NOT_RUN statuses, the blocked gate, git fields in the signed payload (tamper detection), Gardener clustering and proposals, the propose-only guarantee, whitelisted trials and `garden_review` over real MCP stdio. `scripts/verify_mcp.py` asserts the 3 prompts.
- Live: public case N2 with gpt-4o-mini via Vocareum finalized in 56.3 s, with 19 LLM calls, 11 tool calls, 27,970 tokens and \$0.005189. All 12 checks were PASS and `unrun_checks` was empty. The receipt `git_sha` was the v2.6.0 code commit 0fb8c59 (`git_dirty` false) and the signature was valid (`eval/logs/v260_live.json`). The Gardener on 37 real episodes found 2 recurring patterns (`override:clinical_writer` 4, `retrieval:needed` 4). A one-case live trial of the whitelisted override hint on N1 finalized with the hint injected and no override (\$0.005543).
- API version 2.6.0; product zip neuroguard-ai-2.6.0.zip.

## 2.5.3 - 2026-11-14
One solid attempt to improve Arize AX delivery (gRPC + smaller batches). Phoenix remains the complete trace record. No change to agents, MCP, the harness contract or evaluation numbers.
- **Phoenix is the complete record:** a local Phoenix collector is the supported, complete OpenTelemetry record for a run (all spans, export failures = 0, no unknown OpenInference kinds). Arize AX stays an optional, best-effort cloud view of the same redacted spans. Arize's collector may drop spans even when the exporter reports SUCCESS, and OTLP HTTP 5xx is not retried by the upstream client (we retry with backoff). When the two UIs disagree, trust Phoenix.
- **OTLP gRPC to Arize (preferred when installed):** if `opentelemetry-exporter-otlp-proto-grpc` is available, Arize export uses Arize's recommended gRPC transport (`https://otlp.arize.com/v1` → `otlp.arize.com:443`), gzip, and `space_id` / `api_key` / `arize-space-id` / `authorization` metadata, still behind the redacting exporter. Resource attributes stay `openinference.project.name`, `model_id`, `arize.project.name`, `arize.project.type=application`. The gRPC extra lives in `requirements-phoenix.txt` (grpcio is heavy for Streamlit Cloud); without it, export falls back to OTLP HTTP. `ARIZE_OTLP_TRANSPORT=http` forces the fallback.
- **Smaller batches and retries:** BatchSpanProcessor uses max_export_batch_size=64, schedule_delay 1s, max_queue_size 4096. Failed exports retry the same batch with backoff on HTTP 5xx/429 and gRPC UNAVAILABLE.
- **Attribute caps:** `input.value` / `output.value` are truncated to 2 KB; other string attributes to 1 KB; span events are capped at 20, all still redacted.
- Tests: 93 pytest tests (91 run and pass; 2 real-server backend tests skip). New tests cover gRPC vs HTTP fallback, batch settings, 500 retries, and attribute caps.
- **Phoenix e2e (headline evidence):** two real gpt-4o-mini public-data cases (demo67, N2) produced 220 and 216 Phoenix spans, 0 Phoenix export failures, 0 unknown OpenInference kinds, official first-pass `red_team_pass` 2/2, 0 revisions, \USD 0.010849. Arize gRPC also reported 0 export failures (SUCCESS) but GraphQL stabilized at 48 and 33 spans for those traces, so counts still do not match. That is why Arize remains best-effort.
- API version 2.5.3; product zip neuroguard-ai-2.5.3.zip.

## 2.5.2 - 2026-11-14
Fix for hidden OTLP export failures and a tighter TracerProvider setup. No change to agents, MCP, the harness contract or evaluation numbers.
- **Honest export accounting:** a later SUCCESS from the same exporter no longer hides earlier FAILURE results (the v2.5.1 e2e snapshot could report 220 SUCCESS / 0 failures even when the SDK had logged HTTP 500s). Failures are counted for the whole run; ok is true only if every export succeeded. HTTP status counts are recorded (no bodies or headers).
- **Only redacting-wrapped exporters:** init_tracing now builds a standard OpenTelemetry TracerProvider itself and attaches at most two BatchSpanProcessors (Phoenix and/or Arize), each behind the redacting exporter. It no longer calls phoenix.otel.register or arize.otel.register, which attach a default processor that we could miss wrapping. Each destination is a BatchSpanProcessor (not SimpleSpanProcessor): a 220-span run is a handful of HTTP POSTs instead of one request per span. The v2.5.1 e2e lost ~20 of 220 Arize spans to HTTP 500s on that request burst; the OTLP HTTP client does not retry 500, so the wrapper retries a failed 500 once after a short delay. Resource attributes include openinference.project.name, model_id, arize.project.name and arize.project.type=application. Short scripts still call force_flush() so the last batch leaves before exit.
- **OpenInference span kinds:** every manual span sets `openinference.span.kind` so Arize AX does not show `kind: unknown`. `neuroguard.run` is AGENT; planner/orchestrator are CHAIN; specialist nodes are AGENT; MCP tools are TOOL (with `tool.name` and redacted, truncated `input.value`/`output.value`); `verify` is EVALUATOR; `node.gate` is GUARDRAIL. Auto-instrumented spans with a missing or UNKNOWN kind are filled in at export time (LLM stays LLM). The redacting exporter keeps `openinference.span.kind`, `tool.name`, `llm.model_name` and `llm.token_count.*`; it redacts but does not drop `input.value`/`output.value`.
- **Writer pre-check vs official verify:** the first `verify` span in Arize (11/12, only `red_team_pass` failed) was the writer's deterministic self-critique calling `verify(..., red_team=None)` before `red_team_review` ran. There was no confirmed red-team rule or quote; episode logs show `review_verdicts=["approve"]`, `self_critique=clean`, and no reviewer revision. The pre-check now uses `require_red_team=False` and emits `verify.precheck`. The Safety Reviewer and finalize gate still run the full 12-check path, including `red_team_pass`; `confirm_red_team` and the gate thresholds are unchanged. The clinical_writer system prompt restates no medicines/doses, no stroke-diagnosis wording, and no certainty language.
- Tests: 88 pytest tests (86 run and pass; 2 real-server backend tests skip). New tests cover wrapped processors, failure accounting, span kinds, and the writer pre-check / official `red_team_pass` split.
- API version 2.5.2; product zip neuroguard-ai-2.5.2.zip.

## 2.5.1 - 2026-11-14
Optional Arize AX (cloud) export alongside Phoenix. No change to the agents' prompts, the MCP tool set, the harness contract (still v1.3.0, same hash) or the evaluation numbers.
- **Arize AX (optional):** when `ARIZE_SPACE_ID` and `ARIZE_API_KEY` are set (environment or Streamlit secrets), `core/observability.py` also exports the same redacted OpenTelemetry spans to Arize AX via the official `arize-otel` package (`arize.otel.register(..., project_name='neuroguard')` or the OTLP HTTP endpoint `https://otlp.arize.com/v1`). Off when the vars are absent. Phoenix and Arize can be on at the same time (two span processors on one TracerProvider, both behind the redacting exporter so PHI and keys never leave).
- **📈 Monitoring tab** now shows a Phoenix on/off line and an Arize on/off line, with a link to https://app.arize.com/ when Arize is on.
- `arize-otel` is in `requirements.txt` (lightweight, Streamlit-Cloud-safe) so the live app can export once the secrets are added. Phoenix UI/client packages stay in `requirements-phoenix.txt`.
- Tests: 84 pytest tests (82 run and pass; 2 real-server backend tests skip without `NG_TEST_PG_URL` / `NG_TEST_REDIS_URL`), including Arize config detection and a dual-exporter redaction test (mocked, no network). CI does not require a Phoenix or Arize collector.
- API version 2.5.1; product zip `neuroguard-ai-2.5.1.zip`. Architecture diagram optional box labelled "Phoenix / Arize AX".

## 2.5.0 - 2026-11-14
Optional Arize Phoenix tracing and a Streamlit Monitoring dashboard. No change to the agents' prompts, the MCP tool set, the harness contract (still v1.3.0, same hash) or the evaluation numbers.
- **Arize Phoenix (optional):** `core/observability.py` registers `phoenix.otel` (OpenInference LangChain / OpenAI instrumentation when those packages are installed) if `PHOENIX_COLLECTOR_ENDPOINT` or `NG_PHOENIX=1` is set; otherwise tracing is a no-op so Streamlit Cloud and CI work without Phoenix. Project name `neuroguard`. Manual spans around each LangGraph node (planner, specialists, finalize gate), each MCP tool call (tool name, redacted args, latency, ok/error) and the 12-check `verify()`. Attributes: case id, contract SHA, run cost, tokens, LLM/tool call counts, checks passed/failed, decision, confidence. PHI, API keys and raw notes are redacted or dropped.
- **📈 Monitoring tab:** `core/monitoring.py` stores de-identified run metrics in SQLite and charts latency, cost, tokens, LLM/tool calls, check pass rate, decision mix, confidence and the failed-check breakdown. A status line reports whether Phoenix tracing is on and links to http://localhost:6006. Works on Streamlit Cloud without Phoenix.
- Demo: `requirements-phoenix.txt` (not in the Cloud install) and `scripts/run_phoenix_demo.py`.
- Tests: 80 pytest tests (78 run and pass; 2 real-server backend tests skip without `NG_TEST_PG_URL` / `NG_TEST_REDIS_URL`), including in-memory OpenTelemetry span tests and monitoring aggregation. CI installs the OpenTelemetry SDK only, not Phoenix.
- API version 2.5.0; product zip `neuroguard-ai-2.5.0.zip`. Architecture diagram retitled v2.5 with an optional Phoenix box.

## 2.4.1 - 2026-11-14
CI is now active on GitHub Actions.
- Workflow at `.github/workflows/ci.yml` (Python 3.13 and 3.14): pytest and the MCP server check over stdio. First run was green on 2026-11-14 (https://github.com/sandipdey2/neuroguard-agentic-rag/actions/runs/43604794444).
- Docs, coverage table, README CI badge and product zip updated. The earlier copy-this-file draft at `ci/github-actions-ci.yml` was removed as a duplicate.
- API version 2.4.1; product zip `neuroguard-ai-2.4.1.zip`.

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
