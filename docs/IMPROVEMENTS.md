# NeuroGuard AI - improvement roadmap

**Repository:** https://github.com/sandipdey2/neuroguard-agentic-rag · **Live demo:** https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/

The evaluation numbers below come from `eval/summary.json` (the executed `evaluation.ipynb`, 8 real held-out
profiles + 8 attacks) unless marked otherwise. The list is sorted by priority: patient safety and validity first,
then product quality, then infrastructure. "Effort" is a rough estimate for one engineer.

## Already done in this iteration (low risk, tests re-run)
| Change | Why | Evidence |
|---|---|---|
| **Live progress panel** in the app: orchestrator decisions, MCP tool calls, resource/prompt reads, review verdict and gate status stream into an `st.status` box while the run is in progress (`core/pipeline.start_case`, `describe`, LangGraph `astream`) | A ~1 minute wait with only a spinner looked frozen; users now see the agentic workflow live | `scripts/apptest_session.py` asserts progress lines are rendered for every LLM case (`eval/logs/apptest_py313.log`, `apptest_py314.log`) |
| **Live evidence script** `scripts/live_evidence.py` | Proves the MCP server is a real stdio subprocess (PID, handshake, secret-free env) and that gpt-4o-mini itself returns `tool_calls` | `eval/logs/live_evidence_py313.json`, `live_evidence_py314.json` |
| **Preliminary subgroup audit** `scripts/fairness_check.py` | Needed a first look at sex / age / residence behaviour before any claim of fairness | `eval/fairness_preliminary.json` (findings in P1 below) |
| **Diagram generated from code** | `scripts/draw_architecture.py` now reads tool / resource / prompt / skill names and caps from the server source, contract and skills, and fails if counts drift | `architecture_diagram.png` |

The core evaluation numbers were **not** changed by these items (no model, prompt or routing logic changed).

## Done in v2.3.0 (12 audit gaps closed; live end-to-end runs, 79 pytest tests)
| Change | Evidence |
|---|---|
| top-p in contract, LLM call and receipt; 11 success criteria with a target-vs-observed report | `eval/success_criteria.md` (11/11 met) |
| PubMed E-utilities MCP tool, server-side MCP policy engine, `registry://tools` | `scripts/verify_mcp.py`, `tests/test_server_policy.py` |
| FAISS vector store, Redis cache, PostgreSQL backend switch | `tests/test_rag_vector.py`, `tests/test_cache_db.py` (real PostgreSQL 16 and Redis 6 on the dev box) |
| Durable queue + workers, FastAPI API, HMAC webhook, cron trigger | `tests/test_queue.py`, `tests/test_api.py`, `tests/test_triggers.py`; live API run in `eval/logs/e2e_v23.json` |
| Chat tab + Slack bot with conversation memory; evidence agent as a tool | `tests/test_slack_chat.py`, `tests/test_agent_as_tool.py`; live follow-ups 14.3 s / 29.9 s |
| Docker Compose, Kubernetes manifests, optional vLLM profile | `eval/logs/deploy_validation.log` (kubeconform 18/18, compose config valid); not applied to a live cluster |

Still open from this list: networked A2A (item 10; events now reach the durable queue but agents still run in one process), OIDC auth (12), HIPAA readiness (13), parallel gathering (7). Item 11 is partly done: PostgreSQL is supported, row-level security and WORM audit storage are not. The CI workflow ships as `ci/github-actions-ci.yml` and runs once it is copied to `.github/workflows/`.

## Done in v2.1.0 (agentic-AI knowledge-graph coverage; notebook re-executed)
| Change | Effect (eval/summary.json) |
|---|---|
| Planner -> executor | LLM route choices accepted 0.82 -> 0.98 (this addresses item 6 below); plan followed in 8 of 8 runs |
| Hybrid sparse + dense vectors | paraphrase hit@3 0.875 (sparse) -> 1.0 (hybrid) |
| Graph RAG + agentic re-retrieval | re-retrieval in 25% of runs, final topic coverage 1.0 |
| Writer self-critique | self-revision in 12.5% of runs; reviewer first-draft pass unchanged at 0.875 |
| Confidence gate, decision log, OTLP spans, cost | mean confidence 0.911; USD 0.0053 per run at list price |
| Episodic memory + bounded learning loop, multi-modal intake, batch runner | see docs/KNOWLEDGE_GRAPH_COVERAGE.md (partial items) |

Regression: mean latency rose from 56.3 s to 87.2 s and LLM calls from 17.6 to 20.4, so item 7 (parallel gathering) is now more important.

## P0 - validity and safety
1. **External validation on a licensed clinical cohort** (effort: months, needs data access + IRB/DUA).
   The model is trained on an educational Kaggle dataset without event timing or race/ethnicity (test ROC-AUC 0.84,
   PR-AUC 0.26). Validate against a cohort with incident stroke outcomes (e.g. a credentialed MIMIC-IV / UK Biobank style
   dataset), report calibration-in-the-large and decision curves, and compare with an established score
   (e.g. Framingham stroke risk / pooled cohort equations).
2. **Fairness audit by sex, age and residence** (effort: 1-2 weeks for a proper audit). The preliminary check on the test
   split (`eval/fairness_preliminary.json`) shows the model is mostly an age detector:
   * under 45: 0 of 6 strokes flagged (flag rate 0.0); 65+: 35 of 35 flagged but 98% of all 65+ patients are flagged;
   * within-band ROC-AUC is only 0.66-0.75, versus 0.84 overall;
   * sex and residence gaps are smaller (recall 0.83 female vs 0.76 male; 0.70 rural vs 0.89 urban) but the groups hold
     only 21-29 strokes, so the intervals are wide.
   Next steps: bootstrap CIs per subgroup, age-stratified thresholds or an age-conditional model, and an explicit
   "insufficient evidence for this subgroup" message in the report.
3. **Low precision at the recall-oriented threshold** (precision 0.137, recall 0.80, 253 false positives of 1,022).
   Present the result as "discuss risk factors", never as a screening decision (already the wording); add a second,
   precision-oriented operating point for the clinician and show both in the UI; evaluate decision curves.
4. **Regex guardrails -> classifier + rules** (effort: 1 week). Attack block rate was 1.0 on 8 attacks, but the patterns
   are regular expressions a paraphrased injection could evade. Add a small injection / jailbreak classifier (or the
   provider's moderation endpoint) in front of the rules, keep the BE FAST emergency rule deterministic, and grow the
   attack set to 100+ cases (multilingual, encoded, indirect injection inside the note).
5. **Note classifier** (test F1 0.31, AUC 0.881) is advisory only. Replace TF-IDF+LR with a small clinical
   sentence-embedding model or fine-tuned clinical BERT, or drop it from the UI until it beats a rules baseline.

## P1 - agentic quality and latency
6. **Orchestrator routing accuracy** - in v2.0.1 only 82% of the LLM's own route choices were accepted (8 of 9 overrides
   were premature jumps to the Clinical Writer); with the planner in v2.1.0 it is 98%. Options: constrain the structured-output enum
   to the currently eligible agents (removes invalid choices structurally), add two few-shot examples, or let code
   handle the fixed gather phase and keep the LLM for review/revision decisions. Track "LLM choice accepted" as a KPI.
7. **Latency (v2.2.0 run3: median 84.5 s, one 331 s timeout; 18.1 sequential LLM calls)**:
   * run Intake, Risk Analyst and Evidence Researcher in parallel (they are independent; LangGraph fan-out) - estimated, not yet measured:
     roughly 35-40% less wall-clock time;
   * allow parallel tool calls within a round (already returned by the model; execute with `asyncio.gather`);
   * stream the writer's tokens to the UI (v2.3 added a cache for `search_guidelines` / `semantic_search` results, Redis-backed when `REDIS_URL` is set);
   * keep the MCP subprocess warm across runs instead of one spawn per run (~1.2 s handshake each).
8. **Grounding metric** - word-overlap support is 0.55 while LLM-judged faithfulness is 0.977 (one mis-cited bullet); add an NLI / entailment
   check per evidence bullet and report it alongside the LLM judge (an LLM judging an LLM is weak evidence).
9. **Bigger eval set** - 16 cases are too few for confidence intervals; target 100 real profiles + 100 attacks with
   CI reporting, and run it in CI on every merge (nightly with the real key, budget-capped).

## P2 - product and infrastructure
10. **Networked A2A** - the bus is in-process (`inproc://`). Serve each agent behind an A2A JSON-RPC/HTTP endpoint with
    Agent Cards at `/.well-known/agent.json`, auth between agents and timeouts; the payloads are already A2A-shaped.
11. **Persistent, access-controlled storage** - Streamlit Cloud storage is ephemeral SQLite. Move memory and audit
    logs to Postgres with row-level security, retention policy and append-only audit storage (hash chain + WORM bucket).
12. **Authentication and roles** - add sign-in (OIDC), clinician vs patient roles, and bind the approval decision and
    receipt to an authenticated identity; persist `NG_RECEIPT_KEY` in a secrets manager with rotation.
13. **HIPAA / GDPR readiness** - BAA-covered hosting and LLM provider, no PHI to third parties, DPIA, encryption at rest,
    breach process. Until then the product must stay "demo / research use, no real patient data".
14. **Observability** - OpenTelemetry traces per run (LLM, MCP, hook decisions), token/cost dashboards, alerting on
    gate-refusal rate, content-filter stops and override rate; keep the signed receipt as the audit artefact.
15. **Regulatory path** - if the product ever targets clinical use, map it against FDA CDS criteria and SaMD/IEC 62304
    processes; keep the "not a medical device" scope until then.
