# 🧠 NeuroGuard AI — governed multi-agent stroke-risk decision support

**Repository:** [https://github.com/sandipdey2/neuroguard-agentic-rag](https://github.com/sandipdey2/neuroguard-agentic-rag) · **Live demo:** [https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/](https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/) · **Paper:** [module_summary.pdf](module_summary.pdf) · **Evaluation:** [evaluation.ipynb](evaluation.ipynb) · **License:** MIT (code)

> ⚠️ **Decision support only — not a medical device.** No regulatory clearance. Not for diagnosis, treatment or emergency triage.
> If someone shows sudden face drooping, arm weakness or speech difficulty, call 9-1-1.

NeuroGuard AI turns a patient profile (plus an optional free-text note or lab report) into a **cited, verified stroke-risk report**.
A planner and an LLM orchestrator (gpt-4o-mini via the OpenAI-compatible Vocareum endpoint, temperature 0) delegate to five
specialist agents that act only through a **real MCP server** (official `mcp` SDK, stdio subprocess). Retrieval is hybrid
(sparse + dense vectors), graph-based and agentic; a deterministic verifier, an LLM red-team pass and the writer's own
self-critique check every draft; a finalize gate and a confidence gate refuse anything that fails; and every run ends
with an **HMAC-SHA256 signed receipt**, a decision reasoning log, OTLP-JSON spans and a cost estimate.
Every node of the agentic-AI knowledge graph is mapped to code in [docs/KNOWLEDGE_GRAPH_COVERAGE.md](docs/KNOWLEDGE_GRAPH_COVERAGE.md). Udacity AI Mastery Capstone — Project 7 (Integrative Industry Synthesis), healthcare.

![architecture](architecture_diagram.png)

## How a request flows (as implemented)
1. **Input guardrails** (`core/guardrails.py`, no LLM): PHI redaction → prompt-injection/tool-misuse block → CDC BE FAST emergency stop ("call 9-1-1") → medication-dosing refusal → out-of-scope refusal.
2. **Run contract** (`harness_contract.json` v1.2.0) is loaded and SHA-256 hashed before the first token: 9 turns, 36 LLM calls, 32 tool calls, 110k tokens, 300 s, per-agent permissions, 12 required checks, context budgets, confidence policy, list prices, learning and batch limits.
3. **Planner** (LLM) recalls similar de-identified past episodes via MCP `episode_recall` and writes a structured plan; the **orchestrator** (LangGraph) executes it — the LLM picks next agent + skill + instruction; code enforces eligibility, FINISH-only-after-review, one revision, visit/turn caps. Every decision goes to the decision log.
4. **Specialists** — Intake, Risk Analyst, Evidence Researcher, Clinical Writer, Safety Reviewer — exchange **A2A-style** tasks/messages/artifacts (`core/a2a.py`). Tool users run a **ReAct** loop: thought → tool call → **pre-tool-use hook** → MCP observation (PHI-redacted, token-truncated) → back to context. The Evidence Researcher runs in one of three **RAG modes** chosen in the sidebar: *classical* (a fixed pipeline calls `semantic_search` before the LLM and injects the docs), *agentic* (default: the LLM calls hybrid search and **graph RAG** itself and re-retrieves when code finds a required topic uncovered) or *agentic search* (MCP context-retrieval tools over local files incl. `knowledge/Plan.md`, skills, the cohort database, allowlisted CDC/NIH/WHO pages, memory and a sandboxed shell). The **Context window** tab lists every slot with its token count. The writer **self-critiques** and revises once before review.
5. **Finalize gate** re-runs the verifier, then a **confidence score** (calibration, cohort size, coverage, verifier, red team, grounding) refuses below 0.4 and asks for review below 0.7; high-risk or red-flag cases need **clinician sign-off**; one de-identified **episode** is stored for the bounded learning loop; a signed receipt is produced.

## Coverage of the nine required concepts
| # | Concept | Where / how (real code) |
|---|---|---|
| 1 | Harness: agent loop, fixed contract, gate | `react()` loop in `core/agents.py`; `harness_contract.json` hashed in `core/harness.py`; `gate` node refuses to finalize unless all 12 checks pass |
| 2 | Memory & state | Short-term: LangGraph `CaseState`. Long-term: SQLite (`core/memory.py`) via MCP `memory_read`/`memory_write`, consent checkbox + clear button |
| 3 | RAG | three selectable modes (classical / agentic / agentic search, see below); `core/rag.py` hybrid TF-IDF + 24-d LSA vectors over 30 verbatim CDC/NHLBI excerpts, `core/graph_rag.py` graph traversal, agentic re-retrieval → `guideline://{id}` resources → writer cites `[G-xx]`; verifier checks citations were retrieved |
| 4 | MCP | `mcp_server/neuroguard_mcp_server.py`: 19 tools, 5 resources + 1 resource template, 2 prompts; `scripts/verify_mcp.py` |
| 5 | Skills | `skills/*/SKILL.md`; `core/skills.py` exposes name+description only; body injected when selected (trace: `skill_loaded`) |
| 6 | Guardrails & permissions | Input guardrails; permission engine + pre-tool-use hook (`core/permissions.py`); AST-sandboxed calculator; consent for writes; human approval; hash-chained audit log |
| 7 | Evals | Executed `evaluation.ipynb`, `eval/cases.json`, `eval/summary.json`, `eval/raw_results.json` (traces) |
| 8 | A2A | Agent Cards + Task/Message/Artifact on an **in-process** bus modelled on the A2A spec (not a networked A2A server) |
| 9 | Multi-agent system | LLM orchestrator + 5 specialists with a genuine graph cycle and caps |
| + | Verifier, red team, receipt | Deterministic invariants + quote-confirmed LLM red team; HMAC-SHA256 `receipt.json` (download in UI) |

## Agentic-AI knowledge-graph coverage (34 nodes: 29 full, 5 partial)
Full table with file/function, mechanism and evidence: [docs/KNOWLEDGE_GRAPH_COVERAGE.md](docs/KNOWLEDGE_GRAPH_COVERAGE.md); live evidence per node for your own run: app tab **🧩 Knowledge-graph coverage**.

| Group | Node → implementation (status) |
|---|---|
| Foundations | LLM reasoning → 7 gpt-4o-mini roles (full) · Prompt patterns → role prompts, ReAct format, MCP prompts, skills (full) · Tool use → `bind_tools` + MCP (full) · Embeddings & vectors → `core/vectors.py` LSA + hybrid; OpenAI embeddings compared in the notebook (full) · Context windows → `core/context.py` tiktoken budgets (full) |
| Retrieval & Memory | Classical RAG → fixed `semantic_search` before the LLM, docs injected (full) · Graph RAG → `core/graph_rag.py`, MCP `graph_traverse` (full) · Agentic RAG → LLM tool calls, slot-assembled context window, coverage check + re-retrieval (full) · Short-term → `CaseState` (full) · Long-term → `core/memory.py` (full) · Episodic → `core/episodes.py` (full) |
| Agent patterns | ReAct → `react()` thought/action/observation (full) · Planner → executor → `planner` node (full) · Multi-agent orchestration → LangGraph + A2A bus (full) · Context graph traversal → multi-hop paths (full) · Tool selection & routing → orchestrator + permission engine (full) · Reflection → writer self-critique + reviewer (full) |
| Production & governance | Confidence scoring → `core/confidence.py` (full) · Policy gates → guardrails, hook, finalize + confidence gate (full) · Audit trail → hash chain + signed receipt (full) · Decision reasoning logs → `RunContext.decide` (full) · Eval & observability → notebook + OTLP-JSON spans (full) · Cost / token economics → per-call list-price costing (full) |
| RAG modes & agentic search | File search → `search_files` over `knowledge/` (Plan.md) + skills (full) · Skill loading → `load_skill`, own catalog only (full) · Database tools → `semantic_search` + read-only `execute_query` (full) · Web search → CDC/NIH/WHO allowlist, SSRF guard, offline fallback (**partial**: not a general search engine) · Memory tool → `get_memory` (full) · Shell → argv allowlist sandbox (full) · Context-window inspector → app tab + `context_window` trace (full) |
| Frontier | Memory & learning loops → whitelisted hints + bounded boosts (**partial**) · Self-improving agents → prompt-level only (**partial**) · Multi-modal reasoning → `core/multimodal.py` lab image (vision) / PDF (**partial**) · Agentic workflows at scale → `core/batch.py` semaphore + token bucket (**partial**) |

## Integration of prior capstones
| Project | Reused concept | In NeuroGuard |
|---|---|---|
| P1 [ai-programming-foundations](https://github.com/sandipdey2/ai-programming-foundations-project) | logged cleaning workflow | `core/data.py` pipeline log, train-only imputation |
| P2 [statistical-analysis](https://github.com/sandipdey2/statistical-analysis-project) | hypothesis tests | chi-square, Welch t, odds ratios, Wilson CIs → `cohort_stats` tool |
| P3 [breast-cancer-classification](https://github.com/sandipdey2/breast-cancer-classification) | logistic regression | LR vs RF vs GBM, class weights, calibration, exact explanations |
| P4 [imdb-sentiment-rnn](https://github.com/sandipdey2/imdb-sentiment-rnn) | text/sequence modelling | TF-IDF+LR note classifier (MTSamples) + BE FAST rules; lightweight instead of LSTM for Cloud CPU |
| P5 [generative-mnist](https://github.com/sandipdey2/generative-mnist) | generative modelling | Clinical Writer generates the report from an MCP prompt |
| P6 [agentic-rag-system](https://github.com/sandipdey2/agentic-rag-system) | agentic RAG | multi-agent + **real** MCP + verifier; fixes the earlier MCP/loop overclaiming |

## Evaluation results (from the executed notebook, `eval/summary.json`)
| Metric | Result |
|---|---|
| Risk model held-out ROC-AUC / PR-AUC / Brier (1,022 patients, 50 strokes) | 0.84 / 0.26 / 0.041 |
| Recall / precision at recall-oriented threshold | 0.8 / 0.137 |
| CV ROC-AUC: logistic / random forest / gradient boosting | 0.838 / 0.835 / 0.822 |
| Note classifier test F1 (advisory) | 0.31 |
| Retrieval hit@3 / MRR (12 labelled queries), hybrid | 1.0 / 1.0 |
| Paraphrase hit@3 (8 queries): sparse / dense LSA / hybrid / OpenAI embeddings | 0.875 / 1.0 / 1.0 / 1.0 |
| Graph RAG: 49 nodes, 144 edges; factor-query hit@3 / precision@3 | 1.0 / 0.81 |
| Attacks blocked (8) / normal requests wrongly blocked (8) | 1.0 / 0.0 |
| Plan followed / policy-valid routes / LLM route choices accepted | 1.0 / 1.0 / 0.98 |
| Re-retrieval rate / final topic coverage / graph traversal used | 0.25 / 1.0 / 0.625 |
| Writer self-revision rate / reviewer first-draft pass / reports finalized | 0.125 / 0.875 / 1.0 |
| Citation coverage / LLM-judged faithfulness / word-overlap support | 1.0 / 0.977 / 0.55 |
| Confidence mean (min) / gate | 0.911 (0.886) / 8 of 8 pass |
| Mean latency / LLM calls / MCP tool calls / tokens | 87.2 s / 20.4 / 9.8 / 27,840 |
| Cost per run (list price) / decisions logged / spans | USD 0.0053 / 12.6 / 29.0 |
| Lab extraction (5 real rows): image (vision) / PDF field accuracy; invented values | 1.0 / 1.0; 0 |
| Batch of 3 at concurrency 3: wall time / speed-up | 89.3 s / 2.80x |
| Receipts verify / tampered receipt verifies / audit chains valid | yes / no / yes |

**Documented failure:** an earlier full run finalized only 3/8 reports — the provider's content filter (`finish_reason="content_filter"`) truncated drafts that pasted the CDC BE FAST passage verbatim, and the gate correctly refused them. Fixed by short paraphrased findings and one retry on truncation; a notebook probe did not reproduce it in the final run (0 of 3 isolated prompts filtered), so it is reported as a development observation. See the paper for other limitations.

### Live verification (MCP is real, the workflow is agentic)
`python scripts/live_evidence.py` records the MCP initialize handshake (protocol version, server name, capabilities), the
server's separate PID and its secret-free environment, the `tool_calls` gpt-4o-mini itself returns through `bind_tools`
(executed over MCP), and every orchestrator decision of a full run - see `eval/logs/live_evidence_py314.json`.
A preliminary subgroup audit (`scripts/fairness_check.py` -> `eval/fairness_preliminary.json`) shows similar AUC by sex
but strong age dependence (0 of 6 test strokes under 45 flagged); see [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md).

## Run locally
```bash
git clone https://github.com/sandipdey2/neuroguard-agentic-rag && cd neuroguard-agentic-rag
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # Python 3.13 or 3.14
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add OPENAI_API_KEY (+ OPENAI_API_BASE)
python scripts/verify_mcp.py          # MCP over stdio: tools/resources/prompts list + calls
python scripts/smoke_test.py          # one full multi-agent run
python scripts/live_evidence.py       # MCP handshake/PID + LLM-chosen tool_calls + plan, ReAct, decisions
streamlit run app.py
```
Tests: `python scripts/apptest_session.py` (Streamlit AppTest, several consecutive cases + attacks in one session). Rebuild: `scripts/draw_architecture.py`, `scripts/build_coverage_doc.py`, `scripts/build_pdf.py`, `scripts/build_dist.py` (product zip), `scripts/fetch_data.py` (re-download + checksum data).

## Deploy on Streamlit Community Cloud
New app → repo `sandipdey2/neuroguard-agentic-rag`, branch `main`, file `app.py` → Advanced settings → Secrets:
```toml
OPENAI_API_KEY = "your key"
OPENAI_API_BASE = "https://openai.vocareum.com/v1"
NG_RECEIPT_KEY = "a long random string"
```

## Data (public only — see DATA_SOURCES.md)
Stroke Prediction Dataset (fedesoriano, Kaggle; educational use — excluded from the product zip), MTSamples (CC0), 30 verbatim CDC/NHLBI excerpts (U.S. public domain).

## Limitations & ethics
Educational dataset with no race/ethnicity field and no event timing; estimates are associations, not incidence. Guardrails are regex-based; 16 eval cases are too few for confidence intervals; retrieval labels are author-made; runs take ~1.5 minutes (the planner, re-retrieval and self-critique added ~31 s versus v2.0.1); the confidence score is a heuristic, not a calibrated probability of correctness; the learning loop is bounded to whitelisted hints; multi-modal input reads three lab fields only; Streamlit Cloud storage is ephemeral and not HIPAA-compliant; A2A is in-process. The LLM red team was noisy during development, so its findings are quote-confirmed by code.

## Product
Gumroad listing draft: [docs/GUMROAD_LISTING.md](docs/GUMROAD_LISTING.md) · buyer guide: [docs/BUYER_SETUP.md](docs/BUYER_SETUP.md) · improvement roadmap: [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) · [CHANGELOG.md](CHANGELOG.md)
