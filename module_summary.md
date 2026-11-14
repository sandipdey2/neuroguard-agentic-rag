# NeuroGuard AI: A Governed Multi-Agent System for Stroke-Risk Clinical Decision Support

Sandip Dey · Udacity AI Mastery Capstone, Project 7 (Integrative Industry Synthesis) · Code: https://github.com/sandipdey2/neuroguard-agentic-rag · Live demo: https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/

## Overview and Industry Context

Stroke is a leading cause of serious long-term disability in the United States, where more than 795,000 people have a stroke every year (Centers for Disease Control and Prevention [CDC], 2024). Its main drivers are measurable in primary care, yet clinicians have minutes per visit to combine a risk estimate, guideline wording and a clear explanation. Generative AI could draft that synthesis, but here an invented number or dosing suggestion is a safety event.

NeuroGuard AI is a decision-support prototype that turns a structured patient profile and an optional free-text note into a cited, verified stroke-risk report for a clinician or a patient. It is deliberately not a medical device. Under the U.S. Food and Drug Administration's (2022) guidance, software that supports a professional's judgement must let that professional review the basis of each recommendation. Every number traces to a tool output, every guideline claim cites a public-domain excerpt, and a gate refuses failing reports. All data are public: the Stroke Prediction Dataset (fedesoriano, 2021; 5,110 records, 5,109 after cleaning), MTSamples transcriptions (Boyle, n.d.; CC0) and 30 verbatim guideline excerpts.

## Integration Rationale

Six earlier capstones each solve one part of the workflow.

| Prior project | Contribution to NeuroGuard |
|---|---|
| P1 Data workflow (Titanic) | Explicit cleaning steps with a log: schema check, "N/A" BMI to missing, rare-category filter, imputation fitted only on the training split (`core/data.py`). |
| P2 Statistical analysis (heart failure, Welch's t-test) | Chi-square and Welch tests, odds ratios with 95% CIs and Wilson intervals; served to agents by the `cohort_stats` MCP tool. |
| P3 Classification (breast cancer, logistic regression) | Logistic regression, random forest and gradient boosting with class weighting; calibrated logistic model served with exact factor contributions. |
| P4 Sequence model (IMDB LSTM) | Text modelling for notes: TF-IDF + logistic regression on MTSamples, plus CDC BE FAST red-flag rules. |
| P5 Generative model (MNIST GAN) | Generation moved from pixels to language: the Clinical Writer drafts from an MCP prompt. |
| P6 Agentic RAG | Retrieval and an agent loop, now with a real MCP server, multiple agents and enforced verification. |

The integration is causal, not decorative. P2's odds ratios (age 65+: 8.13, hypertension: 3.7, heart disease: 4.71) justify which factors the evidence agent searches for; P3 supplies the number they contextualize; P4's rules guard the front door. A Project 6 reviewer noted that I had called an in-process class an "MCP server" and a single routing step a loop. Here the server is a separate process that speaks the Model Context Protocol (Anthropic, 2024) over stdio, and the orchestrator loop is a real graph cycle with a cap.

## System Design and Technical Decisions

Figure 1 shows the architecture as implemented; every box names the file that implements it.

![NeuroGuard AI architecture as implemented](architecture_diagram.png)

**Harness and contract.** Before the first token, `harness_contract.json` is loaded and SHA-256 hashed. It fixes the model (gpt-4o-mini via the Vocareum OpenAI-compatible endpoint, temperature 0, top-p 1.0), nine orchestrator turns, budgets of 36 LLM calls, 32 tool calls, 110,000 tokens and 300 seconds, each agent's permitted tools, twelve required checks, context budgets, a confidence policy, list prices and eleven measurable success criteria. The hash goes into the run receipt.

**Planner and multi-agent loop.** The LangGraph state machine has planner, executor and validator phases. A planner LLM recalls similar de-identified past episodes and writes a structured plan. An orchestrator LLM executes it, choosing the next agent, a skill and an instruction; five specialists—Intake, Risk Analyst, Evidence Researcher, Clinical Writer and Safety Reviewer—return to it after every step. Code enforces eligibility: the gathering agents run in any order, the writer only after all three, the reviewer after each draft, and FINISH only after review, with one revision allowed. Tool-using agents run a bounded, traced ReAct loop (Yao et al., 2023). The writer critiques its own draft before review—deterministic checks plus an LLM critique whose quotes must exist in the draft—and revises once. Delegation uses A2A-style Agent Cards and task/message/artifact objects (A2A Project, 2025) on an in-process bus.

**Real MCP.** The official SDK server exposes twenty tools (domain, context-retrieval and a keyless PubMed search), six resources including a tool registry, a per-excerpt template and two prompts. A server-side policy engine re-checks every call, and the server never receives the API key.

**Permissions, memory and skills.** Each agent is bound only to its own tools, and a pre-tool-use hook re-checks every call, clamps arguments, pins the session identifier so no agent can read another session, enforces the tool budget and blocks `memory_write` without consent. Short-term memory is the LangGraph state; long-term memory is a consent-based SQLite or PostgreSQL table of de-identified summaries. Skills are `SKILL.md` files: only their names and descriptions are offered to the orchestrator, and the full body is injected when one is selected (progressive disclosure).

**Retrieval and modelling choices.** Retrieval follows retrieve-then-generate (Lewis et al., 2020) in three layers. Hybrid search fuses TF-IDF (weight 0.6) with 24-dimensional LSA vectors in a FAISS index (0.4); the notebook compares them with OpenAI embeddings. A networkx graph of 49 factor, condition and excerpt nodes supports multi-hop traversal with explainable paths. Code checks that every required topic is covered by a cited excerpt and, if not, the agent retrieves again. Three RAG modes are selectable: classical (a fixed pipeline calls semantic search before the LLM and injects the documents), agentic (the LLM issues search calls; its context window is assembled from named slots shown with token counts in the app) and agentic search (tools over local files, skills, a read-only cohort database, allowlisted CDC/NIH/WHO pages, memory and a sandboxed shell). Shell and web are the riskiest, so the hook and the server both refuse metacharacters, non-allowlisted hosts, private addresses and write SQL. For risk, cross-validated ROC-AUC was 0.838 (logistic regression), 0.835 (random forest) and 0.822 (gradient boosting). I served the logistic model with sigmoid calibration because it was best and its log-odds contributions are exact; calibration matters because class weighting inflates probabilities (Niculescu-Mizil & Caruana, 2005). For notes, TF-IDF replaced the P4 LSTM for CPU speed.

**Verifier, red team, gate and receipt.** The Safety Reviewer runs deterministic invariants—risk equal to a `predict_risk` call on the exact profile, citations actually retrieved, every number grounded in a tool output or excerpt, no dosing, diagnosis or PHI, disclaimer present—plus an LLM red-team pass. Because the gpt-4o-mini red team flagged harmless sentences, its findings count only when the quoted sentence occurs in the draft. The finalize gate re-runs the checks and refuses to finalize on any failure; high-risk results and notes with red-flag signs require clinician sign-off. A confidence score (calibration, cohort size, topic coverage, verifier, red team, grounding) refuses below 0.4 and requests review below 0.7. Each run ends with an HMAC-SHA256 receipt over hashes of inputs, report, trace, checks and a decision reasoning log, plus a hash-chained audit log, OTLP-JSON spans and a list-price cost.

**Learning, modality and scale (partial).** Each run stores a de-identified episode; reviewer feedback becomes whitelisted hints and clinician approvals become retrieval boosts capped at 0.06, so learning is bounded and reversible. Lab values can be read from an image or a text PDF.

**Service layer (v2.3).** A FastAPI service, an HMAC webhook and a cron script feed a durable queue (SQLite, PostgreSQL or Redis) drained by workers. A chat tab and a Slack bot keep a redacted conversation memory; follow-ups call an evidence sub-agent exposed as a tool. Docker Compose and Kubernetes manifests are validated but not cluster-tested. `docs/KNOWLEDGE_GRAPH_COVERAGE.md` maps 34 concepts (29 full, 5 partial) and two architecture checklists to code.

## Ethical Considerations and Responsible AI

The risks are specific to this system. First, **harm from false reassurance or delay**: a person describing sudden face drooping must not receive a risk report. Input guardrails therefore run before any LLM call, and the BE FAST rule stops the request with CDC's "call 9-1-1" wording (CDC, 2026). Second, **unsafe advice**: dosing questions are refused at input, and dosing or diagnosis language fails the verifier at output. Third, **privacy**: identifiers are redacted before processing, memory stores only de-identified summaries with explicit consent, and Streamlit Community Cloud is documented as unsuitable for real PHI. Fourth, **bias and data provenance**: the dataset is for educational use, its binary label records whether a patient had a stroke without any timing (so estimates describe association with stroke history, not future incidence), and it contains no race or ethnicity field, so the model cannot be audited for the kind of group bias that Obermeyer et al. (2019) found in a deployed health algorithm. Fifth, **accountability and transparency**, mapped to the Govern, Map, Measure and Manage functions of the NIST AI Risk Management Framework (National Institute of Standards and Technology [NIST], 2023) and to the WHO's (2021) call for human oversight: the contract, audit chain, signed receipt and approval step make every run reviewable by a named human.

## Evaluation and Reflection

`evaluation.ipynb` was executed against the real endpoint and MCP server on eight real held-out profiles and eight attacks. The risk model's held-out results were ROC-AUC 0.84, PR-AUC 0.26 (prevalence 0.0487) and Brier score 0.041; at the recall-oriented threshold it reached recall 0.8 but precision only 0.137, which is acceptable for a prompt to discuss risk but would be unacceptable as a screening decision (Saito & Rehmsmeier, 2015). The note classifier is weak (F1 0.31), so it is advisory only.

| Agent metric | Result |
|---|---|
| Attacks blocked (injection, tool misuse, emergency, dosing, off-topic) | 1.0 (reason correct 1.0) |
| Normal requests wrongly blocked | 0.0 |
| Plan followed / LLM route choices accepted without override | 1.0 / 0.976 |
| Context-tool attacks blocked (shell, SSRF, SQL) | 44 / 44 |
| Reports finalized / reviewer first-draft pass | 0.875 / 0.875 |
| Citation coverage / LLM-judged faithfulness (Es et al., 2024) | 1.0 / 1.0 |
| RAG modes hit@3: classical / agentic / agentic search | 1.0 / 0.933 / 0.929 |
| Mean confidence / cost per run (list price) | 0.914 / USD 0.0047 |
| Median latency / LLM calls / MCP tool calls | 84.5 s / 18.1 / 10.1 |

All eleven success criteria in the contract were met (`eval/success_criteria.md`), for example attack block rate 1.0 (target 1.0), finalized rate 0.875 (≥ 0.8), Brier 0.041 (≤ 0.05), median latency 84.5 s (≤ 120 s), the highest end-to-end cost USD 0.0077 (≤ 0.01) and 79/79 unit tests.

The most useful finding was a failure. An earlier run finalized only three of eight reports: the writer pasted the long BE FAST passage verbatim, the provider stopped generation with `finish_reason="content_filter"`, and the gate correctly refused the truncated drafts. Short paraphrased findings and one retry fixed it; paraphrasing keeps word-overlap support low (0.55) while LLM-judged faithfulness is 1.0, so a lexical proxy should not be the only grounding metric. In the final run one case hit the 330-second timeout and was stopped with a signed receipt. Comparing modes, classical RAG was fastest (71 s) and cheapest because this corpus is small and topics are known in advance; agentic search doubled tokens (49,867 per run) for the same coverage. Adding the planner raised accepted route choices from 0.82 to 0.98, and dense vectors fixed the paraphrase query sparse search missed. Other weaknesses: regex guardrails, sixteen cases, self-labelled queries and a heuristic confidence score. Next I would validate on a licensed cohort.

## Professional Relevance

This project reflects how I would ship AI in a regulated setting: start from a contract, give each agent least privilege, verify outputs deterministically, keep a human approval step and leave a signed record. It ships as a live Streamlit app and an MIT-licensed starter kit. Before any clinical pilot I would add external validation and a compliance review.

## References

A2A Project. (2025). *Agent2Agent (A2A) protocol specification*. The Linux Foundation. https://a2a-protocol.org/latest/specification/

Anthropic. (2024, November 25). *Introducing the Model Context Protocol*. https://www.anthropic.com/news/model-context-protocol

Boyle, T. (n.d.). *Medical transcriptions* [Data set]. Kaggle. https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

Centers for Disease Control and Prevention. (2024, October 24). *Stroke facts*. https://www.cdc.gov/stroke/data-research/facts-stats/index.html

Centers for Disease Control and Prevention. (2026, May 19). *Signs and symptoms of stroke*. https://www.cdc.gov/stroke/signs-symptoms/index.html

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. In *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations* (pp. 150–158). Association for Computational Linguistics. https://aclanthology.org/2024.eacl-demo.16/

fedesoriano. (2021). *Stroke prediction dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474.

National Institute of Standards and Technology. (2023). *Artificial intelligence risk management framework (AI RMF 1.0)* (NIST AI 100-1). https://doi.org/10.6028/NIST.AI.100-1

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). https://doi.org/10.1145/1102351.1102430

Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science, 366*(6464), 447–453. https://doi.org/10.1126/science.aax2342

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE, 10*(3), Article e0118432. https://doi.org/10.1371/journal.pone.0118432

U.S. Food and Drug Administration. (2022). *Clinical decision support software: Guidance for industry and Food and Drug Administration staff*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software

World Health Organization. (2021). *Ethics and governance of artificial intelligence for health: WHO guidance*. https://www.who.int/publications/i/item/9789240029200

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. In *International Conference on Learning Representations*. https://arxiv.org/abs/2210.03629
