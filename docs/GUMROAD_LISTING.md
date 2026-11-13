# Gumroad listing draft - NeuroGuard AI

> Draft copy for the seller to paste into Gumroad. Prices are suggestions. Nothing here has been published.
> No testimonials, user counts or clinical-outcome claims are made, because none exist.

## Title
NeuroGuard AI - Multi-Agent Clinical Decision-Support Starter Kit (MCP + RAG + Guardrails + Signed Receipts)

## Tagline
A working, tested reference app that shows how to build *safe* agentic AI for healthcare: a real MCP server,
cited guideline RAG, least-privilege tools, a verifier that can say no, and a signed receipt for every run.

## Description
NeuroGuard AI is a complete Python / Streamlit codebase for a stroke-risk **decision-support** demo. An LLM
orchestrator (LangGraph, gpt-4o-mini by default, any OpenAI-compatible endpoint) delegates to five specialist
agents - Intake, Risk Analyst, Evidence Researcher, Clinical Writer and Safety Reviewer. The agents can act only
through a real Model Context Protocol server built with the official SDK, and every tool call passes a
pre-tool-use permission hook. Reports cite U.S. public-domain CDC/NHLBI guideline excerpts, a deterministic
verifier checks twelve invariants, and a finalize gate refuses to release anything that fails. Each run ends with
an HMAC-SHA256 signed receipt and a hash-chained audit log.

It is built for developers, AI engineers, health-tech founders and educators who need a concrete, inspectable
pattern for trustworthy agent systems - not a black box.

## What is inside (feature bullets)
- **Real MCP server** (stdio): 8 tools, 5 resources (including a URI template), 2 prompts; verification script included.
- **Multi-agent orchestration**: LLM routing with code-enforced eligibility rules, turn cap, revision loop.
- **A2A-style agent cards** and task/message/artifact objects on an in-process bus (easy to move to HTTP).
- **Harness contract** (`harness_contract.json`): budgets, timeouts, per-agent tool permissions, required checks.
- **Guardrails**: PHI redaction, prompt-injection and tool-misuse blocking, dosing-question refusal,
  BE FAST emergency stop with a call-9-1-1 message, sandboxed AST calculator.
- **Verifier + red team**: deterministic invariants (risk range, number grounding, citations, no dosing, no
  diagnosis, disclaimer, no PHI, budget) plus an LLM red-team pass whose findings are quote-confirmed.
- **Skills with progressive disclosure** (`SKILL.md` files) you can edit without touching code.
- **Explainable ML**: calibrated logistic regression compared with random forest and gradient boosting, exact
  per-factor contributions, cohort statistics with confidence intervals, model card resource.
- **Consent-based memory** (SQLite) with a one-click clear.
- **Polished Streamlit UI**: live trace, A2A log, MCP calls, citations, charts, clinician approval, PDF/Markdown
  report and signed receipt downloads.
- **Executed evaluation notebook** with real results, including documented failure cases.
- Pinned requirements tested on Python 3.13 and 3.14; one-click Streamlit Community Cloud deployment.

## What the buyer gets
- `neuroguard-ai-2.0.0.zip`: full source (MIT license), skills, guideline corpus, MTSamples subset (CC0),
  evaluation notebook and results, architecture diagram, technical paper (PDF), buyer setup guide, changelog.
- A data fetch script for the Stroke Prediction Dataset, which is **not bundled** because its Kaggle terms are
  educational use; bring your own licensed data for commercial deployments.
- No API key is included. You need your own OpenAI-compatible key.

## Suggested pricing tiers
| Tier | Suggested price | Contents |
|---|---|---|
| Developer | $49 | Source zip, docs, notebook, one developer |
| Team | $149 | Same, use across one company or team; editable diagram script and paper sources |
| Educator | $99 | Same as Developer, permission to use in classes and workshops |
(Free updates for 2.x. Support by email on a best-effort basis - state your actual response time before publishing.)

## Setup in 5 steps
1. Unzip, `pip install -r requirements.txt` (Python 3.13/3.14).
2. `python scripts/fetch_data.py` (accept the dataset terms) or add your own data.
3. Put your key in `.streamlit/secrets.toml`.
4. `python scripts/verify_mcp.py` then `streamlit run app.py`.
5. Optional: deploy to Streamlit Community Cloud (see BUYER_SETUP.md).

## Disclaimers (keep these on the listing)
- **Not a medical device.** NeuroGuard AI has no FDA, CE or other regulatory clearance and is not intended for
  diagnosis, treatment, triage or any clinical use without independent validation and regulatory clearance.
- The included model is trained on a small public educational dataset; its estimates are associations and are
  not validated for any real population.
- You are responsible for HIPAA/GDPR compliance, data licensing and security in your own deployment.
- LLM outputs can be wrong; the verifier reduces but does not eliminate this risk.
- Software is provided "as is" under the MIT license.

## Tags
ai agents, mcp, model context protocol, langgraph, rag, streamlit, healthcare ai, guardrails, responsible ai, python
