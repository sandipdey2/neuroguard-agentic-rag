"""Write docs/KNOWLEDGE_GRAPH_COVERAGE.md from core/coverage.NODES + eval/summary.json (numbers cannot drift).
    python scripts/build_coverage_doc.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.coverage import NODES  # noqa: E402

S = json.loads((ROOT / "eval" / "summary.json").read_text())
K = json.loads((ROOT / "harness_contract.json").read_text())
import csv  # noqa: E402
_rows = list(csv.DictReader((ROOT / "eval" / "results_table.csv").open()))     # per-case table written by the notebook
THOUGHTS = f"{sum(int(r['react_thoughts_with_text']) for r in _rows)} of {sum(int(r['react_steps']) for r in _rows)}"
A, R, G, CX, LR, MM, B, KG = S["agent"], S["retrieval"], S["graph_rag"], S["context"], S["learning"], S["multimodal"], S["batch"], S["kg_coverage"]
cmp_ = {r["retriever"]: r for r in R["comparison"]}
hy, sp, de, oa = (cmp_[k] for k in ("hybrid 0.6/0.4 (served)", "sparse (TF-IDF)", "dense (LSA 24-d)", "OpenAI text-embedding-3-small"))

EVID = {
    "LLM reasoning": f"notebook s.8-9: {A['llm_calls_mean']} LLM calls / run (mean), finalized {A['finalized_rate']}; scripts/live_evidence.py",
    "Prompt patterns": "notebook s.12 (prompt probe); trace `skill_loaded`, `mcp_prompt_get`; skills/*/SKILL.md",
    "Tool use / function calling": f"live_evidence: finish_reason=tool_calls from bind_tools; {A['tool_calls_mean']} MCP calls / run; verify_mcp.py (11 tools)",
    "Embeddings & vectors": f"notebook s.5b: paraphrase hit@3 sparse {sp['paraphrase_hit@3']} / dense {de['paraphrase_hit@3']} / hybrid {hy['paraphrase_hit@3']} / OpenAI {oa['paraphrase_hit@3']}",
    "Context windows": f"notebook s.5d + s.9: max {A['max_context_tokens']} tokens in any agent context of {CX['model_context_window']:,}; {A['tool_outputs_truncated_total']} tool outputs truncated",
    "Classical RAG": f"notebook s.5: hit@3 {R['hit_at_3']}, MRR {R['mrr_at_3']} on {R['queries']} labelled queries; citation coverage {A['citation_coverage']}",
    "Graph RAG": f"notebook s.5c: {G['nodes']} nodes / {G['edges']} edges, factor-query hit@3 {G['hit_at_3']}, precision@3 {G['precision_at_3']}; used in {A['graph_traversal_rate']} of runs",
    "Agentic RAG": f"notebook s.9: re-retrieval in {A['re_retrieval_rate']} of runs, final topic coverage {A['final_coverage_mean']}; trace `evidence_sufficiency`",
    "Short-term memory": "LangGraph CaseState; every trace event and decision in eval/raw_results.json",
    "Long-term memory": "notebook s.8 case N1 (consent) writes, s.6 hook blocks writes without consent; UI Memory tab",
    "Episodic memory": f"notebook s.13: {LR['episodes_after_eval']} episodes after the evaluation; reruns recalled similar episodes",
    "ReAct loop": f"notebook s.9: {A['react_steps_mean']} thought/action/observation steps per run, {THOUGHTS} eval steps with explicit thought text; trace `react_step`",
    "Planner -> executor": f"notebook s.9: executed order matched the plan in {A['plan_adherence_rate']} of runs; trace `plan`",
    "Multi-agent orchestration": f"notebook s.9: policy-valid routes {A['route_policy_valid_rate']}, LLM route choices accepted {A['llm_route_choices_accepted']}",
    "Context graph traversal": f"notebook s.5c: multi-hop paths (e.g. tia query has {G['hop2_chunks_for_tia']} hop-2 chunks); MCP `graph_traverse` + `graph://knowledge-graph`",
    "Tool selection & routing": "notebook s.6: hook denials, argument clamps, pinned session_id; s.9 route overrides",
    "Reflection / self-critique": f"notebook s.9: writer self-revised in {A['self_critique_revision_rate']} of runs; first-draft reviewer pass {A['first_draft_pass_rate']}",
    "Confidence scoring": f"notebook s.9: confidence mean {A['confidence_mean']} (min {A['confidence_min']}), gate {A['confidence_gate_counts']}",
    "Policy gates": f"notebook s.8: attacks blocked {A['attack_block_rate']}, false blocks {A['false_block_rate_normal']}; finalize gate + confidence gate",
    "Audit trail design": f"notebook s.6: tampered receipt verifies = {S['harness_unit']['tampered_receipt_verifies']}, chain valid after tamper = {S['harness_unit']['audit_chain_valid_after_tamper']}",
    "Decision reasoning logs": f"notebook s.9: {A['decisions_mean']} decisions per run; SHA-256 of the log in every receipt",
    "Eval & observability": f"evaluation.ipynb + {A['spans_mean']} spans / run exported as OTLP-JSON (UI Observability tab)",
    "Cost / token economics": f"notebook s.9: USD {A['cost_usd_mean']} per run at list price ({A['tokens_mean']} tokens mean)",
    "Memory & learning loops": f"notebook s.13: learned hints {LR['learned_hints']}; {LR['boosted_excerpts']} boosted excerpts after {LR['demo_approvals']} demo approvals (cap {LR['boost_cap']})",
    "Self-improving agents": "notebook s.13 reruns (n=2, illustrative): hints and recalled episodes reach planner and writer",
    "Multi-modal reasoning": f"notebook s.14: image field accuracy {MM['image_field_accuracy']}, PDF {MM['pdf_field_accuracy']} on {MM['documents']} real rows; invented BP values {MM['invented_bp_values']}",
    "Agentic workflows at scale": f"notebook s.15 / eval/logs/batch_demo.json: {B['n']} cases, concurrency {B['max_concurrency']}, speed-up {B['speedup']}x, {B['finalized']}/{B['n']} finalized",
}

lines = ["# Agentic-AI knowledge graph -> NeuroGuard coverage", "",
         "**Repository:** https://github.com/sandipdey2/neuroguard-agentic-rag · **Live demo:** https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/", "",
         "Every node of the agentic-AI knowledge graph (Foundations, Retrieval & Memory, Agent patterns, Production & governance, Frontier) "
         "mapped to the file/function that implements it, how it works, and the evidence/test that shows it working. "
         "This file is generated by `scripts/build_coverage_doc.py` from `core/coverage.py` and `eval/summary.json`, so the numbers "
         "match the executed notebook. The same table, with live evidence from your own run, is the **Knowledge-graph coverage** tab of the app.", "",
         f"**Status: {KG['nodes']} nodes - {KG['full']} full, {KG['partial']} partial ({', '.join(KG['partial_nodes'])}).** "
         "\"Partial\" means implemented and tested but deliberately limited; the limit is stated.", ""]
grp = None
for g, n, st, where, how, note in NODES:
    if g != grp:
        lines += ["", f"## {g}", "", "| Node | Status | File / function | How it works | Evidence / test |", "|---|---|---|---|---|"]
        grp = g
    lim = f" **Limit:** {note}" if note else ""
    lines.append(f"| {n} | {st} | `{where}` | {how}.{lim} | {EVID[n]} |")
lines += ["", "## How the new pieces fit one run", "",
          "1. **Planner** calls MCP `episode_recall` (similar de-identified past episodes + learned hints) and writes a structured plan.",
          "2. **Orchestrator** executes the plan step by step (LLM choice, code-enforced eligibility); every choice goes to the decision log.",
          "3. **Evidence Researcher** runs a ReAct loop over `search_guidelines` (hybrid sparse + dense vectors) and `graph_traverse` "
          "(multi-hop graph RAG). Code measures which required topics (risk factors present + warning signs) are covered by cited "
          "excerpts; if any are missing it re-retrieves once with a reformulation instruction (agentic RAG).",
          "4. **Clinical Writer** receives evidence packed to a token budget (tiktoken), learned hints, then critiques its own draft "
          "(deterministic checks + an LLM critique whose quotes must exist in the draft and show the rule's surface form) and revises once.",
          "5. **Safety Reviewer** (12-check verifier + quote-confirmed red team) -> **finalize gate** -> **confidence gate** "
          f"(refuse < {K['confidence']['refuse_below']}, clinician review < {K['confidence']['review_below']}).",
          "6. The harness writes one de-identified **episode** (MCP `episode_write`, harness-only), prices every LLM call, exports spans "
          "as OTLP-JSON and signs a receipt that includes the decision-log hash, cost and confidence. A clinician's approve/reject is "
          "written back to the episode and feeds the bounded learning loop.", "",
          "## Honest limits", "",
          "- Dense vectors inside the MCP server are local LSA, not OpenAI embeddings, because the server deliberately holds no API key; "
          "the notebook compares both.",
          "- The confidence score is a transparent weighted heuristic, not a calibrated probability that the report is correct.",
          "- The learning loop changes only whitelisted hint texts and bounded retrieval boosts; it never edits code, prompts, the contract or model weights.",
          "- Multi-modal input reads three lab fields; scanned PDFs (no text layer) are refused; it was tested on rendered images of real dataset values, not real scans.",
          "- Batch mode is single-process concurrency for a small cohort, not a distributed queue.",
          ""]
(ROOT / "docs" / "KNOWLEDGE_GRAPH_COVERAGE.md").write_text("\n".join(lines))
print("wrote docs/KNOWLEDGE_GRAPH_COVERAGE.md")
