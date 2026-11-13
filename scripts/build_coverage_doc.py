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
RM, RT = {m["mode"]: m for m in S["rag_modes"]["modes"]}, S["redteam_context_tools"]
SLOTS = S["rag_modes"]["slot_tokens_mean"]
AS = RM["agentic_search"]


def used(tool):
    return "used in the agentic-search eval runs" if tool in AS["tools_used"] else "not chosen by the LLM in the eval runs (exercised in verify_mcp.py)"
cmp_ = {r["retriever"]: r for r in R["comparison"]}
hy, sp, de, oa = (cmp_[k] for k in ("hybrid 0.6/0.4 (served)", "sparse (TF-IDF)", "dense (LSA 24-d)", "OpenAI text-embedding-3-small"))

EVID = {
    "LLM reasoning": f"notebook s.8-9: {A['llm_calls_mean']} LLM calls / run (mean), finalized {A['finalized_rate']}; scripts/live_evidence.py",
    "Prompt patterns": "notebook s.12 (prompt probe); trace `skill_loaded`, `mcp_prompt_get`; skills/*/SKILL.md",
    "Tool use / function calling": f"live_evidence: finish_reason=tool_calls from bind_tools; {A['tool_calls_mean']} MCP calls / run; verify_mcp.py (19 tools)",
    "Embeddings & vectors": f"notebook s.5b: paraphrase hit@3 sparse {sp['paraphrase_hit@3']} / dense {de['paraphrase_hit@3']} / hybrid {hy['paraphrase_hit@3']} / OpenAI {oa['paraphrase_hit@3']}",
    "Context windows": f"notebook s.5d + s.9: max {A['max_context_tokens']} tokens in any agent context of {CX['model_context_window']:,}; {A['tool_outputs_truncated_total']} tool outputs truncated",
    "Classical RAG": f"notebook s.5: hit@3 {R['hit_at_3']}, MRR {R['mrr_at_3']} on {R['queries']} labelled queries; s.17 classical mode: hit@3 {RM['classical']['hit_at_3']}, coverage {RM['classical']['coverage']}, faithfulness {RM['classical']['faithfulness']}, {RM['classical']['latency_mean_s']} s",
    "Graph RAG": f"notebook s.5c: {G['nodes']} nodes / {G['edges']} edges, factor-query hit@3 {G['hit_at_3']}, precision@3 {G['precision_at_3']}; used in {A['graph_traversal_rate']} of runs",
    "Agentic RAG": f"notebook s.9: re-retrieval in {A['re_retrieval_rate']} of runs, final topic coverage {A['final_coverage_mean']}; s.17: context window {RM['agentic']['evidence_context_tokens_mean']} tokens over 9 slots; trace `context_window`",
    "Agentic search: file search": f"verify_mcp.py search_files; {used('search_files')}; shell reads of Plan.md: {used('shell')}",
    "Skill-loading tool": f"verify_mcp.py load_skill; hook refuses skills outside the agent's catalog (s.6b); {used('load_skill')}",
    "Database tools": f"verify_mcp.py execute_query (GROUP BY hypertension) + DROP refused; s.6b: {RT['sql_attacks']} SQL attacks refused; semantic_search {used('semantic_search')}",
    "Web search (allowlisted)": f"s.6b: {RT['ssrf_attacks']} SSRF URLs refused by hook and server, live CDC fetch status {RT['live_cdc_fetch_status']}, offline degrade = {RT['offline_degrades']}; s.17: {AS['web_search_calls']} web_search calls, {AS['web_pages_live_200']} live pages (HTTP 200)",
    "Memory tool": f"verify_mcp.py get_memory; session_id pinned by the hook; {used('get_memory')}",
    "Shell tool (sandboxed)": f"s.6b: {RT['shell_attacks']} shell-injection attempts refused by hook and server, {RT['positive_controls_allowed']}/{RT['positive_controls']} positive controls allowed; denied in default mode = {RT['shell_denied_for_default_agentic_mode']}",
    "Context-window inspector": "s.17 per-slot token table and eval/rag_modes.png; app result tab 'Context window'; AppTest renders it on py3.13 and py3.14",
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
         "plus the RAG-modes diagram (classical RAG, agentic RAG context assembly, agentic search tools) "
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
lines += ["", "## RAG modes compared (notebook s.17, same normal cases)", "",
          "| Mode | Who retrieves | hit@3 | Coverage | Citation validity | Faithfulness (LLM judge) | Latency mean (s) | Tokens / run | Evidence context tokens | Finalized |",
          "|---|---|---|---|---|---|---|---|---|---|"]
WHO = {"classical": "fixed pipeline before the LLM", "agentic": "LLM tool calls (search_guidelines, graph_traverse)",
       "agentic_search": "LLM tool calls over files, skills, DB, web, memory, shell"}
for m in ("classical", "agentic", "agentic_search"):
    x = RM[m]
    lines.append(f"| {m} | {WHO[m]} | {x['hit_at_3']} | {x['coverage']} | {x['citation_validity']} | {x['faithfulness']} | {x['latency_mean_s']} | "
                 f"{x['tokens_mean']:,} | {x['evidence_context_tokens_mean']} | {x['finalized']}/{x['runs']} |")
lines += ["", "Mean Evidence Researcher context-window tokens per slot (tiktoken o200k_base):", "",
          "| Mode | " + " | ".join(sorted({k for v in SLOTS.values() for k in v if k != "total"})) + " | total |",
          "|---|" + "---|" * (len({k for v in SLOTS.values() for k in v if k != "total"}) + 1)]
for m, v in SLOTS.items():
    keys = sorted({k for vv in SLOTS.values() for k in vv if k != "total"})
    lines.append(f"| {m} | " + " | ".join(str(int(v.get(k, 0))) for k in keys) + f" | {int(v['total'])} |")
lines += ["", f"Red team for the context tools (s.6b): {RT['attacks_total']} attacks ({RT['shell_attacks']} shell injection, {RT['ssrf_attacks']} SSRF, "
          f"{RT['sql_attacks']} SQL write/escape) - {RT['hook_blocked']} blocked by the pre-tool-use hook and {RT['server_refused']} refused again inside the "
          f"server; {RT['positive_controls_allowed']}/{RT['positive_controls']} legitimate calls allowed.", ""]
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
          "- Web search is not a general search engine: it ranks the corpus' known CDC / NIH / WHO source pages and fetches them live; "
          "web text is context, never a citation. On the development box DNS answers come from a fake-IP proxy (198.18.0.0/15), which "
          "the SSRF guard accepts only when NG_FAKEIP_DNS=1 is set; Streamlit Cloud resolves normally and needs no flag.",
          "- The shell tool is an argv allowlist sandbox (grep / wc / head / ls, no interpreter, empty env, timeout), not an OS container.",
          "- In agentic-search mode the harness sends one 'source check' message if the LLM finished without consulting a required source "
          "class; the LLM still chooses the call. Without it gpt-4o-mini often stopped after semantic_search.",
          ""]
(ROOT / "docs" / "KNOWLEDGE_GRAPH_COVERAGE.md").write_text("\n".join(lines))
print("wrote docs/KNOWLEDGE_GRAPH_COVERAGE.md")
