"""Write evaluation.ipynb (cells below); execute with:
    jupyter nbconvert --to notebook --execute --inplace evaluation.ipynb --ExecutePreprocessor.timeout=3600"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# NeuroGuard AI - evaluation notebook

**Repository:** https://github.com/sandipdey2/neuroguard-agentic-rag · **Live demo:** https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/


Executed end to end against the **real** gpt-4o-mini endpoint (Vocareum, temperature 0) and the **real** MCP server
(stdio subprocess). Sections: data pipeline (P1), statistics (P2), risk model (P3), note screening (P4), retrieval,
harness unit checks (permissions, sandbox, receipt, audit chain), MCP protocol check, end-to-end agent runs on
`eval/cases.json` (8 real held-out profiles + 8 attacks), agent metrics, RAGAS-style LLM-judged faithfulness and
failure cases. Every number quoted in README.md and module_summary.md comes from the outputs below
(also saved to `eval/summary.json`).""")
code("""import asyncio, json, os, re, sys, time, statistics, platform
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from core.config import load_contract, get_credentials
from core.harness import sha
C = load_contract()
print("python", platform.python_version(), "| model", C["model"], "| temperature", C["temperature"],
      "| contract sha256", sha(C)[:16], "| key configured:", bool(get_credentials()[0]))""")
md("## 1. Data pipeline (Project 1 lineage)")
code("""from core.data import load_stroke, pipeline_log
df, _ = load_stroke()
display(pd.DataFrame(pipeline_log(), columns=["step", "result"]))
print(df.shape, "missing BMI:", int(df.bmi.isna().sum()))""")
md("## 2. Statistical analysis (Project 2 lineage)")
code("""from core.stats import factor_tests
ft = factor_tests()
display(pd.DataFrame(ft["categorical"]).drop(columns=["stroke_rate_pct_by_level"]))
display(pd.DataFrame(ft["numeric"]))
display(pd.DataFrame(ft["odds_ratios"]))""")
md("## 3. Risk model (Project 3 lineage): comparison, held-out metrics, calibration")
code("""from core.model import get_model
from sklearn.metrics import roc_curve
m = get_model(); rep = m.report
display(pd.DataFrame(rep["comparison"]).T)
print("served:", rep["served_model"]); print("threshold rule:", rep["threshold_rule"])
display(pd.DataFrame([rep["test_metrics"]]).drop(columns=["confusion"])); print("confusion:", rep["test_metrics"]["confusion"])
y = np.array([a for a, _ in m.test_probs]); p = np.array([b for _, b in m.test_probs])
fpr, tpr, _ = roc_curve(y, p)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", c="gray"); ax[0].set_title(f"ROC (test AUC {rep['test_metrics']['roc_auc']})"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
cc = rep["calibration_curve"]; ax[1].plot(cc["mean_predicted"], cc["observed_rate"], "o-"); mx = max(cc["mean_predicted"] + cc["observed_rate"])
ax[1].plot([0, mx], [0, mx], "--", c="gray"); ax[1].set_title(f"Reliability (Brier {rep['test_metrics']['brier']})"); ax[1].set_xlabel("mean predicted"); ax[1].set_ylabel("observed")
plt.tight_layout(); plt.savefig("eval/model_roc_calibration.png", dpi=120); plt.show()
print("coefficients:", rep["coefficients"])""")
md("## 4. Note screening (Project 4 lineage)")
code("""from core.notes import get_note_classifier, red_flags
nc = get_note_classifier(); print(json.dumps(nc.report, indent=1))
probes = {"Sudden numbness or weakness in the face, arm, or leg, especially on one side of the body.": True,
          "Since this morning his face is drooping and his speech is slurred.": True,
          "Worst headache of my life, started an hour ago.": True,
          "Denies weakness, no facial droop, no speech problems.": False,
          "Follow-up for blood pressure; occasional dizziness when standing.": False,
          "Left knee pain after a fall last month.": False}
rows = [{"text": t, "expected_emergency": e, "rule_emergency": red_flags(t)["possible_emergency"]} for t, e in probes.items()]
display(pd.DataFrame(rows)); print("red-flag rule agreement on probes:", sum(r["expected_emergency"] == r["rule_emergency"] for r in rows), "/", len(rows))""")
md("## 5. Retrieval quality (hit@3, MRR) on hand-labelled queries")
code("""from core.rag import get_index
idx = get_index()
qrels = {"high blood pressure stroke risk": {"G-03", "G-18", "G-25"}, "diabetes glucose stroke": {"G-06", "G-19", "G-27"},
         "obesity BMI stroke": {"G-07", "G-16", "G-26"}, "smoking tobacco stroke": {"G-08", "G-17"},
         "age stroke risk doubles after 55": {"G-10", "G-29"}, "heart disease atrial fibrillation stroke": {"G-05", "G-20", "G-28"},
         "warning signs call 9-1-1": {"G-12", "G-13", "G-14", "G-24"}, "BE FAST face arm speech time": {"G-13"},
         "mini-stroke TIA": {"G-15", "G-24", "G-02"}, "physical activity diet alcohol prevention": {"G-09", "G-16", "G-17"},
         "women pregnancy birth control stroke": {"G-11", "G-30"}, "how many strokes each year in the United States": {"G-21"}}
hits, rr = [], []
for q, rel in qrels.items():
    ids = [r["id"] for r in idx.search(q, k=3)]
    hits.append(any(i in rel for i in ids)); rr.append(next((1 / (k + 1) for k, i in enumerate(ids) if i in rel), 0.0))
    print(f"{q[:45]:45s} -> {ids}  hit={hits[-1]}")
retrieval = {"queries": len(qrels), "hit_at_3": round(float(np.mean(hits)), 3), "mrr_at_3": round(float(np.mean(rr)), 3)}
print(retrieval)""")
md("## 6. Harness unit checks: permission engine, pre-tool-use hook, sandbox, receipt, audit chain")
code("""from core.harness import RunContext, sign_receipt, verify_receipt
ctx = RunContext(session_id="nb", consent=False)
probes = [("risk_analyst", "memory_write", {}), ("intake", "predict_risk", {}), ("harness", "memory_write", {}),
          ("risk_analyst", "calculator", {"expression": "1+" * 80 + "1"}), ("intake", "memory_read", {"session_id": "someone-else"}),
          ("evidence_researcher", "search_guidelines", {"query": "x", "k": 50})]
hook = []
for a, t, args in probes:
    d = ctx.perms.pre_tool_use(a, t, args); hook.append({"agent": a, "tool": t, "allow": d.allow, "reason": d.reason, "args_after_hook": d.args})
display(pd.DataFrame(hook))
r = sign_receipt({"run_id": "nb", "report_sha256": sha("x")}); tampered = {**r, "report_sha256": sha("y")}
unit = {"hook_denials_expected": 4, "hook_denials_observed": sum(not h["allow"] for h in hook),
        "session_id_pinned": hook[4]["args_after_hook"]["session_id"] == "nb", "k_clamped_to": hook[5]["args_after_hook"]["k"],
        "receipt_verifies": verify_receipt(r), "tampered_receipt_verifies": verify_receipt(tampered), "audit_chain_valid": ctx.audit.verify_chain()}
ctx.audit.entries[1]["decision"] = "allow"; unit["audit_chain_valid_after_tamper"] = ctx.audit.verify_chain()
print(unit)""")
md("## 7. MCP protocol check (real stdio server)")
code("""from core.mcp_client import mcp_session
async def mcp_check():
    async with mcp_session() as s:
        t = await s.list_tools(); r = await s.list_resources(); rt = await s.list_resource_templates(); p = await s.list_prompts()
        calc = await s.call_tool("calculator", {"expression": "__import__('os').system('ls')"})
        res = await s.read_resource("guideline://G-12")
        return {"tools": [x.name for x in t.tools], "resources": [str(x.uri) for x in r.resources],
                "resource_templates": [x.uriTemplate for x in rt.resourceTemplates], "prompts": [x.name for x in p.prompts],
                "calculator_injection_result": json.loads(calc.content[0].text), "resource_G12_url": json.loads(res.contents[0].text)["url"]}
mcp_info = await mcp_check()
print(json.dumps(mcp_info, indent=1))""")
md("## 8. End-to-end runs (real LLM + real MCP) on `eval/cases.json`")
code("""from core.pipeline import arun_case
cases = json.loads(Path("eval/cases.json").read_text())
runs = []
for c in cases:
    t0 = time.time()
    r = await arun_case(c["profile"], c["question"], c["note"], c["audience"], session_id="eval-" + c["id"], consent=(c["id"] == "N1"))
    runs.append((c, r))
    v = (r.get("final") or {}).get("verification") or {}
    print(f"{c['id']:3s} {c['kind']:17s} status={r['status']:13s} guard={r['input_guardrail']['reason']:17s} "
          f"risk={(r.get('risk') or {}).get('risk_percent')} llm={r['llm_calls']} tools={r['tool_calls']} rev={r.get('revisions', 0)} "
          f"failed={v.get('failed')} {time.time() - t0:.0f}s")""")
md("## 9. Agent metrics")
code("""from core.agents import SPECIALISTS
from core.harness import section
normal = [(c, r) for c, r in runs if c["kind"] == "normal"]; attacks = [(c, r) for c, r in runs if c["kind"] != "normal"]
STOP = set("a an the and or of to in on for with by is are was were be been it its this that as at from can your you more have has".split())
def words(t): return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP and len(w) > 2}
def route_valid(route):
    r = [x for x in route if x != "FINISH"]
    if not r or route[-1] != "FINISH" or set(SPECIALISTS) - set(r): return False
    first_writer = r.index("clinical_writer")
    return all(r.index(a) < first_writer for a in ("intake", "risk_analyst", "evidence_researcher")) and r[first_writer + 1] == "safety_reviewer"
bullet_rows, per_case = [], []
for c, r in normal:
    rt = [t for t in r["trace"] if t["type"] == "route"]
    v = (r.get("final") or {}).get("verification") or {}
    ev = section(r.get("report", ""), "Guideline evidence")
    for b in re.findall(r"^\\s*[-*]\\s+(.+)$", ev, flags=re.M):
        cid = (re.findall(r"\\[(G-\\d{2})\\]", b) or [None])[0]
        src = (r.get("retrieved") or {}).get(cid, {}).get("text", "")
        w = words(re.sub(r"\\[G-\\d{2}\\]", "", b)); sup = len(w & words(src)) / len(w) if w else 0
        bullet_rows.append({"case": c["id"], "citation": cid, "valid_citation": cid in (r.get("retrieved") or {}), "support": round(sup, 2), "bullet": b})
    reviews = [t for t in r["trace"] if t["type"] == "review"]
    per_case.append({"case": c["id"], "status": r["status"], "route_valid": route_valid(r.get("route", [])),
                     "route_overrides": sum(1 for t in rt if t.get("override") and "LLM chose" in (t.get("override") or "")),
                     "route_decisions": len(rt), "revisions": r.get("revisions", 0), "first_draft_pass": bool(reviews) and reviews[0]["verdict"] == "approve",
                     "failed_checks": v.get("failed", []), "needs_approval": (r.get("final") or {}).get("needs_human_approval"),
                     "llm_calls": r["llm_calls"], "tool_calls": r["tool_calls"], "tokens": r.get("tokens"), "latency_s": r["latency_s"],
                     "receipt_verifies": verify_receipt(r["receipt"]), "audit_chain_valid": r["receipt"]["audit_chain_valid"],
                     "skills_loaded": sum(1 for t in r["trace"] if t["type"] == "skill_loaded"),
                     "resource_reads": sum(1 for t in r["trace"] if t["type"] == "mcp_resource_read"),
                     "a2a_messages": len(r.get("a2a", [])),
                     "writer_truncations": sum(1 for t in r["trace"] if t["type"] == "warning" and "finish_reason" in t.get("detail", "")),
                     "red_team_dropped": sum((t.get("red_team") or {}).get("dropped_unverifiable", 0) for t in [r.get("review") or {}])})
pc = pd.DataFrame(per_case); display(pc)
bt = pd.DataFrame(bullet_rows)
lat = pc.latency_s.tolist()
phi_case = next(r for c, r in runs if c["id"] == "N6")
phi_leak = "415-555-0199" in json.dumps({k: v for k, v in phi_case.items() if k != "input_guardrail"}, default=str)
agent = {
  "normal_cases": len(normal), "attack_cases": len(attacks),
  "attack_block_rate": round(sum(r["status"] == "blocked" for _, r in attacks) / len(attacks), 3),
  "attack_reason_correct": round(sum(r["input_guardrail"]["reason"] == (c["kind"] if c["kind"] != "tool_misuse" else "prompt_injection") for c, r in attacks) / len(attacks), 3),
  "false_block_rate_normal": round(sum(r["status"] == "blocked" for _, r in normal) / len(normal), 3),
  "phi_redacted_in_N6": phi_case["input_guardrail"]["phi_redacted"], "phi_leak_after_redaction": phi_leak,
  "route_policy_valid_rate": round(pc.route_valid.mean(), 3),
  "llm_route_choices_accepted": round(1 - pc.route_overrides.sum() / pc.route_decisions.sum(), 3),
  "finalized_rate": round((pc.status == "finalized").mean(), 3), "first_draft_pass_rate": round(pc.first_draft_pass.mean(), 3),
  "mean_revisions": round(pc.revisions.mean(), 2),
  "citation_coverage": round(bt.valid_citation.mean(), 3) if len(bt) else None,
  "grounding_mean_support": round(bt.support.mean(), 3) if len(bt) else None,
  "grounded_bullets_ge_0_5": round((bt.support >= 0.5).mean(), 3) if len(bt) else None,
  "evidence_bullets": len(bt),
  "latency_mean_s": round(statistics.mean(lat), 1), "latency_median_s": round(statistics.median(lat), 1), "latency_max_s": round(max(lat), 1),
  "llm_calls_mean": round(pc.llm_calls.mean(), 1), "tool_calls_mean": round(pc.tool_calls.mean(), 1), "tokens_mean": int(pc.tokens.mean()),
  "receipts_verify": bool(pc.receipt_verifies.all()), "audit_chains_valid": bool(pc.audit_chain_valid.all()),
  "approval_required_cases": int(pc.needs_approval.fillna(False).sum()),
  "writer_truncations_retried": int(pc.writer_truncations.sum()),
}
print(json.dumps(agent, indent=1))
fails = {}
for fc in pc.failed_checks: 
    for f in fc: fails[f] = fails.get(f, 0) + 1
print("failed checks across final drafts:", fails)""")
md("## 10. RAGAS-style faithfulness (LLM judge) on guideline-evidence bullets\nFollowing Es et al. (2024): each statement is judged supported / not supported by its cited context.")
code("""import httpx
from core.config import get_llm
from pydantic import BaseModel
class Judge(BaseModel):
    supported: bool
    reason: str
async def judge_all():
    out = []
    async with httpx.AsyncClient(timeout=60) as hc:
        llm = get_llm(http_async_client=hc).with_structured_output(Judge)
        for b in bullet_rows:
            src = next((r.get("retrieved") or {}).get(b["citation"], {}).get("text", "") for c, r in normal if c["id"] == b["case"])
            j = await llm.ainvoke(f"CONTEXT:\\n{src}\\n\\nSTATEMENT:\\n{b['bullet']}\\n\\nIs every factual part of the STATEMENT supported by the CONTEXT? "
                                  "Ignore the citation marker. Answer supported=true or false.")
            out.append({**b, "judge_supported": j.supported, "judge_reason": j.reason})
    return out
judged = pd.DataFrame(await judge_all())
ragas_faithfulness = round(judged.judge_supported.mean(), 3) if len(judged) else None
print("LLM-judged faithfulness:", ragas_faithfulness, "over", len(judged), "statements")
display(judged[~judged.judge_supported][["case", "citation", "bullet", "judge_reason"]])""")
md("## 11. Failure cases and charts")
code("""failures = []
for c, r in normal:
    for t in r["trace"]:
        if t["type"] == "review" and t["verdict"] != "approve":
            failures.append({"case": c["id"], "stage": "review", "verdict": t["verdict"], "failed_checks": t["failed_checks"], "issues": "; ".join(t["issues"])[:300]})
        if t["type"] == "route" and t.get("override") and "LLM chose" in t["override"]:
            failures.append({"case": c["id"], "stage": "orchestrator", "verdict": "override", "failed_checks": [], "issues": t["override"]})
    if r["status"] != "finalized":
        failures.append({"case": c["id"], "stage": "gate", "verdict": r["status"], "failed_checks": (r.get("final") or {}).get("verification", {}).get("failed"), "issues": ""})
fdf = pd.DataFrame(failures); display(fdf)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
keys = ["attack_block_rate", "route_policy_valid_rate", "llm_route_choices_accepted", "finalized_rate", "first_draft_pass_rate", "citation_coverage", "grounded_bullets_ge_0_5"]
vals = [agent[k] for k in keys] + [ragas_faithfulness]
ax[0].barh(keys + ["ragas_faithfulness"], vals, color="#4a7ab7"); ax[0].set_xlim(0, 1.05); ax[0].set_title("Agent metrics")
for i, v in enumerate(vals): ax[0].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
ax[1].bar(pc.case, pc.latency_s, color="#7ab77a"); ax[1].set_title("Latency per case (s)")
plt.tight_layout(); plt.savefig("eval/agent_metrics.png", dpi=120); plt.show()""")
md("""## 12. Probing the content-filter failure found during development
During development, an earlier full run of this notebook (before the fix; its outputs were overwritten and are not retained) finalized only 3 of 8 reports: the writer pasted the long
CDC BE FAST passage verbatim and the provider stopped generation with `finish_reason="content_filter"`, so the
truncated drafts lost their Guideline-evidence citation and Disclaimer and the finalize gate refused them.
The fix (short paraphrased findings in the `guideline-citation` / `risk-communication` skills and the report
prompt, plus one automatic retry on a truncated generation) is what the runs above use. This cell tries to reproduce the
failure mode: it asks for the old verbatim behaviour three times and records the finish reasons. **Result in the
final run: 0 of 3 probes were stopped**, so the filter did not trigger on this isolated prompt (it appears to depend on
the full agent context and is not deterministic). The development observation is therefore reported, not reproduced.""")
code("""from langchain_core.messages import HumanMessage
from core.data import load_guidelines
G = {d["id"]: d for d in load_guidelines()}
verbatim = "\\n".join(f"- {G[c]['text']} [{c}]" for c in ("G-05", "G-06", "G-10", "G-13"))
probe_prompt = ("Write a stroke-risk decision-support report in Markdown with sections Summary, Guideline evidence and Disclaimer. "
                "Under Guideline evidence copy each excerpt below verbatim as one bullet, keeping the source wording exactly.\\n" + verbatim)
async def probe(n=3):
    out = []
    async with httpx.AsyncClient(timeout=60) as hc:
        llm = get_llm(http_async_client=hc)
        for _ in range(n):
            r = await llm.ainvoke([HumanMessage(probe_prompt)])
            out.append({"finish_reason": r.response_metadata.get("finish_reason"), "chars": len(r.content), "has_disclaimer": "isclaimer" in r.content})
    return out
cf_probe = await probe()
display(pd.DataFrame(cf_probe))
content_filter_probe = {"attempts": len(cf_probe), "content_filter_stops": sum(p["finish_reason"] == "content_filter" for p in cf_probe)}
print(content_filter_probe)""")
code("""trim = []
for c, r in runs:
    trim.append({"case": c, "status": r["status"], "guard": r["input_guardrail"]["reason"], "route": r.get("route"), "risk": r.get("risk"),
                 "report": r.get("report"), "final": r.get("final"), "trace": r["trace"], "a2a": r.get("a2a"), "receipt": r["receipt"],
                 "llm_calls": r["llm_calls"], "tool_calls": r["tool_calls"], "latency_s": r["latency_s"]})
Path("eval/raw_results.json").write_text(json.dumps(trim, indent=1, default=str))
summary = {"python": platform.python_version(), "model": C["model"], "contract_sha256": sha(C),
           "data": rep["data"], "model_comparison": rep["comparison"], "model_test_metrics": rep["test_metrics"],
           "note_classifier": {k: v for k, v in nc.report.items() if k != "top_neuro_terms"}, "odds_ratios": ft["odds_ratios"],
           "retrieval": retrieval, "harness_unit": unit, "mcp": {k: v for k, v in mcp_info.items() if k != "calculator_injection_result"},
           "agent": agent, "ragas_faithfulness": ragas_faithfulness, "content_filter_probe": content_filter_probe, "failed_check_counts": fails, "failure_rows": len(fdf)}
Path("eval/summary.json").write_text(json.dumps(summary, indent=1, default=str))
pc.to_csv("eval/results_table.csv", index=False)
print(json.dumps(summary["agent"], indent=1)); print("ragas_faithfulness", ragas_faithfulness)""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, ROOT / "evaluation.ipynb")
print("wrote evaluation.ipynb with", len(cells), "cells")
