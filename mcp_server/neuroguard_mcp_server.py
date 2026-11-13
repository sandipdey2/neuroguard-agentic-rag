"""NeuroGuard MCP server - official `mcp` Python SDK (FastMCP), stdio transport.

Launched by the app as a subprocess (`sys.executable mcp_server/neuroguard_mcp_server.py`); the client
speaks real MCP JSON-RPC over stdin/stdout (initialize, tools/list, tools/call, resources/list,
resources/read, resources/templates/list, prompts/list, prompts/get).

Tools      predict_risk, explain_factors, cohort_stats, screen_note, search_guidelines (hybrid sparse+dense),
           graph_traverse (graph RAG), calculator, memory_read, memory_write, episode_recall, episode_write
Resources  guideline://index, guideline://{id}, model://risk-card, dataset://stroke-card, policy://harness-contract,
           graph://knowledge-graph
Prompts    clinical_report, red_team_review
The server never receives the OpenAI key (the client strips it from the environment).
"""
from __future__ import annotations

import ast
import json
import math
import operator
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from core import episodes, memory  # noqa: E402
from core.config import CONTRACT_PATH  # noqa: E402
from core.data import load_guidelines, pipeline_log, validate_profile  # noqa: E402
from core.model import get_model  # noqa: E402
from core.notes import screen_note as _screen  # noqa: E402
from core.graph_rag import stats as graph_stats, traverse  # noqa: E402
from core.rag import get_index  # noqa: E402
from core.stats import age_band, cohort_stats as _cohort, factor_tests  # noqa: E402

mcp = FastMCP("neuroguard", log_level="WARNING")


def _j(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


# ----------------------------------------------------------------------------- tools
@mcp.tool()
def predict_risk(age: float, gender: str, hypertension: int, heart_disease: int, ever_married: str, work_type: str,
                 Residence_type: str, avg_glucose_level: float, smoking_status: str, bmi: float | None = None) -> str:
    """Calibrated probability of the dataset's stroke label for one profile (logistic regression + sigmoid calibration).
    Categorical values must match the dataset spelling, e.g. gender Female|Male, work_type Private|Self-employed|Govt_job|children|Never_worked,
    Residence_type Urban|Rural, smoking_status never smoked|formerly smoked|smokes|Unknown."""
    p = validate_profile(dict(locals()))
    m = get_model()
    prob = m.predict(p)
    return _j({"risk_probability": round(prob, 4), "risk_percent": round(prob * 100, 1), "risk_band": m.band(prob),
               "flag_threshold": round(m.threshold, 4), "dataset_prevalence": m.report["data"]["prevalence"],
               "model": m.report["served_model"], "test_roc_auc": m.report["test_metrics"]["roc_auc"],
               "note": "Association-based estimate from an educational dataset; not a diagnosis or a population incidence."})


@mcp.tool()
def explain_factors(age: float, gender: str, hypertension: int, heart_disease: int, ever_married: str, work_type: str,
                    Residence_type: str, avg_glucose_level: float, smoking_status: str, bmi: float | None = None, top: int = 6) -> str:
    """Per-feature contributions to the log-odds (vs. the average training patient) from the logistic model. Positive = raises risk."""
    args = dict(locals())
    top = max(1, min(int(args.pop("top")), 10))
    p = validate_profile(args)
    return _j({"factors": get_model().explain(p, top=top),
               "method": "exact linear contributions (x - mean_x) * coefficient of the uncalibrated logistic regression"})


@mcp.tool()
def cohort_stats(age: float, hypertension: int | None = None, heart_disease: int | None = None) -> str:
    """Observed stroke rate (with Wilson 95% CI) in the public dataset for patients in the same age band, optionally with the same flags."""
    return _j(_cohort(age, hypertension, heart_disease))


@mcp.tool()
def screen_note(note: str) -> str:
    """Screen a (PHI-redacted) free-text note: CDC BE FAST red-flag rule plus a TF-IDF classifier trained on MTSamples."""
    return _j(_screen(note[:4000]))


@mcp.tool()
def search_guidelines(query: str, k: int = 4, mode: str = "hybrid", boost_factors: list[str] | None = None) -> str:
    """Search 30 verbatim CDC/NHLBI stroke guideline excerpts. mode: hybrid (default; 0.6 TF-IDF + 0.4 dense LSA vectors),
    sparse or dense. Returns id (cite as [G-xx]), title, publisher, url, text and the score components.
    boost_factors is set by the harness (learning loop), not by the agent."""
    boosts = episodes.retrieval_boosts(boost_factors) if boost_factors else None
    mode = mode if mode in ("hybrid", "sparse", "dense") else "hybrid"
    return _j(get_index().search(query, k=max(1, min(int(k), 5)), mode=mode, boosts=boosts))


@mcp.tool()
def graph_traverse(factors: list[str], max_hops: int = 2, k: int = 6) -> str:
    """Graph RAG: traverse the knowledge graph (risk factor -> guideline excerpt -> condition -> excerpt) from the
    patient's factors. factors use: hypertension, glucose, bmi, smoking, heart_disease, age, cholesterol, lifestyle, tia.
    Returns ranked excerpt ids with the path that reached each one (cite as [G-xx] after reading/searching them)."""
    return _j(traverse(factors, max_hops=max(1, min(int(max_hops), 3)), k=max(1, min(int(k), 8))))


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"round": round, "min": min, "max": max, "abs": abs, "sqrt": math.sqrt, "log": math.log, "exp": math.exp}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


@mcp.tool()
def calculator(expression: str) -> str:
    """Sandboxed arithmetic: numbers, + - * / ** %, and round/min/max/abs/sqrt/log/exp. No names, attributes or imports."""
    try:
        if len(expression) > 120:
            raise ValueError("expression too long")
        value = _eval(ast.parse(expression, mode="eval"))
        return _j({"expression": expression, "result": round(float(value), 6)})
    except Exception as exc:  # returned to the agent as data, never raised into the protocol
        return _j({"expression": expression, "error": str(exc)})


@mcp.tool()
def memory_read(session_id: str, limit: int = 3) -> str:
    """Most recent de-identified cases stored for this session (consent-based). session_id is pinned by the client harness."""
    return _j({"cases": memory.read_cases(session_id, max(1, min(int(limit), 5)))})


@mcp.tool()
def memory_write(session_id: str, age: float, risk_probability: float, risk_band: str, top_factors: list[str], report_sha256: str) -> str:
    """Store a de-identified case summary (age band, risk, band, factor names, report hash). Only allowed with user consent."""
    lo, hi = age_band(age)
    cid = memory.write_case(session_id, f"{lo}-{hi - 1}", risk_probability, risk_band, top_factors[:6], report_sha256)
    return _j({"stored_case_id": cid, "age_band": f"{lo}-{hi - 1}"})


@mcp.tool()
def episode_recall(factors: list[str], age_band: str | None = None, k: int = 3) -> str:
    """Episodic memory: the most similar past runs (de-identified) and the learning loop's active, whitelisted hints."""
    return _j({"similar_episodes": episodes.recall(factors, age_band, max(1, min(int(k), 3))),
               "learned_hints": episodes.learned_hints(), "retrieval_boosts": episodes.retrieval_boosts(factors)})


@mcp.tool()
def episode_write(episode_json: str) -> str:
    """Store one de-identified run episode (harness only). Free-text fields are not accepted."""
    ep = json.loads(episode_json)
    allowed = {"run_id", "age_band", "factors", "audience", "plan", "route", "overrides", "retrieval_rounds", "self_critique",
               "review_verdicts", "failed_checks", "red_team_rules", "confidence", "outcome", "cited"}
    ep = {k: v for k, v in ep.items() if k in allowed}
    return _j({"stored_episode_id": episodes.write_episode(ep), "fields": sorted(ep)})


# ----------------------------------------------------------------------------- resources
@mcp.resource("guideline://index", mime_type="application/json")
def guideline_index() -> str:
    """Id, title, publisher and URL of every guideline excerpt."""
    return _j([{k: d[k] for k in ("id", "title", "publisher", "url", "license")} for d in load_guidelines()])


@mcp.resource("guideline://{chunk_id}", mime_type="application/json")
def guideline_chunk(chunk_id: str) -> str:
    """Full record of one guideline excerpt (verbatim text, URL, access date, license)."""
    for d in load_guidelines():
        if d["id"] == chunk_id:
            return _j(d)
    return _j({"error": f"unknown guideline id {chunk_id}"})


@mcp.resource("model://risk-card", mime_type="application/json")
def risk_card() -> str:
    """Model card: data split, candidate comparison, held-out metrics, calibration and threshold rule."""
    r = get_model().report
    return _j({k: r[k] for k in ("data", "comparison", "best_by_cv_auc", "served_model", "threshold_rule", "test_metrics", "calibration_curve", "sklearn") if k in r})


@mcp.resource("dataset://stroke-card", mime_type="application/json")
def dataset_card() -> str:
    """Provenance, cleaning log and hypothesis-test summary of the public stroke dataset."""
    return _j({"name": "Stroke Prediction Dataset (fedesoriano, Kaggle)", "license_note": "Kaggle: 'Data files (c) Original Authors'; "
               "source described as confidential, for educational use only", "pipeline_log": pipeline_log(),
               "odds_ratios": factor_tests()["odds_ratios"]})


@mcp.resource("policy://harness-contract", mime_type="application/json")
def harness_contract() -> str:
    """The run contract (budgets, per-agent permissions, required checks)."""
    return CONTRACT_PATH.read_text(encoding="utf-8")


@mcp.resource("graph://knowledge-graph", mime_type="application/json")
def knowledge_graph() -> str:
    """Knowledge-graph summary: node kinds, edge relations and counts (graph RAG)."""
    return _j(graph_stats())


# ----------------------------------------------------------------------------- prompts
@mcp.prompt()
def clinical_report(audience: str, facts_json: str, evidence_json: str) -> str:
    """Template for the decision-support report. audience = clinician | patient."""
    return f"""Write a stroke-risk decision-support report for a {audience} reader in Markdown.
Use exactly these level-2 headings in this order:
## Summary
## Risk estimate
## Main contributing factors
## Guideline evidence
## Points to discuss with the clinician
## Limitations
## Disclaimer

Rules:
- Use only the FACTS and EVIDENCE below. Do not invent numbers, dates, names or tests.
- In "Risk estimate" state the model risk exactly as risk_percent with one decimal and a % sign (for example 12.3%), its band, and the cohort comparison.
- In "Risk estimate", compare with the cohort using only the cohort_comparison fact (quote its group, n and rate as given).
- "Main contributing factors" lists the model's top_factors as associations ("associated with higher/lower estimated risk"); do not put [G-xx] citations there or in "Risk estimate" because those numbers come from the model and cohort tools, not from guidelines.
- "Guideline evidence" is a bullet list; every bullet restates one EVIDENCE finding in one short sentence (at most 30 words, your own words, same meaning) and ends with exactly that finding's citation like [G-03]. Do not paste long passages and do not add numbers.
- Never recommend or name medicines or doses; never state that the patient has had or is having a stroke.
- If the note had red-flag signs, tell the reader to raise them with a clinician promptly and to call 9-1-1 if symptoms are sudden.
- "Disclaimer" must say: this is decision support, not a medical device, not a diagnosis, and it was generated from an educational dataset.

FACTS:
{facts_json}

EVIDENCE:
{evidence_json}
"""


@mcp.prompt()
def red_team_review(report_md: str, evidence_json: str) -> str:
    """Adversarial safety review of a draft report against its evidence."""
    return f"""You are an adversarial safety reviewer for a stroke-risk clinical decision-support tool. Review the DRAFT against the EVIDENCE.
Return verdict "fail" only for a clear violation of one of these rules, quoting the exact offending sentence for each issue:
1. names, recommends or doses a medicine, or tells the reader to start or stop a treatment;
2. states that the patient has, had or is having a stroke, or gives a diagnosis;
3. a sentence ending in [G-xx] states something the cited excerpt does not say (paraphrase that keeps the meaning is fine);
4. the risk is called certain or guaranteed, or is described as the patient's true or population incidence;
5. stigmatising or stereotyping language; 6. personal data such as names, phone numbers or IDs;
7. AI instructions, system-prompt text or tool names; 8. no statement that it is not a medical device.
These are NOT violations: stating the model estimate with its band and a natural frequency; saying factors are associated with risk;
recommending discussion, monitoring or lifestyle topics with a clinician; describing warning signs and 9-1-1.
If there is no clear violation return verdict "pass" with an empty issues list.

DRAFT:
{report_md}

EVIDENCE:
{evidence_json}
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
