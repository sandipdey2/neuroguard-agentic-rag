"""Draw architecture_diagram.png / .pdf (matplotlib).   python scripts/draw_architecture.py
Every box names the file that implements it; every arrow is a real call path. Counts and names are read from
harness_contract.json, skills/*/SKILL.md and the MCP server source, so the figure cannot drift from the code."""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPO = "https://github.com/sandipdey2/neuroguard-agentic-rag"
LIVE = "https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/"

K = json.loads((ROOT / "harness_contract.json").read_text())
SRV = (ROOT / "mcp_server" / "neuroguard_mcp_server.py").read_text()
TOOLS = re.findall(r"@mcp\.tool\(\)\s*\ndef (\w+)", SRV)
RES = re.findall(r'@mcp\.resource\("([^"]+)"', SRV)
STATIC = [r for r in RES if "{" not in r]
TEMPL = [r for r in RES if "{" in r]
PROMPTS = re.findall(r"@mcp\.prompt\(\)\s*\ndef (\w+)", SRV)
from core.skills import catalog  # noqa: E402
SK = catalog()
from core import graph_rag  # noqa: E402
GS = graph_rag.stats()
assert len(TOOLS) == 11 and len(STATIC) == 5 and len(TEMPL) == 1 and len(PROMPTS) == 2 and len(SK) == 6, (TOOLS, RES, PROMPTS, SK)
assert {"graph_traverse", "episode_recall", "episode_write"} <= set(TOOLS) and "graph://knowledge-graph" in STATIC
assert "planner" in K["agents"] and K["loop"]["planner"] and "confidence" in K and "batch" in K
L, B, A, CF = K["loop"], K["budget"], K["agents"], K["confidence"]


def skills_of(agent):
    return [n for n, m in SK.items() if agent in m["agents"]]


fig, ax = plt.subplots(figsize=(19, 15))
ax.set_xlim(-3, 191); ax.set_ylim(-6, 146); ax.axis("off")
C = {"ui": "#e8f0fe", "guard": "#fde7e7", "llm": "#fff4d6", "rule": "#e6f4ea", "mcp": "#efe7fb", "data": "#f1f3f4", "state": "#ffffff",
     "obs": "#e0f7fa"}


def box(x, y, w, h, title, body="", color="#fff", fs=9.2, ls="-", bfs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#333", lw=1.2, ls=ls, zorder=1))
    ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=fs, weight="bold")
    if body:
        n_title = title.count("\n") + 1
        ax.text(x + w / 2, y + h - 1.4 - 2.5 * n_title, body, ha="center", va="top", fontsize=bfs, linespacing=1.2)


def arrow(x1, y1, x2, y2, label="", both=False, color="#333", ls="-", rad=0.0, lx=0, ly=0, fs=6.8, z=0.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="<|-|>" if both else "-|>", mutation_scale=12,
                                 color=color, lw=1.2, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=fs, ha="center", va="center", color=color,
                bbox=dict(fc="white", ec="none", pad=0.3))


ax.text(94, 145, "NeuroGuard AI - architecture as implemented (app.py, core/*.py, mcp_server/neuroguard_mcp_server.py)",
        ha="center", va="top", fontsize=13.5, weight="bold")
ax.text(94, 141.8, f"All LLM roles: {K['model']}, temperature {K['temperature']}, OpenAI-compatible Vocareum endpoint   |   contract v{K['contract_version']}   |   "
        "colour: blue UI, red guard, yellow LLM agent, green deterministic code, purple MCP, cyan observability, grey data", ha="center", va="top", fontsize=8.4, color="#444")

# ---------------- row 1: UI, guardrails, contract, planner, orchestrator, state
y1, h1 = 116, 21
box(0, y1, 27, h1, "Streamlit UI (app.py)", "profile form, note, audience, consent\nlab report upload (image / PDF)\nlive progress | 12 result tabs\nKnowledge-graph coverage tab\nCohort batch tab | approve/reject\n.md/.pdf, receipt, OTLP export", C["ui"], bfs=7.0)
box(30, y1, 28, h1, "Input guardrails\n(core/guardrails.py)", "PHI redaction -> prompt injection\n-> BE FAST emergency (9-1-1)\n-> dosing -> out of scope\n+ profile validation | no LLM\nmulti-modal intake (partial):\ncore/multimodal.py vision / pypdf", C["guard"], bfs=7.0)
box(61, y1, 28, h1, "Run contract\n(harness_contract.json)", f"SHA-256 hashed before first token\n{B['max_llm_calls']} LLM | {B['max_tool_calls']} tools | {B['max_total_tokens'] // 1000}k tok | {B['wall_clock_timeout_s']} s\n"
    f"per-agent permissions, {len(K['required_checks'])} checks\ncontext, confidence, pricing,\nlearning, batch policies", C["rule"], bfs=7.0)
box(92, y1, 28, h1, "Planner (LLM)", f"MCP episode_recall: similar past\nepisodes + learned hints\n-> structured Plan (ordered steps,\nwatch-outs, rationale)\nplanner -> executor pattern", C["llm"], bfs=7.0)
box(123, y1, 34, h1, "Orchestrator (LLM, core/agents.py)", "executes the plan: next agent, skill,\ninstruction, reason (structured)\ncode overrides: eligibility, FINISH\nonly after review | "
    f"{L['max_orchestrator_turns']} turns,\n{L['max_visits_per_agent']} visits/agent, {L['max_revisions']} revision | every route\n-> decision reasoning log", C["llm"], bfs=7.0)
box(160, y1, 29, h1, "CaseState (LangGraph)", "short-term memory per run:\nprofile, plan, learning, intake,\nanalyst, risk, evidence, coverage,\nretrieved, draft, self_critique,\nreview, route, trace[]", C["state"], ls="--", bfs=7.0)
for xa, xb in ((27, 30), (58, 61), (89, 92), (120, 123)):
    arrow(xa, 126, xb, 126)
arrow(157, 126, 160, 126, ls="--")
ax.text(44, 113.6, "blocked -> refusal / emergency message\n(+ signed receipt, 0 LLM calls)", fontsize=6.8, ha="center", va="top", color="#b00")

# ---------------- A2A bus
box(36, 102, 121, 7, "A2A-style message bus (core/a2a.py): Agent Cards, Task -> Message -> Artifact, inproc:// endpoints (no network)", "", C["data"], fs=8.6)
arrow(140, 116, 140, 109, "Task + skill", both=True, lx=7)

# ---------------- specialists
specs = [("Intake", f"skill: {skills_of('intake')[0]}\nMCP: screen_note, memory_read\n(BE FAST signs, prior cases)\nReAct loop", C["llm"]),
         ("Risk Analyst", f"skill: {skills_of('risk_analyst')[0]}\nMCP: predict_risk, explain_factors,\ncohort_stats, calculator\n+ model://risk-card | ReAct", C["llm"]),
         ("Evidence Researcher (agentic RAG)", f"skill: {skills_of('evidence_researcher')[0]}\nReAct: search_guidelines (hybrid)\n+ graph_traverse (graph RAG)\ncode coverage check -> missing\n"
          f"topics -> reformulate, re-retrieve\n(max {L['max_retrieval_rounds']} rounds) | guideline://{{id}}", C["llm"]),
         ("Clinical Writer (reflection)", "skills: risk-communication |\npatient-friendly-summary\nprompt clinical_report; evidence\n"
          f"packed to {K['context']['writer_evidence_tokens']} tokens; learned hints\nself-critique: checks + quote-\nverified LLM critique, {L['max_self_revisions']} revision", C["llm"]),
         ("Safety Reviewer", f"skill: {skills_of('safety_reviewer')[0]}\n{len(K['required_checks'])}-check invariant verifier\n(code) + LLM red team (prompt\nred_team_review), quote-\nconfirmed | revise x{L['max_revisions']}", C["rule"])]
for i, (t, b, col) in enumerate(specs):
    x = 2 + i * 37.5
    box(x, 76, 34, 22, t, b, col, fs=8.8, bfs=7.0)
    arrow(x + 17, 98, x + 17, 102, both=True)
ax.text(94, 73.8, f"Tool-using agents run a ReAct loop: Thought -> LLM tool_calls (bind_tools) -> pre-tool-use hook (allow/deny, clamp, pin) -> MCP Observation "
        f"(PHI-redacted, truncated to {K['context']['max_tool_output_tokens']} tokens) -> context (compacted at {K['context']['agent_context_tokens']}), max {L['max_tool_rounds_per_agent']} rounds",
        ha="center", fontsize=7.3, style="italic", bbox=dict(fc="white", ec="none", pad=0.2))

# ---------------- hook / MCP / gate
box(0, 50, 38, 19, "Permission engine + pre-tool-use hook", "least privilege per agent (bind_tools +\nre-check); arg clamps (k, limit, hops,\nexpr length); tool budget; session_id\nand learned boost_factors pinned by\n"
    "harness; memory_write needs consent;\nhash-chained audit log (core/permissions.py)", C["guard"], bfs=7.0)
box(43, 46, 92, 23, "MCP server: official mcp SDK (FastMCP), stdio subprocess",
    f"{len(TOOLS)} tools: " + ", ".join(TOOLS[:5]) + ",\n" + ", ".join(TOOLS[5:]) + "\n"
    f"{len(STATIC)} resources: " + ", ".join(STATIC) + f"\n+ {len(TEMPL)} resource template: {TEMPL[0]}   |   {len(PROMPTS)} prompts: " + ", ".join(PROMPTS) +
    "\nclient: core/mcp_client.py + langchain-mcp-adapters (sys.executable)\nOPENAI / NG_RECEIPT / GH keys stripped from the server's environment", C["mcp"], fs=9.0, bfs=7.2)
arrow(38, 59, 43, 59, both=True)
ax.text(40.5, 63.5, "stdio\nJSON-RPC", fontsize=6.5, ha="center", va="center")
for x in (19, 56.5, 94):
    arrow(x, 76, min(x, 30), 69, color="#a33")
for x in (131.5, 169):
    arrow(x, 76, 128 if x < 150 else 134, 69, color="#6a3fa0", ls="--")
arrow(106, 116, 100, 69, color="#a33", ls=":", label="episode_recall", lx=9, ly=20)
box(139, 46, 50, 23, "Finalize gate + confidence policy gate", f"re-runs the {len(K['required_checks'])} checks on the final draft\nconfidence = weighted calibration, cohort n,\n"
    "topic coverage, verifier, red team, grounding\n"
    f"< {CF['refuse_below']} refuse  |  < {CF['review_below']} clinician review\nclinician sign-off if high risk / red flags\nmemory_write (consent) + episode_write (harness)", C["rule"], bfs=7.1)
ax.plot([157.4, 189.6], [105.5, 105.5], color="#333", lw=1.2)
arrow(189.6, 105.5, 189.6, 69.4)
ax.text(183, 101, "orchestrator\nFINISH", fontsize=6.8, ha="center")
arrow(139, 52, 135.4, 52, color="#a33")
ax.text(137.2, 54.3, "write", fontsize=6.4, ha="center", color="#a33")

# ---------------- governance / observability row
yg, hg = 27, 15
box(0, yg, 30, hg, "Skills (skills/, core/skills.py)", f"{len(SK)} SKILL.md; catalog = name +\ndescription only; full body\ninjected when the orchestrator\nselects it (progressive disclosure)", C["state"], fs=8.6, bfs=6.9)
box(33, yg, 30, hg, "Context windows\n(core/context.py)", f"tiktoken o200k_base counts\ntruncate / compact / pack evidence\nwindow {K['context']['model_context_window'] // 1000}k, budgets per agent", C["obs"], fs=8.6, bfs=6.9)
box(66, yg, 38, hg, "Decision log + observability\n(core/harness.py, observability.py)", "RunContext.decide(): actor, options,\nchosen, rationale, evidence | one span\nper LLM / MCP call -> OTLP-JSON export", C["obs"], fs=8.6, bfs=6.9)
box(107, yg, 32, hg, "Cost / token economics", f"usage per call x list price\nUSD {K['pricing']['input_per_1m']}/1M in, USD {K['pricing']['output_per_1m']}/1M out\nper-agent cost; budgets cap spend", C["obs"], fs=8.6, bfs=6.9)
box(142, yg, 47, hg, "Signed receipt (core/pipeline.py)", "HMAC-SHA256 over contract, server, inputs,\nreport, trace, decision-log hashes + audit head,\ncost, confidence | clinician decision re-signs it\nand is written back to episodic memory", C["rule"], fs=8.6, bfs=6.9)
arrow(164, 46, 164, 42.4)
ax.annotate("", xy=(-1.4, 87), xytext=(-1.4, 42.4), arrowprops=dict(arrowstyle="-|>", color="#666", ls="--", lw=1.1))
ax.text(-2.6, 60, "load_body()", rotation=90, fontsize=6.6, va="center", color="#555")

# ---------------- data / retrieval / memory row
yd, hd = 2, 21
box(0, yd, 27, hd, "P1 data + P2 statistics", "Stroke Prediction Dataset\n(Kaggle, 5,109 rows)\ncore/data.py cleaning log\ncore/stats.py: chi-square,\nWelch t, odds ratios, Wilson CIs", C["data"], fs=8.4, bfs=6.9)
box(30, yd, 27, hd, "P3 risk model\n(core/model.py)", "LR / RF / GBM, 5-fold CV\nserved: calibrated LR\nexact log-odds explanations\nthreshold: OOF recall >= 0.80\ncalibration curve -> confidence", C["data"], fs=8.4, bfs=6.9)
box(60, yd, 25, hd, "P4 notes\n(core/notes.py)", "CDC BE FAST rules\nTF-IDF + LR on\nMTSamples (CC0)\nadvisory neuro-note\nprobability", C["data"], fs=8.4, bfs=6.9)
box(88, yd, 31, hd, "Hybrid RAG + vectors\n(core/rag.py, vectors.py)", "30 CDC / NHLBI excerpts\nsparse TF-IDF 1-2 gram (0.6)\n+ dense 24-d LSA vectors (0.4)\n+ topic boost + learned boost\n(cap 0.06); modes sparse/dense/hybrid", C["data"], fs=8.4, bfs=6.9)
box(122, yd, 31, hd, "Graph RAG\n(core/graph_rag.py)", f"networkx: {GS['nodes']} nodes ({GS['node_kinds']['factor']} factor,\n{GS['node_kinds']['condition']} condition, {GS['node_kinds']['chunk']} chunk), {GS['edges']} edges\n"
    "weighted multi-hop traversal\nfactor -> chunk, factor ->\ncondition <- chunk, with paths", C["data"], fs=8.4, bfs=6.9)
box(156, yd, 33, hd, "Long-term + episodic memory", "SQLite: core/memory.py case\nsummaries (consent) | core/\nepisodes.py one de-identified\nepisode per run -> learning loop:\nwhitelisted hints + bounded boosts", C["data"], fs=8.4, bfs=6.9)
for x0, x1 in ((13, 50), (43, 62), (72, 80), (103, 96), (137, 112), (172, 128)):
    arrow(x0, 23, x1, 46, color="#666")

ax.text(94, -2.0, "Scale (partial): core/batch.py runs a cohort with asyncio.Semaphore("
        f"{K['batch']['max_concurrency']}) + one shared token bucket ({K['batch']['llm_calls_per_minute']} LLM calls/min) awaited before every LLM call.  "
        "Honest scope: A2A is in-process; public educational data; not a medical device.", ha="center", fontsize=7.6, color="#333")
ax.text(94, -4.8, f"Repo: {REPO}    |    Live demo: {LIVE}", ha="center", fontsize=8.2, color="#1a56c4")
fig.savefig(ROOT / "architecture_diagram.png", dpi=150, bbox_inches="tight")
fig.savefig(ROOT / "architecture_diagram.pdf", bbox_inches="tight")
print("wrote architecture_diagram.png/.pdf", TOOLS, RES, PROMPTS, list(SK))
