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
assert len(TOOLS) == 8 and len(STATIC) == 4 and len(TEMPL) == 1 and len(PROMPTS) == 2 and len(SK) == 6, (TOOLS, RES, PROMPTS, SK)
L, B, A = K["loop"], K["budget"], K["agents"]


def skills_of(agent):
    return [n for n, m in SK.items() if agent in m["agents"]]


fig, ax = plt.subplots(figsize=(17, 12.6))
ax.set_xlim(-3, 170); ax.set_ylim(0, 126); ax.axis("off")
C = {"ui": "#e8f0fe", "guard": "#fde7e7", "llm": "#fff4d6", "rule": "#e6f4ea", "mcp": "#efe7fb", "data": "#f1f3f4", "state": "#ffffff"}


def box(x, y, w, h, title, body="", color="#fff", fs=9.4, ls="-", bfs=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#333", lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=fs, weight="bold")
    if body:
        n_title = title.count("\n") + 1
        ax.text(x + w / 2, y + h - 1.6 - 2.6 * n_title, body, ha="center", va="top", fontsize=bfs, linespacing=1.22)


def arrow(x1, y1, x2, y2, label="", both=False, color="#333", ls="-", rad=0.0, lx=0, ly=0, fs=7.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="<|-|>" if both else "-|>", mutation_scale=12,
                                 color=color, lw=1.25, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=fs, ha="center", va="center", color=color,
                bbox=dict(fc="white", ec="none", pad=0.4))


ax.text(85, 124.5, "NeuroGuard AI - architecture as implemented (app.py, core/*.py, mcp_server/neuroguard_mcp_server.py)",
        ha="center", va="top", fontsize=13, weight="bold")
ax.text(85, 121.3, f"All LLM roles: {K['model']}, temperature {K['temperature']}, OpenAI-compatible Vocareum endpoint (OPENAI_API_BASE)   |   "
        f"contract v{K['contract_version']}", ha="center", va="top", fontsize=8.6, color="#444")

# ---------------- row 1
y1, h1 = 98, 19
box(2, y1, 27, h1, "Streamlit UI (app.py)", "profile form, note, audience,\nmemory consent | live trace,\nA2A log, MCP calls, checks\nclinician approve / reject\nreport .md/.pdf + receipt.json", C["ui"])
box(33, y1, 31, h1, "Input guardrails\n(core/guardrails.py)", "PHI redaction -> prompt injection /\ntool misuse -> BE FAST emergency\n(call 9-1-1) -> dosing -> out of scope\n+ profile validation | no LLM call", C["guard"])
box(68, y1, 31, h1, "Run contract\n(harness_contract.json)", f"SHA-256 hashed before first token\n{B['max_llm_calls']} LLM | {B['max_tool_calls']} tool calls | "
    f"{B['max_total_tokens'] // 1000}k tokens\n{B['wall_clock_timeout_s']} s | per-agent permissions\n{len(K['required_checks'])} required checks", C["rule"])
box(103, y1, 37, h1, "Orchestrator (LLM, core/agents.py)", "structured output: next agent, skill,\ninstruction, reason | code overrides:\n"
    "eligibility, FINISH only after review\n"
    f"caps: {L['max_orchestrator_turns']} turns, {L['max_visits_per_agent']} visits/agent,\n{L['max_revisions']} revision, budget; no tools", C["llm"])
box(144, y1, 24, h1, "CaseState\n(LangGraph)", "short-term memory per run:\nprofile, intake, analyst,\nrisk, evidence, retrieved,\ndraft, review, turns,\nvisits, trace[]", C["state"], ls="--")
arrow(29, 107, 33, 107)
arrow(64, 107, 68, 107)
arrow(99, 107, 103, 107)
arrow(140, 107, 144, 107, ls="--")
ax.text(48.5, 95.6, "blocked -> refusal / emergency message\n(+ signed receipt, 0 LLM calls)", fontsize=7.0, ha="center", va="top", color="#b00")

# ---------------- A2A bus
box(36, 83, 104, 7.5, "A2A-style message bus (core/a2a.py): Agent Cards, Task -> Message -> Artifact, inproc:// endpoints (no network)", "", C["data"], fs=8.8)
arrow(121, 98, 121, 90.5, "Task + skill", both=True, lx=7)

# ---------------- specialists
def tools_txt(a):
    t = A[a]["tools"]
    return ", ".join(t)


specs = [("Intake", f"skill: {skills_of('intake')[0]}\nMCP tools: screen_note,\nmemory_read\n(BE FAST signs, prior cases)", C["llm"]),
         ("Risk Analyst", f"skill: {skills_of('risk_analyst')[0]}\nMCP tools: predict_risk,\nexplain_factors, cohort_stats,\ncalculator | resource\nmodel://risk-card", C["llm"]),
         ("Evidence Researcher", f"skill: {skills_of('evidence_researcher')[0]}\nMCP tool: search_guidelines\n+ resource guideline://{{id}}\nreads of cited excerpts", C["llm"]),
         ("Clinical Writer", "skills: risk-communication |\npatient-friendly-summary\nMCP prompt: clinical_report\nno tools; retry once if\nfinish_reason=content_filter", C["llm"]),
         ("Safety Reviewer", f"skill: {skills_of('safety_reviewer')[0]}\n{len(K['required_checks'])}-check invariant verifier (code)\n+ LLM red team (prompt\nred_team_review), quote-\nconfirmed | revise x{L['max_revisions']}", C["rule"])]
for i, (t, b, col) in enumerate(specs):
    x = 4 + i * 33
    box(x, 59, 30, 20, t, b, col)
    arrow(x + 15, 79, x + 15, 83, both=True)
ax.text(80, 57.4, f"Tool-using agents run the harness loop: context -> LLM (bind_tools) -> tool_calls -> pre-tool-use hook (allow/deny) -> MCP result "
        f"(PHI-redacted) -> back to context, max {L['max_tool_rounds_per_agent']} rounds", ha="center", fontsize=7.6, style="italic", bbox=dict(fc="white", ec="none", pad=0.2))

# ---------------- hook / MCP / gate
box(3, 38, 39, 15, "Permission engine + pre-tool-use hook", "least privilege per agent (bind_tools +\nre-check); arg limits (k<=5, limit<=5,\nexpr<=120 chars); tool budget; session_id\npinned; memory_write needs consent;\nhash-chained audit log (core/permissions.py)", C["guard"], bfs=7.1)
box(49, 33, 76, 20, "MCP server: official mcp SDK (FastMCP), stdio subprocess",
    f"{len(TOOLS)} tools: " + ", ".join(TOOLS[:4]) + ",\n" + ", ".join(TOOLS[4:5]) + ", calculator (AST sandbox), " + ", ".join(TOOLS[6:]) + "\n"
    f"{len(STATIC)} resources: " + ", ".join(STATIC) + f"\n+ {len(TEMPL)} resource template: {TEMPL[0]}\n"
    f"{len(PROMPTS)} prompts: " + ", ".join(PROMPTS) + "\nclient: core/mcp_client.py + langchain-mcp-adapters (sys.executable);\nOPENAI / NG_RECEIPT / GH keys stripped from server env", C["mcp"], fs=9.0, bfs=7.2)
arrow(42, 45, 49, 45, both=True)
ax.text(45.5, 49.5, "stdio\nJSON-RPC", fontsize=6.8, ha="center", va="center")
for x in (19, 52, 85):          # tool-using agents -> hook
    arrow(x, 59, min(x, 38) if x < 40 else 36, 53, color="#a33")
for x in (118, 151):            # prompts/get (allowed_prompt check) - no hook, no tools
    arrow(x, 59, 112 if x < 140 else 124, 53, color="#6a3fa0", ls="--")
ax.text(128, 54.7, "prompts/get (allowed_prompt)", fontsize=6.5, ha="center", va="center", color="#6a3fa0", bbox=dict(fc="white", ec="none", pad=0.3))
box(131, 37.5, 37, 15.5, "Finalize gate (graph node)", f"re-runs the {len(K['required_checks'])} checks on the final draft;\nrefuses to finalize if any fails\n"
    "clinician sign-off if high risk, red\nflags or approved-with-warnings\nmemory_write via hook (consent)", C["rule"], bfs=7.2)
ax.plot([140.4, 167], [86.75, 86.75], color="#333", lw=1.25)
arrow(167, 86.75, 167, 53.4)
ax.text(160, 81.5, "orchestrator\nFINISH", fontsize=7.0, ha="center")
arrow(131, 40, 125.4, 40, color="#a33")
ax.text(128.2, 42.3, "write", fontsize=6.6, ha="center", color="#a33")
box(131, 21, 37, 13, "Signed receipt (core/pipeline.py)", "HMAC-SHA256 over contract, MCP server,\ninputs, report, trace hashes + audit head\n"
    "every run (incl. blocked); clinician\ndecision re-signs it (prior sig chained)", C["rule"], fs=9.0, bfs=7.1)
arrow(149.5, 37.5, 149.5, 34.4)

# skills
box(3, 20, 30, 15, "Skills (skills/, core/skills.py)", f"{len(SK)} SKILL.md: note-screening,\ncohort-comparison, guideline-citation,\nrisk-communication, patient-friendly-\nsummary, red-team-review | catalog =\nname + description only; full body\ninjected when orchestrator selects it", C["state"], fs=8.4, bfs=6.9)
ax.annotate("", xy=(1.6, 66), xytext=(1.6, 34.6), arrowprops=dict(arrowstyle="-|>", color="#666", ls="--", lw=1.1))
ax.text(0.2, 47, "load_body()", rotation=90, fontsize=6.8, va="center", color="#555")

# ---------------- data row
box(3, 2, 33, 15.5, "P1 data + P2 statistics", "Stroke Prediction Dataset (Kaggle,\n5,109 rows) | core/data.py cleaning log\ncore/stats.py: chi-square, Welch t,\nodds ratios, Wilson CIs", C["data"], fs=8.8)
box(40, 2, 32, 15.5, "P3 risk model (core/model.py)", "LR / RF / GBM, class weights, 5-fold CV\nserved: calibrated LR (sigmoid)\nexact log-odds explanations\nthreshold: OOF recall >= 0.80", C["data"], fs=8.8)
box(76, 2, 29, 15.5, "P4 notes (core/notes.py)", "CDC BE FAST rules (emergency)\nTF-IDF + LR on MTSamples (CC0)\nadvisory neuro-note probability", C["data"], fs=8.8)
box(109, 2, 27, 15.5, "P6 RAG (core/rag.py)", "TF-IDF 1-2 gram + topic boost\n30 CDC / NHLBI excerpts\n(public domain, with URLs)\ndata/guidelines.json", C["data"], fs=8.8)
box(140, 2, 28, 15.5, "Long-term memory", "SQLite (core/memory.py)\nde-identified case summaries\nper session; consent + clear", C["data"], fs=8.8)
for x0, x1 in ((34.5, 52), (56, 72), (90, 90), (118, 106), (146, 120)):
    arrow(x0, 17.5, x1, 33, color="#666")
ax.text(85, -1.2, "P5 generative: Clinical Writer (LLM report from MCP prompt).  Honest scope: A2A is in-process; data are public educational datasets; not a medical device.",
        ha="center", fontsize=7.8, color="#333")
ax.text(85, -4.0, f"Repo: {REPO}    |    Live demo: {LIVE}", ha="center", fontsize=8.2, color="#1a56c4")
fig.savefig(ROOT / "architecture_diagram.png", dpi=150, bbox_inches="tight")
fig.savefig(ROOT / "architecture_diagram.pdf", bbox_inches="tight")
print("wrote architecture_diagram.png/.pdf", TOOLS, RES, PROMPTS, list(SK))
