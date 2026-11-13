"""Draw architecture_diagram.png / .pdf (matplotlib).   python scripts/draw_architecture.py
Every box names the file that implements it; every arrow is a real call path."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
fig, ax = plt.subplots(figsize=(17, 11))
ax.set_xlim(0, 170); ax.set_ylim(0, 110); ax.axis("off")
C = {"ui": "#e8f0fe", "guard": "#fde7e7", "llm": "#fff4d6", "rule": "#e6f4ea", "mcp": "#efe7fb", "data": "#f1f3f4", "state": "#ffffff"}


def box(x, y, w, h, title, body="", color="#fff", fs=9.6, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#333", lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=fs, weight="bold")
    if body:
        ax.text(x + w / 2, y + h - 5.3, body, ha="center", va="top", fontsize=7.5, linespacing=1.25)


def arrow(x1, y1, x2, y2, label="", both=False, color="#333", ls="-", rad=0.0, lx=0, ly=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="<|-|>" if both else "-|>", mutation_scale=12,
                                 color=color, lw=1.25, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=7.2, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5))


ax.text(85, 108, "NeuroGuard AI - architecture as implemented (app.py, core/*.py, mcp_server/neuroguard_mcp_server.py)",
        ha="center", fontsize=13, weight="bold")

# row 1: UI -> guardrail -> contract -> orchestrator ; state
box(2, 88, 27, 15, "Streamlit UI (app.py)", "profile form, note, audience, consent\nlive trace, A2A log, MCP calls, charts\nclinician approve / reject\nreport .md/.pdf + receipt.json", C["ui"])
box(33, 87, 32, 16, "Input guardrails\n(core/guardrails.py)", "\nPHI redaction -> injection ->\nBE FAST emergency (call 9-1-1) ->\ndosing request -> out of scope\n(no LLM call)", C["guard"])
box(69, 87, 31, 16, "Run contract\n(harness_contract.json)", "\nhashed before the first token\n9 turns | 30 LLM | 24 tool calls\n80k tokens | 240 s | permissions\n12 required checks", C["rule"])
box(104, 87, 34, 16, "Orchestrator (LLM gpt-4o-mini, T=0)", "structured output: next agent, skill,\ninstruction | code rules: eligibility,\nFINISH only after review of latest draft,\n1 revision, visit + turn cap, budget\n(core/agents.py)", C["llm"])
box(143, 87, 25, 16, "CaseState (LangGraph)", "short-term memory per run:\nprofile, intake, analyst, risk,\nevidence, retrieved, draft,\nreview, turns, visits, trace[]", C["state"], ls="--")
arrow(29, 96, 33, 96)
arrow(65, 96, 69, 96)
arrow(100, 96, 104, 96)
arrow(138, 95, 143, 95, ls="--")
ax.text(49, 84.6, "blocked -> refusal / emergency message", fontsize=7.2, ha="center", color="#b00")

# A2A bus
box(40, 73, 96, 8.5, "A2A-style message bus (core/a2a.py) - Agent Cards, Task -> Message -> Artifact, in-process (no network)", "", C["data"], fs=9)
arrow(121, 87, 121, 81.5, "Task + skill", both=True, lx=7)

# specialists
specs = [("Intake", "skill: note-screening\nMCP: screen_note,\nmemory_read", C["llm"]),
         ("Risk Analyst", "skill: cohort-comparison\nMCP: predict_risk, explain_\nfactors, cohort_stats,\ncalculator + model://risk-card", C["llm"]),
         ("Evidence Researcher", "skill: guideline-citation\nMCP: search_guidelines\n+ guideline://{id} reads", C["llm"]),
         ("Clinical Writer", "skill: risk-communication |\npatient-friendly-summary\nMCP prompt: clinical_report", C["llm"]),
         ("Safety Reviewer", "invariant verifier (code)\n+ LLM red team (prompt\nred_team_review), quote-\nconfirmed; revise x1", C["rule"])]
for i, (t, b, col) in enumerate(specs):
    x = 4 + i * 32
    box(x, 50, 29, 18, t, b, col)
    arrow(x + 14.5, 68, x + 14.5, 73, both=True)
ax.text(85, 46.4, "Tool-using agents run the harness loop: context -> LLM -> tool call -> pre-tool-use hook (allow/deny) -> MCP result (PHI-redacted) -> back to context, max 4 rounds",
        ha="center", fontsize=7.6, style="italic")

# hook + MCP
box(4, 30, 40, 13, "Permission engine + pre-tool-use hook", "least privilege per agent (bind_tools + re-check)\narg limits, budget, session_id pinned,\nmemory_write needs consent; hash-chained\naudit log (core/permissions.py)", C["guard"])
box(51, 24, 72, 20, "MCP server: official mcp SDK (FastMCP), stdio subprocess", "8 tools: predict_risk, explain_factors, cohort_stats, screen_note,\nsearch_guidelines, calculator (AST sandbox), memory_read, memory_write\n"
    "5 resources: guideline://index, guideline://{chunk_id}, model://risk-card,\ndataset://stroke-card, policy://harness-contract\n"
    "2 prompts: clinical_report, red_team_review\nclient: core/mcp_client.py + langchain-mcp-adapters; API key stripped from env", C["mcp"], fs=9.2)
arrow(44, 36, 51, 36, both=True)
ax.text(47.5, 38.5, "stdio\nJSON-RPC", fontsize=7, ha="center")
for x in (18.5, 50.5, 82.5):
    arrow(x, 50, 24 if x < 20 else 40, 43, color="#a33")
box(129, 24, 39, 20, "Finalize gate -> receipt", "re-runs verifier on final draft;\nrefuses to finalize if any check fails\nhigh risk / red flags -> clinician sign-off\nmemory_write (consent) via hook\nHMAC-SHA256 receipt.json", C["rule"])
ax.plot([136.4, 166], [77, 77], color="#333", lw=1.25)
arrow(166, 77, 166, 44.4)
ax.text(163, 72, "orchestrator\nFINISH", fontsize=7.2, ha="center")
arrow(128, 30, 123, 30, label="")

# data row
box(4, 4, 33, 16, "P1 pipeline + P2 statistics", "core/data.py: schema, N/A->NaN, filter,\ntrain-split imputation | core/stats.py:\nchi-square, Welch t, odds ratios,\nWilson CIs (cohort_stats)", C["data"])
box(41, 4, 33, 16, "P3 risk model (core/model.py)", "LR / RF / GBM, class weights, 5-fold CV\nserved: calibrated LR (sigmoid)\nexact log-odds explanations\nthreshold: OOF recall >= 0.80", C["data"])
box(78, 4, 30, 16, "P4 note screening (core/notes.py)", "CDC BE FAST rules (emergency)\nTF-IDF + LR on MTSamples (CC0)\nadvisory neuro-note probability", C["data"])
box(112, 4, 27, 16, "P6 RAG (core/rag.py)", "TF-IDF 1-2 gram + topic boost\n30 verbatim CDC / NHLBI\nexcerpts with URLs\n(data/guidelines.json)", C["data"])
box(143, 4, 25, 16, "Long-term memory", "SQLite (core/memory.py)\nde-identified case summaries\nper session, consent + clear", C["data"])
for x in (20, 57, 93, 125, 155):
    arrow(x, 20, min(max(x, 56), 118), 26, color="#666")
ax.text(85, 1.2, "P5 generative: Clinical Writer (LLM report from MCP prompt).  Honest scope: A2A is in-process; data are public educational datasets; not a medical device.",
        ha="center", fontsize=8, color="#333")
fig.savefig(ROOT / "architecture_diagram.png", dpi=150, bbox_inches="tight")
fig.savefig(ROOT / "architecture_diagram.pdf", bbox_inches="tight")
print("wrote architecture_diagram.png/.pdf")
