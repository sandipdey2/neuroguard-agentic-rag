# Gardening queue

Generated 2026-11-15T10:29:40+0000 by `scripts/garden.py` (Gardener, MCP prompt `garden_review`). 37 de-identified episodes from episodes_db (30), raw_results.json (7); min_support 2.

**Proposals only.** Nothing here has been applied. The Gardener never edits code, the contract, prompts or checks; a human reviews each item and applies it through a pull request, then runs its validation plan.

## 1. `override:clinical_writer` - support 4

- **Recurring pattern:** the orchestrator's code override corrected an LLM routing choice (4 of 37 episodes; common factors: age, emergency, smoking, glucose)
- **Episode ids:** 5953be60, 93062a46, a8b5209e, c24b1017
- **Proposed constraint (hint):** Orchestrator: when the LLM routes to clinical_writer before risk+evidence exist, the code override fires; make the planner's order binding in the orchestrator prompt (whitelisted hint 'override:clinical_writer').
- **Existing whitelisted hint:** "Plan to gather intake, risk and evidence before asking the writer."
- **Validation plan:** share of runs with 'override:clinical_writer' must fall below the baseline 4/37 (10.8%) on a re-run of eval/cases.json N1, N2; must hold: all 12 required checks stay required (no loosening); finalized rate does not drop; pytest passes; scripts/verify_mcp.py passes. Trial: `python scripts/garden.py --trial --key override:clinical_writer --case N1 [--live]`
- **Status:** proposed (not applied)

## 2. `retrieval:needed` - support 4

- **Recurring pattern:** evidence coverage was short after round 1 and a second retrieval round was needed (4 of 37 episodes; common factors: age, emergency, glucose, hypertension)
- **Episode ids:** 88b930c1, eb0cd921, 875c51d8, 2bf88ce1
- **Proposed constraint (tool):** Evidence Researcher: issue one search_guidelines query per required topic on the first pass (batch them), so the coverage gap that triggers round 2 is closed up front; whitelisted hint 'retrieval:needed' already exists.
- **Existing whitelisted hint:** "Search each risk factor separately and include the warning-signs excerpt on the first pass."
- **Validation plan:** share of runs with 'retrieval:needed' must fall below the baseline 4/37 (10.8%) on a re-run of eval/cases.json N1, N8; must hold: all 12 required checks stay required (no loosening); finalized rate does not drop; pytest passes; scripts/verify_mcp.py passes. Trial: `python scripts/garden.py --trial --key retrieval:needed --case N1 [--live]`
- **Status:** proposed (not applied)

## LLM review (gpt-4o-mini via `garden_review`)

2 proposal(s) kept, 0 dropped by code (unknown support or loosening wording); cost $0.000297.

- Ensure that clinical writers gather intake, risk, and evidence before drafting reports. (support 4): **check** - Require clinical writers to gather intake, risk, and evidence before drafting reports.
- Search each risk factor separately and include the warning-signs excerpt on the first pass. (support 4): **check** - Implement a requirement to search each risk factor separately and include the warning-signs excerpt on the first pass.

## Trial

```json
{
 "key": "override:clinical_writer",
 "case": "N1",
 "hint": "Plan to gather intake, risk and evidence before asking the writer.",
 "live": true,
 "ran": true,
 "how": "NG_TRIAL_HINT injects this whitelisted hint through episode_recall for this one process only",
 "status": "finalized",
 "hint_injected": true,
 "pattern_recurred": false,
 "latency_s": 55.16,
 "cost_usd": 0.005543,
 "run_id": "85147cc6544544eaadef5bf27199a415",
 "unrun_checks": [],
 "git_sha": "0fb8c595af37622bbc568a7229905258eea19331"
}
```
