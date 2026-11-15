"""Gardener: review past runs for repeated corrections and PROPOSE (never apply) a better check, tool or hint.

Lauren Tan's "Agent Operating System" sheet names three prompts: Coordinator, Verifier and Gardener. The planner
and the Safety Reviewer cover the first two; this script is the Gardener.

Inputs (de-identified only): the episodic-memory table (core/episodes.py; no note, question or report text is
stored there) plus episode records found in eval/raw_results.json and eval/logs/*.json. Free text is never read.

Output: a JSON file and eval/gardening_queue.md listing, for every recurring pattern (support >= min_support):
recurring pattern, support count, episode ids, proposed constraint (check / tool / hint change) and a validation plan.

It never edits code, the contract, prompts or checks. A human applies a proposal through code review.

    python scripts/garden.py                       # deterministic clustering + proposals (no API key)
    python scripts/garden.py --llm                 # also ask gpt-4o-mini via the MCP prompt garden_review
    python scripts/garden.py --trial               # dry-run: show what the top hint trial would inject, on ONE case
    python scripts/garden.py --trial --live        # run that ONE eval/cases.json profile live with the hint (costs ~$0.003)
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import episodes  # noqa: E402
from core.config import load_contract  # noqa: E402

EPISODE_KEYS = {"run_id", "failed_checks", "review_verdicts"}

# Deterministic proposal library: key prefix -> (constraint type, change). Proposals only ADD or TIGHTEN constraints.
PROPOSALS = {
    "check:numbers_grounded": ("check", "Add a writer-side pre-fix that lists the only allowed numbers (FACTS + excerpts) in the clinical_report "
                                        "prompt, plus a regression test with the offending number pattern; numbers_grounded stays required."),
    "check:every_evidence_claim_cited": ("check", "Add a template rule that each Guideline evidence bullet ends with one [G-xx] tag and a unit test "
                                                  "for bullets wrapped over two lines; the check stays required."),
    "check:citations_were_retrieved": ("check", "Pass the retrieved id list to the writer as an enum in structured output so an unretrieved id cannot be emitted."),
    "check:report_states_model_risk": ("check", "Render the risk sentence from predict_risk deterministically (template slot) instead of asking the LLM to copy it."),
    "check:disclaimer_present": ("check", "Append the fixed disclaimer text deterministically after generation; keep the check as the backstop."),
    "check:no_dosing_advice": ("check", "Extend DOSING_PATTERNS with the medicine names seen in the supporting episodes and add a regression test."),
    "check:no_diagnosis_claim": ("check", "Extend DIAGNOSIS_PATTERNS with the phrasing seen in the supporting episodes and add a regression test."),
    "rule:2": ("check", "Promote red-team rule 2 phrasing ('confirms', 'suffered a stroke') into DIAGNOSIS_PATTERNS so it is caught before the red team."),
    "rule:3": ("check", "Add a deterministic lexical-overlap check between each cited bullet and its excerpt (flag low overlap) before red team."),
    "rule:4": ("hint", "Use the whitelisted hint rule:4 and add a regex check for 'certain'/'guaranteed' near the risk sentence."),
    "verdict:revise": ("hint", "Planner hint: run evidence before the writer and inject the most frequent failed check's whitelisted hint on the first draft."),
    "self_critique:revised": ("check", "The writer pre-check (verify.precheck) caught an issue the first draft could avoid: move that rule into the "
                                       "clinical_report template and add a test."),
    "retrieval:needed": ("tool", "Evidence Researcher: issue one search_guidelines query per required topic on the first pass (batch them), so the "
                                 "coverage gap that triggers round 2 is closed up front; whitelisted hint 'retrieval:needed' already exists."),
    "override:clinical_writer": ("hint", "Orchestrator: when the LLM routes to clinical_writer before risk+evidence exist, the code override fires; make "
                                         "the planner's order binding in the orchestrator prompt (whitelisted hint 'override:clinical_writer')."),
    "clinician:REJECTED": ("hint", "Use the whitelisted hint clinician:REJECTED on similar factor sets and surface the rejection in the planner."),
    "outcome:not_finalized": ("check", "Add the most common failed check in these episodes to the writer pre-check so it is fixed before review."),
    "outcome:blocked_unrun": ("tool", "A required check could not run: make the missing input (predict_risk / analyst / red team) a mandatory plan step."),
}


def pattern_keys(e: dict) -> set[str]:
    """Correction signals in one de-identified episode (no free text)."""
    keys = {f"check:{c}" for c in e.get("failed_checks") or []} | {f"rule:{r}" for r in e.get("red_team_rules") or []}
    if "revise" in (e.get("review_verdicts") or []):
        keys.add("verdict:revise")
    if e.get("self_critique") == "revised":
        keys.add("self_critique:revised")
    if int(e.get("retrieval_rounds") or 1) > 1:
        keys.add("retrieval:needed")
    if int(e.get("overrides") or 0) > 0:
        keys.add("override:clinical_writer")
    if e.get("clinician_decision") == "REJECTED":
        keys.add("clinician:REJECTED")
    if e.get("outcome") in ("not_finalized", "blocked_unrun"):
        keys.add(f"outcome:{e['outcome']}")
    return keys


def _walk(obj, found: list):
    if isinstance(obj, dict):
        if EPISODE_KEYS <= set(obj):
            found.append(obj)
        for v in obj.values():
            _walk(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, found)


ALLOWED = {"run_id", "age_band", "factors", "audience", "plan", "route", "overrides", "retrieval_rounds", "self_critique",
           "review_verdicts", "failed_checks", "red_team_rules", "confidence", "outcome", "cited", "clinician_decision"}


def load_episodes(window: int = 500, files: list[str] | None = None, use_db: bool = True) -> tuple[list[dict], dict]:
    """Episodes from the episodic-memory DB and eval logs, de-duplicated by run_id; only whitelisted structured fields are kept."""
    out: dict[str, dict] = {}
    sources: Counter[str] = Counter()
    if use_db:
        try:
            for e in episodes._rows(window):
                out[e["run_id"]] = {k: e.get(k) for k in ALLOWED}
                sources["episodes_db"] += 1
        except Exception as exc:  # noqa: BLE001 - a missing DB just means no DB episodes
            sources["episodes_db_error:" + type(exc).__name__] += 1
    paths = files if files is not None else [str(ROOT / "eval" / "raw_results.json")] + sorted(glob.glob(str(ROOT / "eval" / "logs" / "*.json")))
    for p in paths:
        try:
            data = json.loads(Path(p).read_text())
        except Exception:  # noqa: BLE001
            continue
        found: list = []
        _walk(data, found)
        for e in found:
            rid = str(e["run_id"])
            if rid not in out:
                out[rid] = {k: e.get(k) for k in ALLOWED}
                sources[Path(p).name] += 1
    return list(out.values()), dict(sources)


def cluster(eps: list[dict], min_support: int = 2) -> list[dict]:
    cnt: Counter[str] = Counter()
    ids: dict[str, list[str]] = {}
    factors: dict[str, Counter[str]] = {}
    for e in eps:
        for k in pattern_keys(e):
            cnt[k] += 1
            ids.setdefault(k, []).append(str(e["run_id"])[:8])
            factors.setdefault(k, Counter()).update(e.get("factors") or [])
    return [{"key": k, "support": n, "rate": round(n / max(1, len(eps)), 3), "episode_ids": ids[k],
             "common_factors": [f for f, _ in factors[k].most_common(4)]}
            for k, n in cnt.most_common() if n >= min_support]


def _cases_for(factors: list[str], n: int = 2) -> list[str]:
    try:
        from core.agents import profile_topics
        cases = json.loads((ROOT / "eval" / "cases.json").read_text())
        scored = []
        for c in cases:
            if c.get("kind") != "normal":
                continue
            t = set(profile_topics(c["profile"]))
            scored.append((len(t & set(factors)), c["id"]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [cid for _, cid in scored[:n]]
    except Exception:  # noqa: BLE001
        return ["N1", "N2"]


def propose(patterns: list[dict], n_episodes: int) -> list[dict]:
    lib = episodes.HINT_LIBRARY
    out = []
    for p in patterns:
        typ, change = PROPOSALS.get(p["key"], ("check", f"Add a regression test for '{p['key']}' and a deterministic pre-check."))
        cases = _cases_for(p["common_factors"])
        metric = (f"share of runs with '{p['key']}' must fall below the baseline {p['support']}/{n_episodes} ({p['rate']:.1%}) "
                  f"on a re-run of eval/cases.json {', '.join(cases)}")
        out.append({
            "pattern": describe(p["key"]),
            "key": p["key"], "support": p["support"], "episode_ids": p["episode_ids"], "common_factors": p["common_factors"],
            "proposed_constraint": {"type": typ, "change": change, "existing_whitelisted_hint": lib.get(p["key"])},
            "validation_plan": {"cases": cases, "metric": metric,
                                "must_hold": ["all 12 required checks stay required (no loosening)", "finalized rate does not drop",
                                              "pytest passes", "scripts/verify_mcp.py passes"],
                                "trial": (f"python scripts/garden.py --trial --key {p['key']} --case {cases[0]} [--live]"
                                          if p["key"] in lib else "code change: review in a pull request, then re-run the cases")},
            "status": "proposed (not applied)"})
    return out


def describe(key: str) -> str:
    kind, _, val = key.partition(":")
    return {"check": f"verify check '{val}' failed on a draft and needed correction",
            "rule": f"red-team rule {val} flagged a confirmed quote in a draft",
            "verdict": "the Safety Reviewer sent a draft back for revision",
            "self_critique": "the writer's deterministic pre-check forced a self-revision",
            "retrieval": "evidence coverage was short after round 1 and a second retrieval round was needed",
            "override": "the orchestrator's code override corrected an LLM routing choice",
            "clinician": "a clinician rejected a released report",
            "outcome": f"the finalize gate ended the run as {val}"}.get(kind, key)


async def llm_review(patterns: list[dict]) -> dict:
    """Optional: fetch the garden_review MCP prompt and ask gpt-4o-mini (contract model) for proposals. Proposals only."""
    from langchain_core.messages import HumanMessage
    from core.config import get_llm
    from core.mcp_client import mcp_session
    c = load_contract()
    async with mcp_session() as s:
        p = await s.get_prompt("garden_review", {"patterns_json": json.dumps(patterns), "hint_library_json": json.dumps(episodes.HINT_LIBRARY),
                                                 "required_checks_json": json.dumps(c["required_checks"])})
    llm = get_llm().bind(response_format={"type": "json_object"})
    t0 = time.perf_counter()
    msg = await llm.ainvoke([HumanMessage(p.messages[0].content.text)])
    u = getattr(msg, "usage_metadata", None) or {}
    from core.harness import price
    try:
        body = json.loads(msg.content)
    except Exception:  # noqa: BLE001
        body = {"raw": msg.content[:2000]}
    # code verifies the LLM: drop proposals that cite unknown keys/ids or try to loosen a required check
    known = {p_["key"]: p_ for p_ in patterns}
    kept, dropped = [], 0
    for pr in body.get("proposals", []) if isinstance(body, dict) else []:
        txt = json.dumps(pr).lower()
        loosens = any(w in txt for w in ("remove the check", "disable", "skip the red team", "loosen", "make optional", "drop the check"))
        sup = pr.get("support")
        if loosens or (sup is not None and sup not in {p_["support"] for p_ in known.values()}):
            dropped += 1
            continue
        kept.append(pr)
    return {"model": c["model"], "prompt": "garden_review", "latency_s": round(time.perf_counter() - t0, 2),
            "tokens": u.get("total_tokens"), "cost_usd": round(price(int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)), 0, c["pricing"]), 6),
            "proposals": kept, "dropped_unverifiable_or_loosening": dropped}


def trial(key: str, case_id: str | None, live: bool) -> dict:
    """Trial ONE whitelisted hint on ONE eval/cases.json profile. Without --live this is a dry run (no LLM)."""
    import os
    if key not in episodes.HINT_LIBRARY:
        return {"key": key, "ran": False, "reason": "only existing HINT_LIBRARY keys can be trialled (no free-text hints)"}
    cases = {c["id"]: c for c in json.loads((ROOT / "eval" / "cases.json").read_text())}
    case = cases.get(case_id or "N1")
    if case is None:
        return {"key": key, "ran": False, "reason": f"unknown case {case_id}"}
    res = {"key": key, "case": case["id"], "hint": episodes.HINT_LIBRARY[key], "live": live, "ran": False,
           "how": "NG_TRIAL_HINT injects this whitelisted hint through episode_recall for this one process only"}
    if not live:
        res["reason"] = "dry run (pass --live to run the one case with gpt-4o-mini)"
        return res
    os.environ["NG_TRIAL_HINT"] = key
    try:
        from core.pipeline import run_case
        r = run_case(case["profile"], case["question"], case.get("note", ""), session_id="garden-trial")
    finally:
        os.environ.pop("NG_TRIAL_HINT", None)
    ep = r.get("episode") or {}
    plan: dict = next((t for t in r.get("trace", []) if t.get("type") == "plan"), {})
    res.update({"ran": True, "status": r["status"], "hint_injected": episodes.HINT_LIBRARY[key] in (plan.get("learned_hints") or []),
                "pattern_recurred": key in pattern_keys(ep), "latency_s": r.get("latency_s"),
                "cost_usd": (r.get("cost") or {}).get("cost_usd"), "run_id": r.get("run_id"),
                "unrun_checks": (r.get("receipt") or {}).get("unrun_checks"), "git_sha": (r.get("receipt") or {}).get("git_sha")})
    return res


def to_markdown(rep: dict) -> str:
    L = ["# Gardening queue", "",
         f"Generated {rep['generated_at']} by `scripts/garden.py` (Gardener, MCP prompt `garden_review`). "
         f"{rep['episodes']} de-identified episodes from {', '.join(f'{k} ({v})' for k, v in rep['sources'].items())}; "
         f"min_support {rep['min_support']}.", "",
         "**Proposals only.** Nothing here has been applied. The Gardener never edits code, the contract, prompts or checks; "
         "a human reviews each item and applies it through a pull request, then runs its validation plan.", ""]
    if not rep["proposals"]:
        L += ["No recurring correction pattern reached min_support.", ""]
    for i, p in enumerate(rep["proposals"], 1):
        c, v = p["proposed_constraint"], p["validation_plan"]
        L += [f"## {i}. `{p['key']}` - support {p['support']}", "",
              f"- **Recurring pattern:** {p['pattern']} ({p['support']} of {rep['episodes']} episodes; common factors: {', '.join(p['common_factors']) or '-'})",
              f"- **Episode ids:** {', '.join(p['episode_ids'][:12])}{' ...' if len(p['episode_ids']) > 12 else ''}",
              f"- **Proposed constraint ({c['type']}):** {c['change']}"]
        if c.get("existing_whitelisted_hint"):
            L.append(f"- **Existing whitelisted hint:** \"{c['existing_whitelisted_hint']}\"")
        L += [f"- **Validation plan:** {v['metric']}; must hold: {'; '.join(v['must_hold'])}. Trial: `{v['trial']}`",
              f"- **Status:** {p['status']}", ""]
    if rep.get("llm_review"):
        lr = rep["llm_review"]
        L += ["## LLM review (gpt-4o-mini via `garden_review`)", "",
              f"{len(lr.get('proposals', []))} proposal(s) kept, {lr.get('dropped_unverifiable_or_loosening', 0)} dropped by code "
              f"(unknown support or loosening wording); cost ${lr.get('cost_usd')}.", ""]
        for pr in lr.get("proposals", []):
            pc = pr.get("proposed_constraint") or {}
            L.append(f"- {pr.get('pattern')} (support {pr.get('support')}): **{pc.get('type')}** - {pc.get('change')}")
        L.append("")
    if rep.get("trial"):
        L += ["## Trial", "", "```json", json.dumps(rep["trial"], indent=1), "```", ""]
    return "\n".join(L)


def run(min_support: int = 2, window: int = 500, files: list[str] | None = None, use_db: bool = True) -> dict:
    eps, sources = load_episodes(window, files, use_db)
    pats = cluster(eps, min_support)
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": "propose_only", "episodes": len(eps), "sources": sources,
            "min_support": min_support, "patterns": pats, "proposals": propose(pats, len(eps))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min-support", type=int, default=load_contract().get("gardening", {}).get("min_support", 2))
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--llm", action="store_true", help="also ask gpt-4o-mini through the garden_review MCP prompt")
    ap.add_argument("--trial", action="store_true", help="trial ONE whitelisted hint on ONE eval/cases.json profile")
    ap.add_argument("--key", help="hint key to trial (default: top proposal with a whitelisted hint)")
    ap.add_argument("--case", help="eval/cases.json case id for the trial (default: the proposal's first case)")
    ap.add_argument("--live", action="store_true", help="run the trial case live (otherwise dry run)")
    ap.add_argument("--json-out", default=str(ROOT / "eval" / "logs" / "garden_report.json"))
    ap.add_argument("--md-out", default=str(ROOT / "eval" / "gardening_queue.md"))
    a = ap.parse_args(argv)
    rep = run(a.min_support, a.window)
    if a.llm and rep["patterns"]:
        rep["llm_review"] = asyncio.run(llm_review(rep["patterns"]))
    if a.trial:
        cand = [p for p in rep["proposals"] if p["proposed_constraint"].get("existing_whitelisted_hint")]
        key = a.key or (cand[0]["key"] if cand else None)
        case = a.case or (cand[0]["validation_plan"]["cases"][0] if cand and not a.key else None)
        rep["trial"] = trial(key, case, a.live) if key else {"ran": False, "reason": "no proposal with a whitelisted hint"}
    Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json_out).write_text(json.dumps(rep, indent=1, default=str))
    Path(a.md_out).write_text(to_markdown(rep))
    print(json.dumps({"episodes": rep["episodes"], "sources": rep["sources"],
                      "patterns": [(p["key"], p["support"]) for p in rep["patterns"]], "json": a.json_out, "markdown": a.md_out,
                      "llm_proposals": len((rep.get("llm_review") or {}).get("proposals", [])) if a.llm else None,
                      "trial": rep.get("trial")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
