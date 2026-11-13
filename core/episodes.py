"""Episodic memory and the bounded learning loop.

Episodic memory (``episodes`` table, same SQLite file as long-term memory): after every finalized or refused run
the harness stores one de-identified *episode* - age band, risk-factor names, audience, the plan, the executed
route, number of orchestrator overrides, re-retrievals, self-critique result, reviewer verdicts, failed checks,
red-team rule numbers, confidence, outcome and (later) the clinician's approve/reject decision. No note, question
or free text is stored. ``recall`` returns the most similar past episodes (Jaccard similarity over factor sets plus
an age-band match bonus) so the planner can learn from what happened on similar cases.

Learning loop (memory & learning / self-improvement, deliberately bounded):
  * ``learned_hints`` aggregates feedback over the last ``window`` episodes. A hint is activated only when the same
    kind of feedback (a failed check, a red-team rule, a clinician rejection, an override pattern) occurred at
    least ``min_support`` times. Hint texts come from a fixed whitelist (``HINT_LIBRARY``) - the system never
    writes its own instructions, never edits code, prompts on disk, the contract or the model.
  * at most ``max_hints`` hints are injected (into the planner and writer prompts); each injection is logged
    in the decision log with the supporting episode count;
  * clinician-approved reports give their cited excerpts a small retrieval boost (+0.01 per approval, capped at
    0.06 in core/rag.py) for the same factor; rejections remove that boost;
  * everything is reversible: ``reset_learning`` clears episodes (UI button), and the contract switch
    ``learning.enabled`` turns the loop off.
"""
from __future__ import annotations

import json
import time
from collections import Counter

from .memory import _conn

SCHEMA = """CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT UNIQUE, created_at REAL, age_band TEXT, factors TEXT, audience TEXT,
  plan TEXT, route TEXT, overrides INTEGER, retrieval_rounds INTEGER, self_critique TEXT, review_verdicts TEXT,
  failed_checks TEXT, red_team_rules TEXT, confidence REAL, outcome TEXT, cited TEXT, clinician_decision TEXT)"""

HINT_LIBRARY = {
    "check:numbers_grounded": "Use only numbers that appear in the facts or excerpts; do not round or derive new numbers.",
    "check:every_evidence_claim_cited": "End every Guideline evidence bullet with exactly one [G-xx] citation.",
    "check:citations_were_retrieved": "Cite only excerpt ids that appear in the EVIDENCE list.",
    "check:disclaimer_present": "Keep the Disclaimer section complete: decision support, not a medical device, not a diagnosis.",
    "check:report_states_model_risk": "State the model risk exactly as risk_percent with one decimal and a % sign.",
    "check:no_diagnosis_claim": "Never say the patient has, had or is having a stroke.",
    "check:no_dosing_advice": "Do not name medicines or doses; refer medication questions to the clinician.",
    "rule:3": "Paraphrase each excerpt conservatively; do not add claims the excerpt does not state.",
    "rule:2": "Avoid any wording that sounds like a diagnosis (e.g. 'confirms', 'has had a stroke').",
    "rule:4": "Describe the risk as a model estimate, never as certain or as the patient's true incidence.",
    "clinician:REJECTED": "A clinician rejected a report for a similar case: keep claims minimal and highlight limitations.",
    "override:clinical_writer": "Plan to gather intake, risk and evidence before asking the writer.",
    "retrieval:needed": "Search each risk factor separately and include the warning-signs excerpt on the first pass.",
}


def _c():
    c = _conn()
    c.execute(SCHEMA)
    return c


def write_episode(ep: dict) -> int:
    with _c() as c:
        cur = c.execute("INSERT OR REPLACE INTO episodes(run_id, created_at, age_band, factors, audience, plan, route, overrides, "
                        "retrieval_rounds, self_critique, review_verdicts, failed_checks, red_team_rules, confidence, outcome, cited, "
                        "clinician_decision) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (ep["run_id"], time.time(), ep.get("age_band"), json.dumps(sorted(ep.get("factors", []))), ep.get("audience"),
                         json.dumps(ep.get("plan", [])), json.dumps(ep.get("route", [])), int(ep.get("overrides", 0)),
                         int(ep.get("retrieval_rounds", 1)), ep.get("self_critique"), json.dumps(ep.get("review_verdicts", [])),
                         json.dumps(ep.get("failed_checks", [])), json.dumps(ep.get("red_team_rules", [])),
                         float(ep.get("confidence") or 0), ep.get("outcome"), json.dumps(ep.get("cited", [])), None))
        return int(cur.lastrowid)


def set_clinician_decision(run_id: str, decision: str) -> bool:
    with _c() as c:
        return c.execute("UPDATE episodes SET clinician_decision=? WHERE run_id=?", (decision, run_id)).rowcount > 0


def _rows(limit: int) -> list[dict]:
    with _c() as c:
        rows = c.execute("SELECT run_id, created_at, age_band, factors, audience, plan, route, overrides, retrieval_rounds, self_critique, "
                         "review_verdicts, failed_checks, red_team_rules, confidence, outcome, cited, clinician_decision "
                         "FROM episodes ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    keys = ["run_id", "created_at", "age_band", "factors", "audience", "plan", "route", "overrides", "retrieval_rounds", "self_critique",
            "review_verdicts", "failed_checks", "red_team_rules", "confidence", "outcome", "cited", "clinician_decision"]
    out = []
    for r in rows:
        d = dict(zip(keys, r))
        for k in ("factors", "plan", "route", "review_verdicts", "failed_checks", "red_team_rules", "cited"):
            d[k] = json.loads(d[k] or "[]")
        out.append(d)
    return out


def recall(factors: list[str], age_band: str | None = None, k: int = 3, window: int = 200) -> list[dict]:
    fs = set(factors)
    scored = []
    for e in _rows(window):
        ef = set(e["factors"])
        sim = (len(fs & ef) / len(fs | ef) if fs | ef else 0.0) + (0.2 if age_band and e["age_band"] == age_band else 0.0)
        scored.append((round(sim, 3), e))
    scored.sort(key=lambda t: -t[0])
    return [{"similarity": s, **{k_: e[k_] for k_ in ("run_id", "age_band", "factors", "route", "overrides", "retrieval_rounds",
                                                      "self_critique", "review_verdicts", "failed_checks", "red_team_rules",
                                                      "confidence", "outcome", "clinician_decision")}}
            for s, e in scored[:max(1, min(int(k), 5))] if s > 0]


def learned_hints(window: int = 30, min_support: int = 2, max_hints: int = 3) -> list[dict]:
    cnt, support = Counter(), {}
    for e in _rows(window):
        keys = {f"check:{c}" for c in e["failed_checks"]} | {f"rule:{r}" for r in e["red_team_rules"]}
        if e["clinician_decision"] == "REJECTED":
            keys.add("clinician:REJECTED")
        if e["overrides"]:
            keys.add("override:clinical_writer")
        if e["retrieval_rounds"] > 1:
            keys.add("retrieval:needed")
        for k_ in keys & set(HINT_LIBRARY):
            cnt[k_] += 1
            support.setdefault(k_, []).append(e["run_id"][:8])
    active = [(k_, n) for k_, n in cnt.most_common() if n >= min_support][:max_hints]
    return [{"key": k_, "support": n, "hint": HINT_LIBRARY[k_], "episodes": support[k_][:5]} for k_, n in active]


def retrieval_boosts(factors: list[str], window: int = 200, step: float = 0.01) -> dict:
    """Per-chunk boost from clinician-APPROVED episodes that share a factor (rejections cancel approvals)."""
    b: dict[str, float] = {}
    fs = set(factors)
    for e in _rows(window):
        if not (fs & set(e["factors"])) or e["clinician_decision"] not in ("APPROVED", "REJECTED"):
            continue
        sign = 1 if e["clinician_decision"] == "APPROVED" else -1
        for cid in e["cited"]:
            b[cid] = b.get(cid, 0.0) + sign * step
    return {k_: round(v, 3) for k_, v in b.items() if v > 0}


def stats() -> dict:
    rows = _rows(10_000)
    return {"episodes": len(rows), "with_clinician_decision": sum(1 for e in rows if e["clinician_decision"]),
            "outcomes": dict(Counter(e["outcome"] for e in rows))}


def reset_learning() -> int:
    with _c() as c:
        return c.execute("DELETE FROM episodes").rowcount
