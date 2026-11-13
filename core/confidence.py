"""Confidence scoring and the confidence policy gate.

The score (0-1) combines six components, each in [0, 1]; weights and gate thresholds live in the contract
(``confidence``) so they are hashed into every receipt:
  calibration   1 - |observed - predicted| / 0.10 in the held-out reliability bin nearest the patient's probability
                (how well the calibrated model has matched reality for patients like this one)
  cohort        min(1, n / 100) for the cohort_stats subgroup (how much data supports the comparison)
  coverage      share of the patient's required evidence topics (factors + warning signs) covered by a cited excerpt
  verifier      share of required invariant checks that passed
  red_team      1.0 if the quote-confirmed red team passed, else 0.5
  grounding     mean word overlap between each evidence bullet and its cited excerpt (a lexical proxy)
Gate policy: score < ``refuse_below`` -> the gate refuses to finalize ("low_confidence"); score < ``review_below``
-> a clinician must approve; otherwise normal rules apply. The score is a heuristic summary of evidence quality,
not a probability that the report is correct, and it is labelled that way in the UI.
"""
from __future__ import annotations

import re

STOP = set("a an the and or of to in on for with by is are was were be been it its this that as at from can your you more have has".split())


def _words(t: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP and len(w) > 2}


def grounding_support(report_md: str, retrieved: dict, section_fn) -> float | None:
    ev = section_fn(report_md, "Guideline evidence")
    sups = []
    for b in re.findall(r"^\s*[-*]\s+(.+)$", ev, flags=re.M):
        cid = (re.findall(r"\[(G-\d{2})\]", b) or [None])[0]
        w = _words(re.sub(r"\[G-\d{2}\]", "", b))
        src = _words((retrieved.get(cid) or {}).get("text", ""))
        if w:
            sups.append(len(w & src) / len(w))
    return round(sum(sups) / len(sups), 3) if sups else None


def calibration_component(prob: float | None, curve: dict) -> float:
    if prob is None or not curve.get("mean_predicted"):
        return 0.0
    mp, obs = curve["mean_predicted"], curve["observed_rate"]
    i = min(range(len(mp)), key=lambda j: abs(mp[j] - prob))
    return round(max(0.0, 1 - abs(obs[i] - mp[i]) / 0.10), 3)


def score(components: dict, weights: dict) -> float:
    tot = sum(weights.values())
    return round(sum(weights[k] * float(components.get(k) or 0.0) for k in weights) / tot, 3)


def gate_decision(value: float, policy: dict) -> str:
    if value < policy["refuse_below"]:
        return "refuse"
    if value < policy["review_below"]:
        return "require_clinician_review"
    return "pass"
