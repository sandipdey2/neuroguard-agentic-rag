"""Input guardrails and PHI redaction (deterministic, run before any LLM call).

Order: redact PHI -> prompt-injection check -> emergency red flags -> medication-dosing request -> scope check.
Every decision is returned as a dict and written to the audit log by the harness.
"""
from __future__ import annotations

import re

from .notes import red_flags

INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (your|the|all) (rules|instructions|guardrails|system prompt)",
    r"(reveal|print|show|repeat|output) (your|the) (system prompt|hidden prompt|instructions|api key|secrets?)",
    r"\b(jailbreak|developer mode|dan mode|do anything now)\b",
    r"you are now (a|an|in) ",
    r"(call|use|invoke) (the )?(memory_write|tool) .*(all|every|delete|other patients)",
    r"(act|pretend) as (if you are |)(an? )?(unrestricted|unfiltered|doctor who prescribes)",
    r"</?(system|assistant|tool)>",
    r"(bypass|disable|turn off) (the )?(guardrails?|safety|filters?|verifier)",
]
DOSING = [
    r"\b(how (much|many)|what|which|correct|right|safe|max(imum)?)\b.{0,40}\b(dose|dosage|mg|milligrams?|tablets?|pills?)\b",
    r"\b(prescribe|prescription for|should i take|can i take|start taking|stop taking)\b",
    r"\b(dose|dosage|titrat\w*)\b.{0,30}\b(of|for)\b",
]
IN_SCOPE = r"(stroke|risk|blood pressure|hypertension|glucose|diabet|bmi|weight|smok|heart|cholesterol|prevent|factor|patient|" \
           r"tia|symptom|report|explain|summar|lifestyle|exercise|activity|age|guideline|cdc|neuro|brain|clinic|health|doctor|nurse|follow)"
PHI = [
    ("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("phone", r"(?<!\d)(\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
    ("mrn", r"\b(mrn|medical record( number)?|patient id)[:#\s]*[A-Za-z0-9-]{4,}\b"),
    ("date_of_birth", r"\b(dob|date of birth|born on)[:\s]*\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b"),
    ("street_address", r"\b\d{1,5}\s+([A-Z][a-z]+\s){1,3}(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b"),
    ("person_name", r"\b(Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-z]+(\s[A-Z][a-z]+)?"),
    ("api_key", r"\b(sk|voc)-[A-Za-z0-9._-]{12,}"),
]
EMERGENCY_MESSAGE = ("Possible stroke warning signs were described. This is a medical emergency: call 9-1-1 (or your local "
                     "emergency number) now and note the time symptoms started. Do not drive yourself to hospital. "
                     "NeuroGuard does not assess emergencies and has stopped this request.")


def redact(text: str) -> tuple[str, list[str]]:
    found = []
    out = text or ""
    for name, pat in PHI:
        out, n = re.subn(pat, f"[REDACTED-{name.upper()}]", out, flags=re.IGNORECASE if name in ("mrn", "date_of_birth") else 0)
        if n:
            found += [name] * n
    return out, found


def check_input(question: str, note: str = "") -> dict:
    q_red, q_found = redact(question)
    n_red, n_found = redact(note)
    both = f"{q_red}\n{n_red}".lower()
    base = {"question": q_red, "note": n_red, "phi_redacted": q_found + n_found}
    for pat in INJECTION:
        if re.search(pat, both):
            return {**base, "allowed": False, "reason": "prompt_injection", "detail": f"matched pattern: {pat[:60]}"}
    rf = red_flags(f"{question}\n{note}")
    if rf["possible_emergency"]:
        return {**base, "allowed": False, "reason": "emergency", "detail": EMERGENCY_MESSAGE, "red_flags": rf["signs"]}
    if any(re.search(p, q_red.lower()) for p in DOSING):
        return {**base, "allowed": False, "reason": "medication_dosing",
                "detail": "Medication choice and dosing are outside NeuroGuard's scope. Ask the treating clinician or pharmacist."}
    if q_red.strip() and len(q_red.split()) > 2 and not re.search(IN_SCOPE, q_red.lower()):
        return {**base, "allowed": False, "reason": "out_of_scope",
                "detail": "NeuroGuard only answers questions about stroke-risk factors for the profile entered."}
    return {**base, "allowed": True, "reason": "ok", "detail": "passed input guardrails", "red_flags": rf["signs"]}
