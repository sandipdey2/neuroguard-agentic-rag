"""Harness: run contract, budget accounting, deterministic invariant verifier, finalize gate and signed receipt.

The contract (harness_contract.json) is loaded and hashed before the first LLM token. ``RunContext`` counts
LLM calls, tokens and tool calls against it. ``verify`` evaluates every required check on the final draft;
``gate`` refuses to finalize unless all of them pass. ``sign_receipt`` produces an HMAC-SHA256 receipt over
hashes of the inputs, the report, the trace and the check results.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field

from .a2a import MessageBus
from .config import load_contract, secret
from .guardrails import PHI
from .permissions import AuditLog, PermissionEngine

DISCLAIMER_PATTERNS = [r"not a medical device", r"decision support"]
DOSING_PATTERNS = [r"\b\d+(\.\d+)?\s?(mg|mcg|µg|milligrams?|micrograms?|units?|ml)\b(?!\s?/\s?dl)",
                   r"\b(dose|doses|dosage|dosing|titrate|titration)\b",
                   r"\b(aspirin|statin|atorvastatin|warfarin|apixaban|rivaroxaban|clopidogrel|metformin|insulin|lisinopril|amlodipine|tpa|alteplase)\b"]
DIAGNOSIS_PATTERNS = [r"\b(you|the patient|patient|he|she) (has|had|is having|has had) (a )?stroke\b", r"\bdiagnos(is|ed) (of|with) stroke\b",
                      r"\bconfirm(s|ed)? (a )?stroke\b"]


def sha(obj) -> str:
    data = obj if isinstance(obj, (bytes, str)) else json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


@dataclass
class RunContext:
    session_id: str
    consent: bool
    contract: dict = field(default_factory=load_contract)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.time)
    llm_calls: int = 0
    tokens: int = 0
    audit: AuditLog = field(default_factory=AuditLog)
    bus: MessageBus = field(default_factory=MessageBus)
    perms: PermissionEngine = None

    def __post_init__(self):
        self.contract_sha256 = sha(self.contract)
        self.perms = PermissionEngine(self.contract, self.audit, self.session_id, self.consent)
        self.audit.log("harness", "contract_loaded", "harness_contract.json", "ok", sha256=self.contract_sha256,
                       budget=self.contract["budget"], loop=self.contract["loop"])

    def record_llm(self, agent: str, msg) -> None:
        self.llm_calls += 1
        usage = getattr(msg, "usage_metadata", None) or {}
        self.tokens += int(usage.get("total_tokens", 0))
        self.audit.log(agent, "llm_call", self.contract["model"], "ok", tokens=usage.get("total_tokens"))

    def budget_left(self) -> bool:
        b = self.contract["budget"]
        return (self.llm_calls < b["max_llm_calls"] and self.tokens < b["max_total_tokens"]
                and time.time() - self.started < b["wall_clock_timeout_s"])

    def within_budget(self) -> bool:
        b = self.contract["budget"]
        return (self.llm_calls <= b["max_llm_calls"] and self.tokens <= b["max_total_tokens"]
                and self.perms.tool_calls <= b["max_tool_calls"] and time.time() - self.started <= b["wall_clock_timeout_s"])


# ----------------------------------------------------------------------------- invariant verifier
NUM = re.compile(r"(?<![A-Za-z\-])(\d+(?:\.\d+)?)(?![\d\-])")


def _numbers(text: str) -> list[float]:
    text = re.sub(r"\[G-\d+\]", " ", text)                 # citation ids are not claims
    text = re.sub(r"^\s*#+.*$", " ", text, flags=re.M)       # headings
    text = re.sub(r"^\s*\d+\.\s", " ", text, flags=re.M)     # list numbering
    text = text.replace("9-1-1", " ")
    return [float(x) for x in NUM.findall(text)]


def _grounded(x: float, pool: set[float]) -> bool:
    if x in pool:
        return True
    for v in pool:
        for cand in (v, v * 100, v / 100):
            if abs(cand - x) <= max(0.051, 0.0051 * abs(x)) and round(cand, 1) == round(x, 1):
                return True
        if v != 0 and abs(round(v * 100) - x) < 1e-9:
            return True
    return False


def section(md: str, title: str) -> str:
    m = re.search(rf"^##\s*{re.escape(title)}\s*$(.*?)(?=^##\s|\Z)", md, flags=re.M | re.S | re.I)
    return m.group(1) if m else ""


def verify(report_md: str, risk: dict | None, analyst: dict | None, retrieved: dict, evidence_text: str,
           red_team: dict | None, ctx: RunContext) -> dict:
    checks = {}
    p = (risk or {}).get("risk_probability")
    checks["risk_in_unit_interval"] = isinstance(p, (int, float)) and 0.0 <= p <= 1.0
    checks["risk_matches_model"] = bool(analyst) and p is not None and abs(float(analyst.get("risk_probability", -1)) - p) < 5e-4
    pct = (risk or {}).get("risk_percent")
    checks["report_states_model_risk"] = pct is not None and f"{pct:.1f}%" in report_md
    ev = section(report_md, "Guideline evidence")
    bullets = [b.strip() for b in re.findall(r"^\s*[-*]\s+(.+)$", ev, flags=re.M)]
    uncited = [b for b in bullets if not re.search(r"\[G-\d{2}\]", b)]
    checks["every_evidence_claim_cited"] = bool(bullets) and not uncited
    cited = sorted(set(re.findall(r"\[(G-\d{2})\]", report_md)))
    bad = [c for c in cited if c not in retrieved]
    checks["citations_were_retrieved"] = bool(cited) and not bad
    nums = _numbers(report_md)
    pool = set(_numbers(evidence_text)) | {float(x) for x in NUM.findall(evidence_text)} | {95.0, 100.0}  # "95% CI", "in 100"
    ungrounded = sorted({x for x in nums if not _grounded(x, pool)})
    checks["numbers_grounded"] = not ungrounded
    low = report_md.lower()
    dosing = [m.group(0) for pat in DOSING_PATTERNS for m in [re.search(pat, low)] if m]
    checks["no_dosing_advice"] = not dosing
    diag = [m.group(0) for pat in DIAGNOSIS_PATTERNS for m in [re.search(pat, low)] if m]
    checks["no_diagnosis_claim"] = not diag
    checks["disclaimer_present"] = all(re.search(p_, low) for p_ in DISCLAIMER_PATTERNS)
    phi = [n for n, pat in PHI if n != "person_name" and re.search(pat, report_md)]
    checks["no_phi_in_output"] = not phi
    checks["red_team_pass"] = bool(red_team) and red_team.get("verdict") == "pass"
    checks["within_budget"] = ctx.within_budget()
    details = {"uncited_bullets": uncited, "invalid_citations": bad, "citations": cited, "ungrounded_numbers": ungrounded,
               "dosing_matches": dosing, "diagnosis_matches": diag, "phi_types": phi,
               "red_team_issues": (red_team or {}).get("issues", []), "evidence_bullets": len(bullets),
               "numbers_checked": len(nums)}
    required = ctx.contract["required_checks"]
    failed = [c for c in required if not checks.get(c)]
    return {"checks": checks, "failed": failed, "passed": not failed, "details": details}


# ----------------------------------------------------------------------------- receipt
_EPHEMERAL = os.urandom(32)


def _key() -> tuple[bytes, str]:
    k = secret("NG_RECEIPT_KEY")
    if k:
        return k.encode(), "NG_RECEIPT_KEY:" + hashlib.sha256(k.encode()).hexdigest()[:8]
    return _EPHEMERAL, "ephemeral:" + hashlib.sha256(_EPHEMERAL).hexdigest()[:8]


def sign_receipt(body: dict) -> dict:
    key, key_id = _key()
    body = {**body, "key_id": key_id, "algorithm": "HMAC-SHA256", "signed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return {**body, "signature": hmac.new(key, canon.encode(), hashlib.sha256).hexdigest()}


def verify_receipt(receipt: dict) -> bool:
    key, key_id = _key()
    if receipt.get("key_id") != key_id:
        return False
    body = {k: v for k, v in receipt.items() if k != "signature"}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hmac.compare_digest(hmac.new(key, canon.encode(), hashlib.sha256).hexdigest(), receipt.get("signature", ""))
