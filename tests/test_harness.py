"""Harness: contract hash, PreToolUse hook, signed receipt (sign / verify / tamper), audit chain, the 12-check verifier."""
import copy
import json

from core.config import load_contract
from core.harness import RunContext, sha, sign_receipt, verify, verify_receipt
from core.permissions import AuditLog


def test_contract_has_top_p_and_success_criteria():
    c = load_contract()
    assert c["top_p"] == 1.0 and c["temperature"] == 0
    assert c["contract_version"] == "1.3.0"
    assert len(c["required_checks"]) == 12
    assert {t["metric"] for t in c["success_criteria"]["targets"]} >= {"attack_block_rate", "finalized_rate", "cost_per_run_usd"}


def test_contract_hash_is_sha256_of_canonical_json():
    ctx = RunContext(session_id="s", consent=False)
    assert ctx.contract_sha256 == sha(load_contract())
    assert len(ctx.contract_sha256) == 64
    changed = copy.deepcopy(load_contract())
    changed["budget"]["max_llm_calls"] += 1
    assert sha(changed) != ctx.contract_sha256
    assert ctx.audit.entries[0]["action"] == "contract_loaded" and ctx.audit.entries[0]["detail"]["sha256"] == ctx.contract_sha256


def test_get_llm_passes_top_p(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from core.config import get_llm
    llm = get_llm()
    assert llm.top_p == 1.0 and llm.temperature == 0 and llm.model_name == "gpt-4o-mini"


def test_receipt_sign_verify_and_tamper():
    rc = sign_receipt({"run_id": "r1", "contract_sha256": "a" * 64, "top_p": 1.0, "status": "finalized"})
    assert rc["algorithm"] == "HMAC-SHA256" and verify_receipt(rc)
    for field, value in [("status", "blocked"), ("contract_sha256", "b" * 64), ("top_p", 0.9)]:
        bad = {**rc, field: value}
        assert not verify_receipt(bad), field
    assert not verify_receipt({**rc, "signature": "0" * 64})


def test_audit_chain_detects_tamper():
    a = AuditLog()
    for i in range(4):
        a.log("agent", "tool_call", f"t{i}", "allow")
    assert a.verify_chain()
    a.entries[1]["decision"] = "deny"
    assert not a.verify_chain()


def test_pre_tool_use_hook_least_privilege_and_pinning():
    ctx = RunContext(session_id="ses-1", consent=False)
    p = ctx.perms
    assert not p.pre_tool_use("clinical_writer", "predict_risk", {}).allow                  # not granted
    d = p.pre_tool_use("intake", "memory_read", {"session_id": "other-session", "limit": 99})
    assert d.allow and d.args["session_id"] == "ses-1" and d.args["limit"] == 5            # pinned + clamped
    assert not p.pre_tool_use("harness", "memory_write", {"session_id": "x"}).allow         # needs consent
    assert not p.pre_tool_use("risk_analyst", "calculator", {"expression": "1+" * 100}).allow
    d = p.pre_tool_use("evidence_researcher", "search_guidelines", {"query": "x", "k": 50})
    assert d.allow and d.args["k"] == 5
    denied = [e for e in ctx.audit.entries if e["action"] == "tool_call" and e["decision"] == "deny"]
    assert len(denied) == 3 and ctx.audit.verify_chain()


def test_hook_tool_budget():
    ctx = RunContext(session_id="s", consent=False)
    ctx.perms.tool_calls = ctx.contract["budget"]["max_tool_calls"]
    assert ctx.perms.pre_tool_use("intake", "screen_note", {"note": "x"}).reason == "tool-call budget exhausted"


def _good_report():
    return ("## Summary\nModel estimate.\n## Risk estimate\nThe model risk is 15.0% (high band).\n## Main contributing factors\n- age\n"
            "## Guideline evidence\n- High blood pressure is a leading cause of stroke [G-03].\n- Know the warning signs [G-13].\n"
            "## Points to discuss with the clinician\n- blood pressure\n## Limitations\nEducational dataset.\n"
            "## Disclaimer\nThis is decision support, not a medical device and not a diagnosis.")


def _verify(report, red=None, retrieved=None):
    ctx = RunContext(session_id="s", consent=False)
    risk = {"risk_probability": 0.1498, "risk_percent": 15.0}
    retrieved = retrieved if retrieved is not None else {"G-03": {}, "G-13": {}}
    return verify(report, risk, {"risk_probability": 0.1498}, retrieved, json.dumps(risk), red or {"verdict": "pass", "issues": []}, ctx)


def test_verifier_all_12_checks_pass_on_good_report():
    v = _verify(_good_report())
    assert v["passed"], v["failed"]
    assert set(v["checks"]) == set(load_contract()["required_checks"])


def test_verifier_each_failure_is_caught():
    g = _good_report()
    cases = {
        "report_states_model_risk": g.replace("15.0%", "about fifteen percent"),
        "every_evidence_claim_cited": g.replace(" [G-13]", ""),
        "citations_were_retrieved": g.replace("[G-13]", "[G-29]"),
        "numbers_grounded": g.replace("Model estimate.", "Model estimate of 42 percent."),
        "no_dosing_advice": g.replace("- blood pressure", "- take aspirin 81 mg"),
        "no_diagnosis_claim": g.replace("Model estimate.", "The patient has had a stroke."),
        "disclaimer_present": g.replace("not a medical device and ", ""),
        "no_phi_in_output": g.replace("Model estimate.", "Call 555-123-4567."),
    }
    for check, report in cases.items():
        v = _verify(report)
        assert check in v["failed"], (check, v["failed"])
    assert "red_team_pass" in _verify(g, red={"verdict": "fail", "issues": ["x"]})["failed"]
