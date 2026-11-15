"""Harness: contract hash, PreToolUse hook, signed receipt (sign / verify / tamper), audit chain, the 12-check verifier."""
import copy
import json

from core.config import load_contract
from core.harness import RunContext, sha, sign_receipt, verify, verify_receipt
from core.permissions import AuditLog


def test_contract_has_top_p_and_success_criteria():
    c = load_contract()
    assert c["top_p"] == 1.0 and c["temperature"] == 0
    assert c["contract_version"] == "1.4.0"
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


def test_writer_precheck_does_not_fail_red_team_pass_before_red_team_runs():
    """Observed Arize pattern: first verify span was 11/12 with only red_team_pass failed.

    That call was the writer's deterministic pre-check, which invoked verify(..., red_team=None)
    before the Safety Reviewer ran red_team_review. There was no confirmed rule/quote. The
    official 12-check path (default require_red_team=True) must still fail when red team is
    missing or failed; the pre-check must not.
    """
    ctx = RunContext(session_id="s", consent=False)
    risk = {"risk_probability": 0.1498, "risk_percent": 15.0}
    retrieved = {"G-03": {}, "G-13": {}}
    report = _good_report()
    pre = verify(report, risk, {"risk_probability": 0.1498}, retrieved, json.dumps(risk), None, ctx,
                 require_red_team=False, stage="precheck")
    assert "red_team_pass" not in pre["failed"], pre["failed"]
    assert pre["passed"] is True
    assert pre["details"]["verify_stage"] == "precheck"
    # checks still record the raw red_team_pass=False, but it is not a failed official check
    assert pre["checks"]["red_team_pass"] is False

    assert pre["check_status"]["red_team_pass"] == "NOT_RUN" and pre["unrun_checks"] == []

    # v2.6: the official path with no red-team result is NOT_RUN (blocked), never a pass
    official_missing = verify(report, risk, {"risk_probability": 0.1498}, retrieved, json.dumps(risk), None, ctx)
    assert official_missing["check_status"]["red_team_pass"] == "NOT_RUN"
    assert "red_team_pass" in official_missing["unrun_checks"]
    assert official_missing["passed"] is False

    official_fail = verify(report, risk, {"risk_probability": 0.1498}, retrieved, json.dumps(risk),
                           {"verdict": "fail", "issues": ["rule 1: quote"]}, ctx)
    assert "red_team_pass" in official_fail["failed"]
    official_pass = _verify(report)
    assert official_pass["passed"] is True and official_pass["checks"]["red_team_pass"] is True


# ----------------------------------------------------------------------------- v2.6: PASS / FAIL / NOT_RUN
def test_check_status_all_pass_and_unrun_empty():
    v = _verify(_good_report())
    assert set(v["check_status"].values()) == {"PASS"}
    assert v["unrun_checks"] == [] and v["failed"] == [] and v["passed"] is True
    assert set(v["check_status"]) == set(load_contract()["required_checks"])


def test_failed_check_is_fail_not_not_run():
    v = _verify(_good_report().replace("- blood pressure", "- take aspirin 81 mg"))
    assert v["check_status"]["no_dosing_advice"] == "FAIL" and "no_dosing_advice" in v["failed"]
    assert v["unrun_checks"] == []


def test_missing_predict_risk_is_not_run_and_blocks():
    ctx = RunContext(session_id="s", consent=False)
    v = verify(_good_report(), None, None, {"G-03": {}, "G-13": {}}, "{}", {"verdict": "pass", "issues": []}, ctx)
    for c in ("risk_in_unit_interval", "risk_matches_model", "report_states_model_risk"):
        assert v["check_status"][c] == "NOT_RUN" and c in v["unrun_checks"], c
        assert v["checks"][c] is False          # backward-compatible bool view never says True
    assert v["passed"] is False


def test_missing_analyst_only_risk_matches_model_not_run():
    ctx = RunContext(session_id="s", consent=False)
    risk = {"risk_probability": 0.1498, "risk_percent": 15.0}
    v = verify(_good_report(), risk, None, {"G-03": {}, "G-13": {}}, json.dumps(risk), {"verdict": "pass", "issues": []}, ctx)
    assert v["unrun_checks"] == ["risk_matches_model"] and v["passed"] is False


def test_empty_draft_text_checks_are_not_run_not_silent_pass():
    ctx = RunContext(session_id="s", consent=False)
    risk = {"risk_probability": 0.1498, "risk_percent": 15.0}
    v = verify("", risk, {"risk_probability": 0.1498}, {}, json.dumps(risk), {"verdict": "pass", "issues": []}, ctx)
    for c in ("no_dosing_advice", "no_phi_in_output", "no_diagnosis_claim"):
        assert v["check_status"][c] == "NOT_RUN" and c in v["unrun_checks"]
    assert v["passed"] is False


def test_contract_check_unknown_to_verifier_is_not_run():
    ctx = RunContext(session_id="s", consent=False)
    ctx.contract = {**ctx.contract, "required_checks": list(ctx.contract["required_checks"]) + ["future_check"]}
    risk = {"risk_probability": 0.1498, "risk_percent": 15.0}
    v = verify(_good_report(), risk, {"risk_probability": 0.1498}, {"G-03": {}, "G-13": {}}, json.dumps(risk),
               {"verdict": "pass", "issues": []}, ctx)
    assert v["check_status"]["future_check"] == "NOT_RUN" and v["unrun_checks"] == ["future_check"] and not v["passed"]


def test_not_reviewed_result_blocks_every_required_check():
    from core.harness import not_reviewed
    ctx = RunContext(session_id="s", consent=False)
    v = not_reviewed(ctx)
    assert v["passed"] is False and set(v["unrun_checks"]) == set(ctx.contract["required_checks"])
    assert set(v["check_status"].values()) == {"NOT_RUN"}


def test_contract_v14_declares_not_run_policy_and_gardener():
    c = load_contract()
    assert c["verification"]["statuses"] == ["PASS", "FAIL", "NOT_RUN"]
    assert c["agents"]["gardener"]["prompts"] == ["garden_review"] and c["agents"]["gardener"]["tools"] == []
    assert c["gardening"]["mode"] == "propose_only"
