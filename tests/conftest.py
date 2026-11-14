"""Test isolation: every test session gets its own runtime dir (SQLite files, FAISS index, model cache); web access
is off (PubMed / CDC fall back to the committed real snapshots); no API key is needed - LLM calls are mocked."""
import os
import sys
import tempfile
from pathlib import Path

_RT = tempfile.mkdtemp(prefix="ng-test-")
os.environ["NG_RUNTIME_DIR"] = _RT
os.environ["NG_WEB_OFFLINE"] = "1"
os.environ.pop("DATABASE_URL", None)
os.environ.pop("REDIS_URL", None)
os.environ.pop("NG_QUEUE_URL", None)
os.environ.setdefault("NG_RECEIPT_KEY", "test-receipt-key-not-a-secret")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

GOOD_PROFILE = {"age": 67.0, "gender": "Male", "hypertension": 0, "heart_disease": 1, "ever_married": "Yes", "work_type": "Private",
                "Residence_type": "Urban", "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked"}


@pytest.fixture
def profile():
    return dict(GOOD_PROFILE)


@pytest.fixture
def runtime_dir():
    return Path(_RT)


def fake_result(profile=None, status="finalized"):
    """A pipeline-shaped result with a REAL signed receipt (used wherever run_case is mocked)."""
    from core.harness import RunContext, sha, sign_receipt
    ctx = RunContext(session_id="test", consent=False)
    risk = {"risk_probability": 0.1498, "risk_percent": 15.0, "risk_band": "high"}
    rc = sign_receipt({"run_id": ctx.run_id, "contract_sha256": ctx.contract_sha256, "contract_version": ctx.contract["contract_version"],
                       "top_p": ctx.contract["top_p"], "status": status, "inputs_sha256": sha(profile or {})})
    return {"run_id": ctx.run_id, "status": status, "risk": risk, "analyst": {"top_factors": ["age (67)", "glucose (228.69)"],
            "cohort_comparison": "60-69: 7.0%"}, "evidence": {"findings": [{"claim": "High blood pressure raises stroke risk.", "citation_id": "G-03"}]},
            "final": {"verification": {"failed": []}, "needs_human_approval": True, "approval_reasons": ["risk_band == high"]},
            "confidence": {"score": 0.91}, "latency_s": 1.2, "cost": {"cost_usd": 0.0047}, "receipt": rc, "report": "## Summary\nok",
            "a2a": [], "input_guardrail": {"reason": "ok"}, "rag_mode": "agentic"}


@pytest.fixture
def fake_run():
    calls = []

    def run(profile, question, note="", audience="clinician", session_id="x", consent=False, **kw):
        calls.append({"profile": profile, "question": question, "session_id": session_id, **kw})
        return fake_result(profile)
    run.calls = calls
    return run
