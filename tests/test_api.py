"""FastAPI service with the LLM pipeline mocked: auth, cases, queue worker, receipt verify, webhook trigger, CORS, OpenAPI."""
import hashlib
import hmac
import importlib
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("NG_QUEUE_DB", str(tmp_path / "api_queue.sqlite"))
    monkeypatch.setenv("NG_API_INLINE_WORKER", "0")
    monkeypatch.setenv("NG_API_KEY", "test-api-key")
    monkeypatch.setenv("NG_WEBHOOK_SECRET", "test-webhook-secret")
    monkeypatch.setenv("NG_CORS_ORIGINS", "https://app.example.org")
    from core import job_queue
    job_queue.set_queue(None)
    import api.main as m
    m = importlib.reload(m)
    with TestClient(m.app) as client:
        yield client
    job_queue.set_queue(None)


H = {"X-API-Key": "test-api-key"}
BODY = {"profile": {"age": 67, "gender": "Male", "hypertension": 0, "heart_disease": 1, "ever_married": "Yes",
                    "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 228.69, "bmi": 36.6,
                    "smoking_status": "formerly smoked"}, "rag_mode": "classical"}


def test_health_and_openapi(api):
    h = api.get("/health").json()
    assert h["status"] == "ok" and h["version"] == "2.3.0" and h["queue"]["backend"] == "sqlite" and h["auth"] == "api-key"
    spec = api.get("/openapi.json").json()
    assert {"/v1/cases", "/v1/cases/{job_id}", "/v1/receipts/{job_id}/verify", "/v1/webhooks/case"} <= set(spec["paths"])
    assert api.get("/docs").status_code == 200


def test_auth(api):
    assert api.post("/v1/cases", json=BODY).status_code == 401
    assert api.post("/v1/cases", json=BODY, headers={"X-API-Key": "wrong"}).status_code == 401


def test_unconfigured_key_refuses(api, monkeypatch):
    monkeypatch.delenv("NG_API_KEY")
    assert api.post("/v1/cases", json=BODY, headers=H).status_code == 503


def test_validation(api):
    bad = {**BODY, "profile": {**BODY["profile"], "age": 500}}
    assert api.post("/v1/cases", json=bad, headers=H).status_code == 422


def test_case_flow_with_worker_and_receipt(api, fake_run):
    from core.worker import process_one
    r = api.post("/v1/cases", json=BODY, headers=H)
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert api.get(f"/v1/cases/{jid}", headers=H).json()["status"] == "queued"
    job = process_one(run=fake_run)
    assert job["status"] == "done" and fake_run.calls[0]["rag_mode"] == "classical"
    got = api.get(f"/v1/cases/{jid}", headers=H).json()
    assert got["summary"]["risk_band"] == "high" and got["receipt"]["algorithm"] == "HMAC-SHA256"
    v = api.get(f"/v1/receipts/{jid}/verify", headers=H).json()
    assert v["valid"] is True and len(v["contract_sha256"]) == 64
    assert api.get("/v1/cases/nope", headers=H).status_code == 404
    from core.job_queue import get_queue
    q = get_queue()
    job = q.get(jid)
    job["result"]["receipt"]["status"] = "blocked"                 # tamper with the stored receipt
    q.complete(jid, job["result"])
    assert api.get(f"/v1/receipts/{jid}/verify", headers=H).json()["valid"] is False


def test_webhook_signature(api):
    raw = json.dumps(BODY).encode()
    sig = "sha256=" + hmac.new(b"test-webhook-secret", raw, hashlib.sha256).hexdigest()
    assert api.post("/v1/webhooks/case", content=raw, headers={"Content-Type": "application/json"}).status_code == 401
    assert api.post("/v1/webhooks/case", content=raw, headers={"X-NeuroGuard-Signature": "sha256=00"}).status_code == 401
    r = api.post("/v1/webhooks/case", content=raw, headers={"X-NeuroGuard-Signature": sig, "Content-Type": "application/json"})
    assert r.status_code == 202
    from core.job_queue import get_queue
    assert get_queue().get(r.json()["job_id"])["source"] == "webhook"


def test_cors_for_mobile_web_clients(api):
    r = api.options("/v1/cases", headers={"Origin": "https://app.example.org", "Access-Control-Request-Method": "POST",
                                          "Access-Control-Request-Headers": "X-API-Key,Content-Type"})
    assert r.headers.get("access-control-allow-origin") == "https://app.example.org"
    r = api.options("/v1/cases", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") is None


def test_events_endpoint(api):
    from core.job_queue import get_queue
    get_queue().publish("a2a", {"event": "task/send"})
    assert api.get("/v1/events/a2a", headers=H).json()["events"][-1]["event"] == "task/send"
