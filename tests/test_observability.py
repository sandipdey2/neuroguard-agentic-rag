"""Phoenix / Arize AX / OpenTelemetry tracing (in-memory exporter) and the monitoring aggregation store."""
import json
import time

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from core.harness import RunContext, verify
from core.observability import (
    arize_requested,
    arize_status,
    attach_exporter,
    configure,
    emit_tool_span,
    export_status,
    force_flush,
    init_tracing,
    node_span,
    phoenix_requested,
    phoenix_status,
    reset_tracing,
    safe_attributes,
    start_span,
)
from core import monitoring


@pytest.fixture
def exporter():
    reset_tracing()
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    configure(provider)
    yield exp
    reset_tracing()
    exp.clear()


def _attrs(span) -> dict:
    return dict(span.attributes or {})


def _blob(spans) -> str:
    return json.dumps([{"name": s.name, "attrs": _attrs(s)} for s in spans], default=str)


def test_phoenix_is_off_by_default():
    reset_tracing()
    assert phoenix_requested() is False
    status = phoenix_status()
    assert status["enabled"] is False and status["project"] == "neuroguard"
    assert "off" in status["reason"].lower() or "unset" in status["reason"].lower()


def test_noop_mode_does_not_raise(monkeypatch):
    reset_tracing()
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    monkeypatch.delenv("NG_PHOENIX", raising=False)
    monkeypatch.delenv("ARIZE_SPACE_ID", raising=False)
    monkeypatch.delenv("ARIZE_API_KEY", raising=False)
    ctx = RunContext(session_id="s", consent=False)
    with node_span("planner", ctx) as span:
        assert span is None
    emit_tool_span(ctx, "risk_analyst", "predict_risk", time.time(), time.time(), "allow", args={"age": 67}, ok=True)
    verify("## Summary\nnot a medical device. decision support.\n", None, None, {}, "", None, ctx)


def test_node_mcp_and_verify_spans(exporter):
    ctx = RunContext(session_id="s", consent=False)
    with node_span("planner", ctx) as span:
        assert span is not None
    with node_span("gate", ctx):
        pass
    ctx.record_tool_span("risk_analyst", "predict_risk", time.time(), time.time() + 0.01, "allow",
                         args={"age": 67, "hypertension": 1}, ok=True)
    report = ("## Summary\nDecision support only, not a medical device.\n"
              "## Risk estimate\n15.0%\n"
              "## Guideline evidence\n- High blood pressure raises stroke risk. [G-03]\n")
    # A realistic verify() call; we only care that a span is emitted.
    verify(report, {"risk_probability": 0.15, "risk_percent": 15.0},
           {"risk_probability": 0.15}, {"G-03": "High blood pressure raises stroke risk."},
           "High blood pressure raises stroke risk. 15.0 0.15",
           {"verdict": "pass", "issues": []}, ctx)
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "node.planner" in names and "node.gate" in names
    assert "mcp.tools/call.predict_risk" in names
    assert "verify" in names
    by = {s.name: s for s in spans}
    planner_attrs = _attrs(by["node.planner"])
    assert planner_attrs.get("case.id") == ctx.run_id
    assert planner_attrs.get("contract.sha256") == ctx.contract_sha256
    assert "run.cost_usd" in planner_attrs and "run.tokens" in planner_attrs
    assert "run.llm_calls" in planner_attrs and "run.tool_calls" in planner_attrs
    tool_attrs = _attrs(by["mcp.tools/call.predict_risk"])
    assert tool_attrs.get("tool") == "predict_risk" and tool_attrs.get("hook") == "allow"
    verify_attrs = _attrs(by["verify"])
    assert "checks.passed" in verify_attrs and "checks.failed" in verify_attrs


def test_span_attributes_redact_phi_and_keys(exporter):
    ctx = RunContext(session_id="s", consent=False)
    dirty = {
        "note": "Patient Jane Doe, MRN 12345, email jane@example.com",
        "question": "What is her risk?",
        "api_key": "voc-TESTFAKESECRETVALUE99",
        "authorization": "Bearer FAKESECRET_not_a_github_token",
        "case.id": ctx.run_id,
        "comment": "Contact jane@example.com or 555-123-4567",
    }
    cleaned = safe_attributes(dirty)
    assert "note" not in cleaned and "question" not in cleaned
    assert "api_key" not in cleaned and "authorization" not in cleaned
    assert "jane@example.com" not in json.dumps(cleaned)
    assert "voc-TESTFAKESECRETVALUE99" not in json.dumps(cleaned)
    with node_span("planner", ctx) as span:
        from core.observability import safe_set
        safe_set(span, dirty)
    ctx.record_tool_span("intake", "screen_note", time.time(), time.time(), "allow",
                         args={"note": "Jane Doe jane@example.com voc-TESTFAKESECRETVALUE99", "query": "age"},
                         ok=True)
    blob = _blob(exporter.get_finished_spans())
    assert "jane@example.com" not in blob.lower()
    assert "voc-TESTFAKESECRETVALUE99" not in blob
    assert "FAKESECRET_not_a_github_token" not in blob
    assert "Patient Jane Doe" not in blob




class _RecordingExporter:
    """In-process exporter used to assert dual attach + redaction with no network."""

    def __init__(self):
        self.batches = []

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult
        self.batches.append([{"name": s.name, "attrs": dict(s.attributes or {})} for s in spans])
        return SpanExportResult.SUCCESS

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis=0):
        return True


def test_arize_is_off_by_default():
    reset_tracing()
    assert arize_requested() is False
    status = arize_status()
    assert status["enabled"] is False and status["project"] == "neuroguard"
    assert status["ui"] == "https://app.arize.com/"
    assert "off" in status["reason"].lower() or "unset" in status["reason"].lower()


def test_arize_requested_needs_both_vars(monkeypatch):
    reset_tracing()
    monkeypatch.delenv("ARIZE_SPACE_ID", raising=False)
    monkeypatch.delenv("ARIZE_API_KEY", raising=False)
    assert arize_requested() is False
    monkeypatch.setenv("ARIZE_SPACE_ID", "PASTE_YOUR_SPACE_ID_HERE")
    assert arize_requested() is False
    monkeypatch.setenv("ARIZE_API_KEY", "PASTE_YOUR_ARIZE_API_KEY_HERE")
    assert arize_requested() is True
    monkeypatch.delenv("ARIZE_SPACE_ID")
    assert arize_requested() is False


def test_both_exporters_attach_and_redact():
    """Phoenix and Arize share one TracerProvider; both see redacted attributes, never PHI or keys."""
    reset_tracing()
    phoenix_exp = _RecordingExporter()
    arize_exp = _RecordingExporter()
    provider = TracerProvider()
    attach_exporter(provider, phoenix_exp, "phoenix")
    attach_exporter(provider, arize_exp, "arize")
    configure(provider)
    dirty = {
        "note": "Patient Jane Doe, MRN 12345, email jane@example.com",
        "api_key": "voc-TESTFAKESECRETVALUE99",
        "comment": "Contact jane@example.com or 555-123-4567",
        "case.id": "case-dual",
    }
    with start_span("neuroguard.run", dirty):
        pass
    force_flush()
    assert phoenix_exp.batches and arize_exp.batches
    assert phoenix_exp.batches == arize_exp.batches
    blob = json.dumps(phoenix_exp.batches)
    assert "jane@example.com" not in blob.lower()
    assert "voc-TESTFAKESECRETVALUE99" not in blob
    assert "Patient Jane Doe" not in blob
    assert any(item["attrs"].get("case.id") == "case-dual" for batch in phoenix_exp.batches for item in batch)
    status = export_status()
    assert status["phoenix"]["ok"] is True and status["phoenix"]["result"] == "SUCCESS"
    assert status["arize"]["ok"] is True and status["arize"]["result"] == "SUCCESS"
    assert status["phoenix"]["spans"] >= 1 and status["arize"]["spans"] >= 1
    reset_tracing()


def test_init_tracing_attaches_phoenix_and_arize(monkeypatch):
    """init_tracing registers both destinations on one provider (mocked exporters, no network)."""
    reset_tracing()
    monkeypatch.setenv("NG_PHOENIX", "1")
    monkeypatch.setenv("ARIZE_SPACE_ID", "PASTE_YOUR_SPACE_ID_HERE")
    monkeypatch.setenv("ARIZE_API_KEY", "PASTE_YOUR_ARIZE_API_KEY_HERE")
    phoenix_exp = _RecordingExporter()
    arize_exp = _RecordingExporter()

    def fake_phoenix():
        provider = TracerProvider()
        attach_exporter(provider, phoenix_exp, "phoenix")
        return provider

    monkeypatch.setattr("core.observability._register_phoenix", fake_phoenix)
    monkeypatch.setattr("core.observability._make_arize_exporter", lambda space_id, api_key: arize_exp)
    provider = init_tracing()
    assert provider is not None
    ph = phoenix_status()
    ax = arize_status()
    assert ph["enabled"] is True and ax["enabled"] is True
    assert ax["ui"] == "https://app.arize.com/"
    with start_span("node.planner", {"case.id": "init-dual", "note": "jane@example.com"}):
        pass
    force_flush()
    assert phoenix_exp.batches and arize_exp.batches
    blob = json.dumps(phoenix_exp.batches + arize_exp.batches)
    assert "jane@example.com" not in blob.lower()
    assert export_status()["phoenix"]["ok"] is True
    assert export_status()["arize"]["ok"] is True
    reset_tracing()


def test_monitoring_aggregate(runtime_dir):
    assert monitoring.aggregate()["n"] == 0
    monitoring.record_run({
        "run_id": "aaa", "status": "finalized", "rag_mode": "agentic", "latency_s": 10.0,
        "tokens": 1000, "llm_calls": 8, "tool_calls": 4, "cost": {"cost_usd": 0.002},
        "final": {"verification": {"checks": {"a": True, "b": True, "c": False}, "failed": ["c"]},
                  "confidence": {"score": 0.8, "gate": "pass"}},
        "contract_sha256": "ab" * 32, "decisions": [{}, {}], "spans": [{}],
    })
    monitoring.record_run({
        "run_id": "bbb", "status": "blocked", "rag_mode": "agentic", "latency_s": 0.2,
        "tokens": 0, "llm_calls": 0, "tool_calls": 0, "cost": {"cost_usd": 0.0},
        "final": {"verification": {"checks": {}, "failed": []}, "confidence": {}},
        "receipt": {"cost_usd": 0.0, "counts": {"decisions": 1, "spans": 0}},
    })
    data = monitoring.aggregate()
    assert data["n"] == 2
    assert data["decision_mix"]["finalized"] == 1 and data["decision_mix"]["blocked"] == 1
    assert data["failed_checks"] == {"c": 1}
    assert data["summary"]["latency_s_max"] == 10.0
    assert data["summary"]["cost_usd_sum"] == 0.002
    assert len(data["series"]["latency_s"]) == 2
    ids = {r["run_id"] for r in data["runs"]}
    assert ids == {"aaa", "bbb"}
    row = next(r for r in data["runs"] if r["run_id"] == "aaa")
    assert row["checks_passed"] == 2 and row["checks_total"] == 3
    assert abs(row["check_pass_rate"] - 2 / 3) < 1e-9
    assert "note" not in json.dumps(data)
