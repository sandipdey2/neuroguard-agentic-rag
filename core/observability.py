"""Eval & observability: span recording and an OpenTelemetry-compatible JSON export (no collector needed).

``RunContext`` records a span for every LLM call (agent, model, input/output tokens, cost, finish reason) and
every MCP tool call (tool, hook decision, latency). ``otlp_json`` converts them to the OTLP/JSON trace layout
(resourceSpans -> scopeSpans -> spans with traceId, spanId, parentSpanId, start/end in unix nanoseconds and
attributes), so the file can be sent to any OpenTelemetry backend (Jaeger, Honeycomb, Grafana Tempo) with an
OTLP/HTTP exporter. ``metrics`` gives per-agent latency, token and cost totals for the UI and the evaluation.
Honest scope: export only - this app does not run a collector or ship spans anywhere by itself.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict


def _id(*parts, n=16) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:n]


def otlp_json(run_id: str, spans: list[dict], started: float, ended: float) -> dict:
    trace_id = run_id[:32].ljust(32, "0")
    root = _id(run_id, "root")
    out = [{"traceId": trace_id, "spanId": root, "name": "neuroguard.run", "kind": 1,
            "startTimeUnixNano": int(started * 1e9), "endTimeUnixNano": int(ended * 1e9), "attributes": []}]
    for i, s in enumerate(spans):
        out.append({"traceId": trace_id, "spanId": _id(run_id, i), "parentSpanId": root, "name": s["name"], "kind": 3 if s["kind"] == "client" else 1,
                    "startTimeUnixNano": int(s["start"] * 1e9), "endTimeUnixNano": int(s["end"] * 1e9),
                    "attributes": [{"key": k, "value": {"stringValue": str(v)}} for k, v in s.get("attrs", {}).items()]})
    return {"resourceSpans": [{"resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "neuroguard-ai"}}]},
                               "scopeSpans": [{"scope": {"name": "neuroguard.harness"}, "spans": out}]}]}


def metrics(spans: list[dict]) -> dict:
    agg = defaultdict(lambda: {"llm_calls": 0, "tool_calls": 0, "llm_seconds": 0.0, "tool_seconds": 0.0, "input_tokens": 0,
                               "output_tokens": 0, "cost_usd": 0.0})
    for s in spans:
        a = agg[s["attrs"].get("agent", "?")]
        if s["name"].startswith("llm"):
            a["llm_calls"] += 1
            a["llm_seconds"] += s["end"] - s["start"]
            a["input_tokens"] += int(s["attrs"].get("input_tokens", 0))
            a["output_tokens"] += int(s["attrs"].get("output_tokens", 0))
            a["cost_usd"] += float(s["attrs"].get("cost_usd", 0))
        else:
            a["tool_calls"] += 1
            a["tool_seconds"] += s["end"] - s["start"]
    return {k: {**v, "llm_seconds": round(v["llm_seconds"], 2), "tool_seconds": round(v["tool_seconds"], 3), "cost_usd": round(v["cost_usd"], 6)}
            for k, v in agg.items()}
