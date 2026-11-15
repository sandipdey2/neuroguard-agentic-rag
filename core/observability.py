"""Eval & observability: span recording, OTLP/JSON export, and optional Phoenix / Arize AX tracing.

``RunContext`` records a span for every LLM call (agent, model, input/output tokens, cost, finish reason) and
every MCP tool call (tool, hook decision, latency). ``otlp_json`` converts them to the OTLP/JSON trace layout
so the file can be sent to any OpenTelemetry backend. ``metrics`` gives per-agent latency, token and cost totals.

When ``PHOENIX_COLLECTOR_ENDPOINT`` or ``NG_PHOENIX=1`` is set, the same run exports live OpenTelemetry
spans (plus OpenInference LangChain / OpenAI instrumentation, if those packages are installed) to Arize Phoenix,
project name ``neuroguard``. When ``ARIZE_SPACE_ID`` and ``ARIZE_API_KEY`` are set (environment or Streamlit
secrets), the same redacted spans are also exported to Arize AX (``arize.otel`` / ``https://otlp.arize.com/v1``).
Phoenix and Arize can be on at the same time (two span processors on one TracerProvider, both behind the
redacting exporter). Otherwise tracing is a no-op: Streamlit Cloud and the pytest suite do not need a collector.
Span attributes never carry PHI, API keys or raw patient notes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

PROJECT_NAME = "neuroguard"
DEFAULT_UI = "http://localhost:6006"
_SECRET_KEY = re.compile(r"(api[_-]?key|authorization|password|secret|passwd|credential|bearer|(^|[._-])token($|[._-]))", re.I)
_DROP_SEGS = {"note", "question", "report", "report_md", "prompt", "content", "input", "output",
              "input_value", "output_value", "messages", "body", "text", "authorization",
              "patient_note", "free_text", "input_messages", "output_messages", "prompt_template",
              "completion", "invocation_parameters"}
_GH_TOKEN = re.compile(r"\b(ghp|gho|github_pat)_[A-Za-z0-9_]{8,}")

_provider = None
_inited = False
_init_error = None


# --------------------------------------------------------------------------- OTLP/JSON (always available, no collector)

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



# --------------------------------------------------------------------------- optional Phoenix / Arize AX / OpenTelemetry

ARIZE_UI = "https://app.arize.com/"
DEFAULT_ARIZE_HTTP = "https://otlp.arize.com/v1/traces"

_export_stats: dict[str, dict] = {}


def _read_secret(name: str) -> str:
    """Environment or Streamlit ``st.secrets``. Never logs the value."""
    try:
        from .config import secret
        val = secret(name)
        if val:
            return str(val).strip()
    except Exception:
        pass
    return (os.environ.get(name) or "").strip()


def phoenix_requested() -> bool:
    if os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or os.environ.get("NG_PHOENIX") == "1":
        return True
    return bool(_read_secret("PHOENIX_COLLECTOR_ENDPOINT") or _read_secret("NG_PHOENIX") == "1")


def collector_endpoint() -> str | None:
    if os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
        return os.environ["PHOENIX_COLLECTOR_ENDPOINT"].rstrip("/")
    secret_ep = _read_secret("PHOENIX_COLLECTOR_ENDPOINT")
    if secret_ep:
        return secret_ep.rstrip("/")
    if os.environ.get("NG_PHOENIX") == "1" or _read_secret("NG_PHOENIX") == "1":
        return DEFAULT_UI
    return None


def ui_url(endpoint: str | None = None) -> str:
    raw = (endpoint or collector_endpoint() or DEFAULT_UI).rstrip("/")
    if raw.endswith("/v1/traces"):
        raw = raw[: -len("/v1/traces")]
    return raw or DEFAULT_UI


def arize_credentials() -> tuple[str, str] | None:
    space = _read_secret("ARIZE_SPACE_ID")
    key = _read_secret("ARIZE_API_KEY")
    if space and key:
        return space, key
    return None


def arize_requested() -> bool:
    return arize_credentials() is not None


def arize_endpoint() -> str:
    raw = _read_secret("ARIZE_COLLECTOR_ENDPOINT") or DEFAULT_ARIZE_HTTP
    return _http_traces_endpoint(raw)


def arize_project() -> str:
    return _read_secret("ARIZE_PROJECT_NAME") or PROJECT_NAME


def phoenix_status() -> dict:
    requested = phoenix_requested()
    endpoint = collector_endpoint()
    enabled = requested and _provider is not None
    if not requested:
        reason = "PHOENIX_COLLECTOR_ENDPOINT is unset and NG_PHOENIX is not 1; tracing is off (the Streamlit Cloud default)"
    elif _provider is None and _inited:
        reason = "Phoenix / OpenTelemetry packages failed to initialise; tracing is off" + (f" ({_init_error})" if _init_error else "")
    elif not _inited:
        reason = "tracing will start on the first instrumented run"
    else:
        reason = f"exporting OpenTelemetry spans to {endpoint} (project {PROJECT_NAME})"
    return {"enabled": bool(enabled), "configured": requested, "endpoint": endpoint,
            "ui": ui_url(endpoint), "project": PROJECT_NAME, "reason": reason}


def arize_status() -> dict:
    requested = arize_requested()
    enabled = requested and _provider is not None
    if not requested:
        reason = "ARIZE_SPACE_ID or ARIZE_API_KEY is unset; Arize AX export is off (the Streamlit Cloud default until secrets are added)"
    elif _provider is None and _inited:
        reason = "Arize / OpenTelemetry packages failed to initialise; export is off" + (f" ({_init_error})" if _init_error else "")
    elif not _inited:
        reason = "tracing will start on the first instrumented run"
    else:
        reason = f"exporting OpenTelemetry spans to Arize AX (project {arize_project()})"
    return {"enabled": bool(enabled), "configured": requested, "endpoint": DEFAULT_ARIZE_HTTP if requested else None,
            "ui": ARIZE_UI, "project": arize_project(), "reason": reason}


def export_status() -> dict:
    """Last OTLP export result per destination (``phoenix`` / ``arize``). No credentials."""
    return {k: dict(v) for k, v in _export_stats.items()}


def reset_tracing() -> None:
    """Drop the tracer provider (tests). Does not try to shut down a live exporter."""
    global _provider, _inited, _init_error, _export_stats
    _provider = None
    _inited = False
    _init_error = None
    _export_stats = {}


def configure(provider) -> None:
    """Tests inject an in-memory OpenTelemetry TracerProvider. Does not talk to a collector."""
    global _provider, _inited
    _provider = provider
    _inited = True


def _http_traces_endpoint(base: str) -> str:
    base = base.rstrip("/")
    return base if base.endswith("/v1/traces") else base + "/v1/traces"


def _scrub_error(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}"[:400]
    secrets: list[str] = []
    for name, val in list(os.environ.items()):
        upper = name.upper()
        if val and any(tok in upper for tok in ("API_KEY", "SECRET", "TOKEN", "PASSWORD", "SPACE_ID", "AUTHORIZATION")):
            secrets.append(val)
    for name in ("ARIZE_API_KEY", "ARIZE_SPACE_ID", "OPENAI_API_KEY", "PHOENIX_API_KEY"):
        val = _read_secret(name)
        if val:
            secrets.append(val)
    for val in sorted(set(secrets), key=len, reverse=True):
        if len(val) >= 8:
            msg = msg.replace(val, "[REDACTED]")
    return msg


def _record_export(name: str, result, n: int, error: str | None = None) -> None:
    res_name = getattr(result, "name", None) or str(result)
    ok = res_name in {"SUCCESS", "SpanExportResult.SUCCESS"} or result == 0
    prev = _export_stats.get(name) or {"spans": 0, "exports": 0, "failures": 0}
    prev["spans"] = int(prev.get("spans", 0)) + int(n)
    prev["exports"] = int(prev.get("exports", 0)) + 1
    prev["result"] = res_name
    if error:
        prev["failures"] = int(prev.get("failures", 0)) + 1
        prev["error"] = error
        prev["ok"] = False
    else:
        prev["error"] = None
        prev["ok"] = bool(ok)
        prev["failures"] = int(prev.get("failures", 0))
    _export_stats[name] = prev


class _RedactingExporter:
    """Wraps an OTLP exporter so attributes are scrubbed even after the SDK freezes them."""

    def __init__(self, wrapped, name: str = "exporter"):
        self._wrapped = wrapped
        self.name = name

    def export(self, spans):
        for span in spans:
            _redact_span(span)
        n = len(spans) if spans is not None else 0
        try:
            result = self._wrapped.export(spans)
            _record_export(self.name, result, n)
            return result
        except Exception as exc:
            _record_export(self.name, "ERROR", n, error=type(exc).__name__)
            raise

    def shutdown(self):
        return self._wrapped.shutdown()

    def force_flush(self, timeout_millis: int = 30000):
        flush = getattr(self._wrapped, "force_flush", None)
        return flush(timeout_millis) if flush else True


def _guess_exporter_name(exp) -> str:
    if isinstance(exp, _RedactingExporter):
        return exp.name
    endpoint = str(getattr(exp, "_endpoint", None) or getattr(exp, "endpoint", "") or "").lower()
    if "arize.com" in endpoint:
        return "arize"
    return "phoenix"


def _install_redacting_exporter(provider) -> None:
    msp = getattr(provider, "_active_span_processor", None)
    processors = getattr(msp, "_span_processors", ()) if msp is not None else ()
    for proc in processors:
        exp = getattr(proc, "span_exporter", None)
        if exp is None:
            exp = getattr(getattr(proc, "_batch_processor", None), "_exporter", None)
        if exp is None or isinstance(exp, _RedactingExporter):
            continue
        wrapped = _RedactingExporter(exp, name=_guess_exporter_name(exp))
        if hasattr(proc, "span_exporter"):
            proc.span_exporter = wrapped
        bp = getattr(proc, "_batch_processor", None)
        if bp is not None and hasattr(bp, "_exporter"):
            bp._exporter = wrapped


def _add_span_processor(provider, processor) -> None:
    try:
        provider.add_span_processor(processor, replace_default_processor=False)
    except TypeError:
        provider.add_span_processor(processor)


def attach_exporter(provider, exporter, name: str) -> None:
    """Attach ``exporter`` behind the redacting wrapper (tests and dual Phoenix + Arize setup)."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    wrapped = exporter if isinstance(exporter, _RedactingExporter) else _RedactingExporter(exporter, name=name)
    _add_span_processor(provider, SimpleSpanProcessor(wrapped))


def _make_arize_exporter(space_id: str, api_key: str):
    endpoint = arize_endpoint()
    try:
        from arize.otel import HTTPSpanExporter
        return HTTPSpanExporter(endpoint=endpoint, space_id=space_id, api_key=api_key)
    except Exception:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        return OTLPSpanExporter(
            endpoint=endpoint,
            headers={
                "authorization": api_key,
                "api_key": api_key,
                "arize-space-id": space_id,
                "space_id": space_id,
            },
        )


def _register_phoenix():
    from phoenix.otel import register
    endpoint = _http_traces_endpoint(collector_endpoint() or DEFAULT_UI)
    return register(project_name=PROJECT_NAME, endpoint=endpoint, protocol="http/protobuf",
                    auto_instrument=True, batch=False, verbose=False, set_global_tracer_provider=True)


def _register_arize_provider(space_id: str, api_key: str):
    from arize.otel import register, Transport
    return register(
        space_id=space_id,
        api_key=api_key,
        project_name=arize_project(),
        endpoint=arize_endpoint(),
        transport=Transport.HTTP,
        batch=False,
        verbose=False,
        set_global_tracer_provider=True,
    )


def _auto_instrument(provider) -> None:
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument(tracer_provider=provider)
    except Exception:
        pass
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument(tracer_provider=provider)
    except Exception:
        pass


def _fallback_arize_provider(space_id: str, api_key: str):
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    resource = Resource.create({
        "openinference.project.name": arize_project(),
        "model_id": arize_project(),
        "service.name": arize_project(),
    })
    provider = TracerProvider(resource=resource)
    otel_trace.set_tracer_provider(provider)
    attach_exporter(provider, _make_arize_exporter(space_id, api_key), "arize")
    return provider


def init_tracing():
    """Idempotent. Registers Phoenix and/or Arize AX only when configured; otherwise a no-op."""
    global _provider, _inited, _init_error
    if _inited:
        return _provider
    _inited = True
    want_ph = phoenix_requested()
    want_ax = arize_requested()
    if not want_ph and not want_ax:
        _provider = None
        return None
    try:
        provider = None
        if want_ph:
            provider = _register_phoenix()
        elif want_ax:
            space_id, api_key = arize_credentials()  # type: ignore[misc]
            try:
                provider = _register_arize_provider(space_id, api_key)
            except Exception:
                provider = _fallback_arize_provider(space_id, api_key)
            try:
                _auto_instrument(provider)
            except Exception:
                pass
        try:
            _install_redacting_exporter(provider)
        except Exception:
            pass
        if want_ph and want_ax:
            space_id, api_key = arize_credentials()  # type: ignore[misc]
            attach_exporter(provider, _make_arize_exporter(space_id, api_key), "arize")
        _provider = provider
        _init_error = None
        return provider
    except Exception as exc:
        _provider = None
        _init_error = _scrub_error(exc)
        return None


def get_tracer():
    init_tracing()
    if _provider is None:
        return None
    return _provider.get_tracer("neuroguard.harness")


def _redact_text(value: str) -> str:
    from .guardrails import redact
    text, _ = redact(str(value))
    text = _GH_TOKEN.sub("[REDACTED-TOKEN]", text)
    return text[:500]


def _drop_key(key: str) -> bool:
    return any(seg.replace("-", "_") in _DROP_SEGS for seg in str(key).lower().split("."))


def _redact_span(span) -> None:
    attrs = getattr(span, "_attributes", None)
    if attrs is None:
        return
    try:
        if hasattr(attrs, "_immutable"):
            attrs._immutable = False
        redacted = safe_attributes(dict(attrs))
        attrs.clear()
        attrs.update(redacted)
        if hasattr(attrs, "_immutable"):
            attrs._immutable = True
    except Exception:
        pass


def safe_attributes(attrs: dict | None) -> dict:
    """Drop or redact anything that could be PHI, an API key or a raw note. Safe to call with no OTel installed."""
    out = {}
    for key, value in (attrs or {}).items():
        k = str(key)
        if _SECRET_KEY.search(k) or _drop_key(k):
            continue
        if value is None or isinstance(value, bool):
            out[k] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            out[k] = value
        elif isinstance(value, float):
            out[k] = value
        elif isinstance(value, (list, tuple)):
            items = [_redact_text(v) if not isinstance(v, (int, float, bool)) else v for v in value[:20]]
            out[k] = tuple(items)
        else:
            if isinstance(value, dict):
                value = json.dumps(value, default=str)[:500]
            out[k] = _redact_text(value)
    return out


def ctx_attributes(ctx, **extra) -> dict:
    perms = getattr(ctx, "perms", None)
    return {
        "case.id": getattr(ctx, "run_id", ""),
        "contract.sha256": getattr(ctx, "contract_sha256", ""),
        "run.cost_usd": round(float(getattr(ctx, "cost_usd", 0) or 0), 6),
        "run.tokens": int(getattr(ctx, "tokens", 0) or 0),
        "run.llm_calls": int(getattr(ctx, "llm_calls", 0) or 0),
        "run.tool_calls": int(getattr(perms, "tool_calls", 0) or 0),
        **extra,
    }


def safe_set(span, attrs: dict) -> None:
    if span is None:
        return
    cleaned = safe_attributes(attrs)
    if cleaned:
        span.set_attributes(cleaned)


@contextmanager
def start_span(name: str, attributes: dict | None = None) -> Iterator[Any]:
    tracer = get_tracer()
    attrs = safe_attributes(attributes or {})
    if tracer is None:
        yield None
        return
    from opentelemetry.trace import Status, StatusCode
    with tracer.start_as_current_span(name, attributes=attrs) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, _redact_text(str(exc))[:200]))
            raise


def node_span(name: str, ctx) -> Iterator[Any]:
    return start_span(f"node.{name}", ctx_attributes(ctx, node=name))


def emit_span(name: str, start: float, end: float, attributes: dict | None = None, ok: bool = True, error: str | None = None) -> None:
    tracer = get_tracer()
    if tracer is None:
        return
    from opentelemetry.trace import Status, StatusCode
    span = tracer.start_span(name, start_time=int(start * 1e9), attributes=safe_attributes(attributes or {}))
    if ok:
        span.set_status(Status(StatusCode.OK))
    else:
        span.set_status(Status(StatusCode.ERROR, _redact_text(error or "error")[:200]))
    span.end(end_time=int(end * 1e9))


def emit_tool_span(ctx, agent: str, tool: str, start: float, end: float, decision: str,
                   args: dict | None = None, ok: bool = True, error: str | None = None) -> None:
    attrs = ctx_attributes(ctx, agent=agent, tool=tool, hook=decision, ok=ok,
                           latency_ms=int((end - start) * 1000))
    if args:
        attrs["tool.args"] = json.dumps(args, default=str)[:400]
    if error:
        attrs["error"] = error
    emit_span(f"mcp.tools/call.{tool}", start, end, attrs, ok=ok and decision == "allow", error=error)


def emit_verify_span(ctx, result: dict, start: float, end: float | None = None) -> None:
    checks = result.get("checks") or {}
    failed = result.get("failed") or []
    attrs = ctx_attributes(ctx,
                           **{"checks.passed": sum(1 for v in checks.values() if v),
                              "checks.failed": ",".join(failed),
                              "checks.total": len(checks),
                              "verify.passed": bool(result.get("passed")),
                              "decision": "pass" if result.get("passed") else "fail"})
    emit_span("verify", start, end or time.time(), attrs, ok=True)


def force_flush(timeout_millis: int = 5000) -> bool:
    init_tracing()
    if _provider is None:
        return False
    try:
        return bool(_provider.force_flush(timeout_millis))
    except Exception:
        return False
