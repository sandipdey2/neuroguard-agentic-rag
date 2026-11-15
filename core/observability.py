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
SPAN_KIND_ATTR = "openinference.span.kind"
_KEEP_KEYS = {SPAN_KIND_ATTR, "tool.name", "llm.model_name", "llm.request.model_name", "llm.response.model_name"}
_KEEP_PREFIXES = ("llm.token_count.",)
_REDACT_KEEP_KEYS = {"input.value", "output.value"}
_REDACT_VALUE_LIMIT = 2048  # input.value / output.value cap (a few KB, not unbounded)
_OTHER_ATTR_LIMIT = 1024
_MAX_EVENTS = 20
_MAX_EXPORT_ATTEMPTS = 4
_RETRY_HTTP = frozenset({429, 500, 502, 503, 504})
_CHAIN_NODES = {"planner", "orchestrator"}
_GUARD_NODES = {"gate"}
_SPECIALIST_NODES = {"intake", "risk_analyst", "evidence_researcher", "clinical_writer", "safety_reviewer"}
_LLM_SPAN_NAMES = {"ChatOpenAI", "ChatCompletion", "OpenAI"}

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
# Arize's recommended OTLP gRPC endpoint (register() default). The gRPC exporter
# uses the URL netloc, i.e. otlp.arize.com:443 with TLS.
DEFAULT_ARIZE_GRPC = "https://otlp.arize.com/v1"

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
    """HTTP traces URL (fallback transport, and status when gRPC is unavailable)."""
    raw = _read_secret("ARIZE_COLLECTOR_ENDPOINT") or DEFAULT_ARIZE_HTTP
    return _http_traces_endpoint(raw)


def arize_grpc_endpoint() -> str:
    """gRPC collector URL. Never appends /v1/traces (that path is HTTP-only)."""
    raw = (_read_secret("ARIZE_COLLECTOR_ENDPOINT") or DEFAULT_ARIZE_GRPC).strip().rstrip("/")
    if raw.endswith("/v1/traces"):
        raw = raw[: -len("/traces")]
    if "://" not in raw:
        raw = "https://" + raw
    return raw


def grpc_available() -> bool:
    """True when the optional gRPC OTLP exporter can be imported (not on Streamlit Cloud)."""
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # noqa: F401
        from grpc import Compression  # noqa: F401
        return True
    except Exception:
        return False


def arize_transport() -> str:
    """Preferred Arize transport: 'grpc' when the optional package is installed, else 'http'."""
    wanted = (_read_secret("ARIZE_OTLP_TRANSPORT") or "grpc").strip().lower()
    if wanted in {"http", "https"}:
        return "http"
    return "grpc" if grpc_available() else "http"


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
    endpoint = None
    if requested:
        endpoint = arize_grpc_endpoint() if arize_transport() == "grpc" else DEFAULT_ARIZE_HTTP
    return {"enabled": bool(enabled), "configured": requested, "endpoint": endpoint,
            "transport": arize_transport() if requested else None,
            "ui": ARIZE_UI, "project": arize_project(), "reason": reason}


def export_status() -> dict:
    """Last OTLP export result per destination (``phoenix`` / ``arize``). No credentials."""
    out = {}
    for k, v in _export_stats.items():
        item = dict(v)
        item["http_statuses"] = dict(v.get("http_statuses") or {})
        out[k] = item
    return out


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


def _record_export(name: str, result, n: int, error: str | None = None, http_status=None) -> None:
    res_name = getattr(result, "name", None) or str(result)
    succeeded = res_name in {"SUCCESS", "SpanExportResult.SUCCESS"} or result == 0
    prev = _export_stats.get(name) or {"spans": 0, "exports": 0, "failures": 0, "http_statuses": {}}
    prev["spans"] = int(prev.get("spans", 0)) + int(n)
    prev["exports"] = int(prev.get("exports", 0)) + 1
    prev["result"] = res_name
    if http_status is not None:
        statuses = prev.setdefault("http_statuses", {})
        key = str(http_status)
        statuses[key] = int(statuses.get(key, 0)) + 1
    if error or not succeeded:
        prev["failures"] = int(prev.get("failures", 0)) + 1
        prev["error"] = error or res_name
        prev["ok"] = False
    else:
        prev["ok"] = int(prev.get("failures", 0)) == 0
        if prev["ok"]:
            prev["error"] = None
    _export_stats[name] = prev


def _hook_exporter_status(exporter) -> None:
    """Remember the last HTTP or gRPC status from an OTLP client (no headers, no bodies)."""
    if getattr(exporter, "_ng_status_hooked", False):
        return
    client = getattr(exporter, "_client", None)
    if client is not None and hasattr(client, "export") and not getattr(client, "_ng_status_hooked", False):
        orig_client = client.export

        def client_export(data):
            result = orig_client(data)
            code = getattr(result, "status_code", None)
            client._ng_last_status = code
            exporter._ng_last_status = code
            return result

        client.export = client_export  # type: ignore[method-assign]
        client._ng_status_hooked = True

    orig = exporter.export

    def export(data):
        try:
            result = orig(data)
            if getattr(exporter, "_ng_last_status", None) is None:
                client = getattr(exporter, "_client", None)
                exporter._ng_last_status = getattr(client, "_ng_last_status", None) if client is not None else None
            return result
        except Exception as exc:
            code = getattr(exc, "code", None)
            if callable(code):
                try:
                    exporter._ng_last_status = str(code())
                except Exception:
                    exporter._ng_last_status = type(exc).__name__
            else:
                exporter._ng_last_status = type(exc).__name__
            raise

    exporter.export = export  # type: ignore[method-assign]
    exporter._ng_status_hooked = True


def _safe_endpoint(exporter) -> str | None:
    raw = str(getattr(exporter, "_endpoint", None) or getattr(exporter, "endpoint", "") or "")
    if not raw:
        return None
    # host + path only; never query string or userinfo
    try:
        from urllib.parse import urlparse
        typ = type(exporter).__name__.upper()
        transport = getattr(exporter, "_ng_transport", "")
        if transport == "grpc" or "GRPC" in typ:
            if "://" not in raw:
                host = raw.split("/")[0]
                if ":" not in host:
                    host = host + ":443"
                return f"https://{host}"
        p = urlparse(raw if "://" in raw else "https://" + raw)
        host = p.hostname or ""
        if not host:
            return raw.split("?")[0]
        port = f":{p.port}" if p.port else ""
        return f"{p.scheme}://{host}{port}{p.path}"
    except Exception:
        return "(unparseable)"


class _RedactingExporter:
    """Wraps an OTLP exporter so attributes are scrubbed even after the SDK freezes them."""

    def __init__(self, wrapped, name: str = "exporter"):
        self._wrapped = wrapped
        self.name = name
        _hook_exporter_status(wrapped)

    def _succeeded(self, result) -> bool:
        res_name = getattr(result, "name", None) or str(result)
        return res_name in {"SUCCESS", "SpanExportResult.SUCCESS"} or result == 0

    def _export_once(self, spans):
        result = self._wrapped.export(spans)
        status = getattr(self._wrapped, "_ng_last_status", None)
        if status is None:
            status = getattr(getattr(self._wrapped, "_client", None), "_ng_last_status", None)
        return result, status

    def _retryable(self, result, status) -> bool:
        if status in _RETRY_HTTP:
            return True
        if isinstance(status, str) and any(tok in status.upper() for tok in ("UNAVAILABLE", "DEADLINE", "RESOURCE_EXHAUSTED", "500")):
            return True
        return False

    def _export_with_retry(self, spans):
        """Retry the same batch with backoff on HTTP 5xx/429 or gRPC UNAVAILABLE. Do not split."""
        delay = 0.5
        result, status = None, None
        for attempt in range(_MAX_EXPORT_ATTEMPTS):
            result, status = self._export_once(spans)
            if self._succeeded(result):
                break
            if attempt >= _MAX_EXPORT_ATTEMPTS - 1 or not self._retryable(result, status):
                break
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
        _record_export(self.name, result, len(spans), http_status=status)
        return result

    def export(self, spans):
        for span in spans or []:
            _redact_span(span)
        n = len(spans) if spans is not None else 0
        if n == 0:
            from opentelemetry.sdk.trace.export import SpanExportResult
            return SpanExportResult.SUCCESS
        try:
            return self._export_with_retry(list(spans))
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


def _iter_processors(provider):
    msp = getattr(provider, "_active_span_processor", None)
    return list(getattr(msp, "_span_processors", ()) or ()) if msp is not None else []


def _processor_exporter(proc):
    exp = getattr(proc, "span_exporter", None)
    if exp is None:
        exp = getattr(getattr(proc, "_batch_processor", None), "_exporter", None)
    return exp


def _install_redacting_exporter(provider) -> None:
    for proc in _iter_processors(provider):
        exp = _processor_exporter(proc)
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
    """Attach ``exporter`` behind the redacting wrapper (tests and dual Phoenix + Arize setup).

    Uses BatchSpanProcessor (max_export_batch_size=64, schedule_delay 1s,
    max_queue_size 4096). Short scripts must still call ``force_flush()``.
    """
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    wrapped = exporter if isinstance(exporter, _RedactingExporter) else _RedactingExporter(exporter, name=name)
    processor = BatchSpanProcessor(
        wrapped,
        max_queue_size=4096,
        schedule_delay_millis=1000,
        max_export_batch_size=64,
        export_timeout_millis=30000,
    )
    _add_span_processor(provider, processor)


def exporter_inventory(provider=None) -> list[dict]:
    """Describe every span processor/exporter on ``provider`` (no credentials)."""
    provider = provider if provider is not None else _provider
    if provider is None:
        return []
    out = []
    default = getattr(provider, "_default_processor", None)
    for i, proc in enumerate(_iter_processors(provider)):
        exp = _processor_exporter(proc)
        inner = getattr(exp, "_wrapped", exp) if exp is not None else None
        out.append({
            "index": i,
            "processor": type(proc).__name__,
            "redacting": isinstance(exp, _RedactingExporter),
            "name": getattr(exp, "name", None) if exp is not None else None,
            "inner": type(inner).__name__ if inner is not None else None,
            "endpoint": _safe_endpoint(inner) if inner is not None else None,
            "transport": getattr(inner, "_ng_transport", None) if inner is not None else None,
            "is_default": proc is default,
        })
    return out


def _make_phoenix_exporter():
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    return OTLPSpanExporter(endpoint=_http_traces_endpoint(collector_endpoint() or DEFAULT_UI))


def _make_arize_http_exporter(space_id: str, api_key: str):
    endpoint = arize_endpoint()
    try:
        from arize.otel import HTTPSpanExporter
        exp = HTTPSpanExporter(endpoint=endpoint, space_id=space_id, api_key=api_key)
    except Exception:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        exp = OTLPSpanExporter(
            endpoint=endpoint,
            headers={
                "authorization": api_key,
                "api_key": api_key,
                "arize-space-id": space_id,
                "space_id": space_id,
            },
        )
    exp._ng_transport = "http"
    return exp


def _make_arize_grpc_exporter(space_id: str, api_key: str):
    """Arize-recommended OTLP gRPC (otlp.arize.com:443, gzip, space_id/api_key metadata)."""
    from grpc import Compression
    endpoint = arize_grpc_endpoint()
    try:
        from arize.otel import GRPCSpanExporter
        exp = GRPCSpanExporter(
            endpoint=endpoint,
            space_id=space_id,
            api_key=api_key,
            compression=Compression.Gzip,
            timeout=30,
        )
    except Exception:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        exp = OTLPSpanExporter(
            endpoint=endpoint,
            headers={
                "authorization": api_key,
                "api_key": api_key,
                "arize-space-id": space_id,
                "space_id": space_id,
                "arize-interface": "otel",
            },
            compression=Compression.Gzip,
            timeout=30,
        )
    exp._ng_transport = "grpc"
    return exp


def _make_arize_exporter(space_id: str, api_key: str):
    """Prefer gRPC when the optional proto-grpc package is installed; otherwise HTTP."""
    if arize_transport() == "grpc":
        try:
            return _make_arize_grpc_exporter(space_id, api_key)
        except Exception:
            pass
    return _make_arize_http_exporter(space_id, api_key)


def _tracing_resource():
    from opentelemetry.sdk.resources import Resource
    project = arize_project()
    return Resource.create({
        "service.name": project,
        "openinference.project.name": project,
        "model_id": project,
        "arize.project.name": project,
        "arize.project.type": "application",
    })


def _set_global_provider(provider) -> None:
    """Set the global provider only if one has not already been installed."""
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import ProxyTracerProvider
        current = otel_trace.get_tracer_provider()
        if current is None or type(current).__name__ == "ProxyTracerProvider" or isinstance(current, ProxyTracerProvider):
            otel_trace.set_tracer_provider(provider)
    except Exception:
        pass


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


def _build_provider(want_ph: bool, want_ax: bool):
    """Build a TracerProvider with only redacting-wrapped OTLP exporters. Does not call register()."""
    from opentelemetry.sdk.trace import TracerProvider
    # Use the SDK TracerProvider (not phoenix.otel.TracerProvider) so no default
    # processor is self-attached and then left unwrapped.
    provider = TracerProvider(resource=_tracing_resource())
    if want_ph:
        attach_exporter(provider, _make_phoenix_exporter(), "phoenix")
    if want_ax:
        creds = arize_credentials()
        if creds:
            space_id, api_key = creds
            attach_exporter(provider, _make_arize_exporter(space_id, api_key), "arize")
    _set_global_provider(provider)
    try:
        _auto_instrument(provider)
    except Exception:
        pass
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
        provider = _build_provider(want_ph, want_ax)
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
    return text[:_OTHER_ATTR_LIMIT]


def _drop_key(key: str) -> bool:
    return any(seg.replace("-", "_") in _DROP_SEGS for seg in str(key).lower().split("."))


def _redact_mapping(attrs) -> None:
    """In-place safe_attributes on a possibly frozen OTel attribute mapping."""
    if attrs is None:
        return
    try:
        if hasattr(attrs, "_immutable"):
            attrs._immutable = False
        redacted = safe_attributes(dict(attrs))
        if hasattr(attrs, "clear"):
            attrs.clear()
            attrs.update(redacted)
        if hasattr(attrs, "_immutable"):
            attrs._immutable = True
    except Exception:
        pass


def _redact_events(span) -> None:
    """Scrub span events (OpenInference often puts prompts here). Attributes are already covered."""
    events = getattr(span, "events", None) or getattr(span, "_events", None) or ()
    ev_list = list(events)
    if len(ev_list) > _MAX_EVENTS:
        ev_list = ev_list[:_MAX_EVENTS]
        if hasattr(span, "_events"):
            try:
                span._events = ev_list
            except Exception:
                pass
    for ev in ev_list:
        ev_attrs = getattr(ev, "_attributes", None)
        if ev_attrs is None:
            ev_attrs = getattr(ev, "attributes", None)
        _redact_mapping(ev_attrs)


def _redact_span(span) -> None:
    attrs = getattr(span, "_attributes", None)
    if attrs is not None:
        try:
            if hasattr(attrs, "_immutable"):
                attrs._immutable = False
            redacted = safe_attributes(dict(attrs))
            redacted[SPAN_KIND_ATTR] = infer_span_kind(getattr(span, "name", "") or "", redacted.get(SPAN_KIND_ATTR))
            attrs.clear()
            attrs.update(redacted)
            if hasattr(attrs, "_immutable"):
                attrs._immutable = True
        except Exception:
            pass
    _redact_events(span)
    status = getattr(span, "_status", None) or getattr(span, "status", None)
    desc = getattr(status, "description", None) if status is not None else None
    if desc:
        try:
            from opentelemetry.trace import Status, StatusCode
            code = getattr(status, "status_code", None) or getattr(status, "code", None)
            if hasattr(span, "_status"):
                span._status = Status(code or StatusCode.UNSET, _redact_text(desc)[:200])
        except Exception:
            pass


def infer_span_kind(name: str, existing: str | None = None) -> str:
    """OpenInference span kind. Never returns UNKNOWN so Arize AX does not show 'kind: unknown'."""
    if existing and str(existing).strip() and str(existing).strip().upper() != "UNKNOWN":
        return str(existing).strip().upper()
    n = name or ""
    bare = n[5:] if n.startswith("node.") else n
    if n == "neuroguard.run":
        return "AGENT"
    if n == "verify" or n.startswith("verify."):
        return "EVALUATOR"
    if n.startswith("mcp.tools/call.") or n.startswith("mcp."):
        return "TOOL"
    if bare in _CHAIN_NODES:
        return "CHAIN"
    if bare in _GUARD_NODES:
        return "GUARDRAIL"
    if n.startswith("node.") or bare in _SPECIALIST_NODES:
        return "AGENT"
    if n in _LLM_SPAN_NAMES or n.startswith("Chat") or "ChatCompletion" in n:
        return "LLM"
    if n.startswith("Runnable") or n in {"LangGraph", "Unnamed", "_check_refusal", "RunnableSequence",
                                          "RunnableLambda", "RunnableWithFallbacks"} or n.startswith("Runnable"):
        return "CHAIN"
    # LangChain often emits a span named after the tool itself
    if bare in {"predict_risk", "explain_factors", "cohort_stats", "screen_note", "search_guidelines",
                "episode_recall", "episode_write", "memory_read", "memory_write", "graph_traverse",
                "semantic_search", "search_files", "load_skill", "execute_query", "web_search",
                "web_fetch", "get_memory", "shell", "pubmed_search", "calculator"}:
        return "TOOL"
    return "CHAIN"


def _with_kind(name: str, attrs: dict | None) -> dict:
    out = dict(attrs or {})
    out[SPAN_KIND_ATTR] = infer_span_kind(name, out.get(SPAN_KIND_ATTR))
    return out


def _keep_key(key: str) -> bool:
    return key in _KEEP_KEYS or key.startswith(_KEEP_PREFIXES)


def safe_attributes(attrs: dict | None) -> dict:
    """Drop or redact anything that could be PHI, an API key or a raw note. Safe to call with no OTel installed."""
    out = {}
    for key, value in (attrs or {}).items():
        k = str(key)
        if _keep_key(k):
            if isinstance(value, (int, float, bool)) or value is None:
                out[k] = value
            else:
                out[k] = str(value)[:200]
            continue
        if k in _REDACT_KEEP_KEYS:
            if isinstance(value, dict):
                value = json.dumps(value, default=str)
            out[k] = _redact_text(value)[:_REDACT_VALUE_LIMIT]
            continue
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
                value = json.dumps(value, default=str)[:_OTHER_ATTR_LIMIT]
            out[k] = _redact_text(value)[:_OTHER_ATTR_LIMIT]
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
    attrs = safe_attributes(_with_kind(name, attributes or {}))
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
    span = tracer.start_span(name, start_time=int(start * 1e9), attributes=safe_attributes(_with_kind(name, attributes or {})))
    if ok:
        span.set_status(Status(StatusCode.OK))
    else:
        span.set_status(Status(StatusCode.ERROR, _redact_text(error or "error")[:200]))
    span.end(end_time=int(end * 1e9))


def emit_tool_span(ctx, agent: str, tool: str, start: float, end: float, decision: str,
                   args: dict | None = None, ok: bool = True, error: str | None = None) -> None:
    attrs = ctx_attributes(ctx, agent=agent, tool=tool, hook=decision, ok=ok,
                           latency_ms=int((end - start) * 1000))
    attrs["tool.name"] = tool
    if args:
        payload = json.dumps(args, default=str)[:_REDACT_VALUE_LIMIT]
        attrs["tool.args"] = payload
        attrs["input.value"] = payload
    summary = {"ok": bool(ok), "hook": decision}
    if error:
        attrs["error"] = error
        summary["error"] = error
    attrs["output.value"] = json.dumps(summary, default=str)[:_REDACT_VALUE_LIMIT]
    emit_span(f"mcp.tools/call.{tool}", start, end, attrs, ok=ok and decision == "allow", error=error)


def emit_verify_span(ctx, result: dict, start: float, end: float | None = None, stage: str = "review") -> None:
    checks = result.get("checks") or {}
    failed = result.get("failed") or []
    stage = stage or "review"
    required = (result.get("details") or {}).get("require_red_team", stage != "precheck")
    # Official review/gate spans stay named "verify". The writer pre-check is
    # "verify.precheck" so Arize does not present it as a failed first 12-check.
    name = "verify.precheck" if stage == "precheck" else "verify"
    total = len([c for c in checks if required or c != "red_team_pass"]) if not required else len(checks)
    passed_n = sum(1 for c, v in checks.items() if v and (required or c != "red_team_pass"))
    if not required:
        total = len(checks) - (1 if "red_team_pass" in checks else 0)
        passed_n = sum(1 for c, v in checks.items() if c != "red_team_pass" and v)
    attrs = ctx_attributes(ctx,
                           **{"checks.passed": passed_n,
                              "checks.failed": ",".join(failed),
                              "checks.total": total,
                              "verify.passed": bool(result.get("passed")),
                              "verify.stage": stage,
                              "decision": "pass" if result.get("passed") else "fail"})
    emit_span(name, start, end or time.time(), attrs, ok=True)


def force_flush(timeout_millis: int = 30000) -> bool:
    init_tracing()
    if _provider is None:
        return False
    try:
        return bool(_provider.force_flush(timeout_millis))
    except Exception:
        return False
