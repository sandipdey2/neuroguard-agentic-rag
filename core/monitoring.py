"""De-identified run-metrics store used by the Streamlit Monitoring tab.

Every finished (or blocked / timed-out) case writes a row: latency, cost, tokens, LLM and tool call counts,
check pass rate, decision, confidence and failed-check names. Nothing here is PHI, a patient note, a question
or an API key. The store is a local SQLite file (or PostgreSQL when DATABASE_URL is set) so the dashboard
works on Streamlit Community Cloud without Arize Phoenix. Cloud disks are ephemeral, so the history lasts
for the life of the container.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

from .config import RUNTIME
from .db import connect

SCHEMA = """CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  created_at REAL NOT NULL,
  status TEXT,
  rag_mode TEXT,
  latency_s REAL,
  cost_usd REAL,
  tokens INTEGER,
  llm_calls INTEGER,
  tool_calls INTEGER,
  checks_passed INTEGER,
  checks_total INTEGER,
  failed_checks TEXT,
  confidence REAL,
  confidence_gate TEXT,
  decision_count INTEGER,
  span_count INTEGER,
  contract_sha256 TEXT,
  phoenix_on INTEGER
)"""


def db_path() -> Path:
    return Path(os.environ.get("NG_MONITOR_DB", RUNTIME / "neuroguard_monitor.sqlite"))


def _conn():
    c = connect(db_path())
    c.execute(SCHEMA)
    return c


def record_run(result: dict, phoenix: dict | None = None) -> int:
    """Persist one run's metrics. Safe to call with a blocked or partial result."""
    receipt = result.get("receipt") or {}
    final = result.get("final") or {}
    verification = final.get("verification") or {}
    checks = verification.get("checks") or receipt.get("checks") or {}
    failed = list(verification.get("failed") or receipt.get("failed_checks") or [])
    failed += [f"NOT_RUN:{c}" for c in (verification.get("unrun_checks") or receipt.get("unrun_checks") or [])]
    cost = result.get("cost") or {}
    cost_usd = receipt.get("cost_usd")
    if cost_usd is None:
        cost_usd = cost.get("cost_usd", 0.0)
    conf = result.get("confidence") or final.get("confidence") or {}
    counts = receipt.get("counts") or {}
    phoenix = phoenix or {}
    created = time.time()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO runs(run_id, created_at, status, rag_mode, latency_s, cost_usd, tokens, llm_calls, tool_calls, "
            "checks_passed, checks_total, failed_checks, confidence, confidence_gate, decision_count, span_count, "
            "contract_sha256, phoenix_on) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                result.get("run_id") or receipt.get("run_id") or "",
                created,
                result.get("status") or receipt.get("status"),
                result.get("rag_mode") or receipt.get("rag_mode"),
                result.get("latency_s") if result.get("latency_s") is not None else receipt.get("latency_s"),
                float(cost_usd or 0),
                int(result.get("tokens") or counts.get("tokens") or 0),
                int(result.get("llm_calls") or counts.get("llm_calls") or 0),
                int(result.get("tool_calls") or counts.get("tool_calls") or 0),
                int(sum(1 for v in checks.values() if v)) if checks else 0,
                int(len(checks)),
                json.dumps(failed),
                (conf.get("score") if isinstance(conf, dict) else conf) or receipt.get("confidence"),
                (conf.get("gate") if isinstance(conf, dict) else None) or receipt.get("confidence_gate"),
                int(counts.get("decisions") or len(result.get("decisions") or [])),
                int(counts.get("spans") or len(result.get("spans") or [])),
                result.get("contract_sha256") or receipt.get("contract_sha256"),
                1 if phoenix.get("enabled") else 0,
            ),
        )
        return int(cur.lastrowid)


def list_runs(limit: int = 200) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT run_id, created_at, status, rag_mode, latency_s, cost_usd, tokens, llm_calls, tool_calls, "
            "checks_passed, checks_total, failed_checks, confidence, confidence_gate, decision_count, span_count, "
            "contract_sha256, phoenix_on FROM runs ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    out = []
    for r in rows:
        total = int(r[10] or 0)
        passed = int(r[9] or 0)
        out.append({
            "run_id": r[0],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[1])),
            "created_ts": r[1],
            "status": r[2],
            "rag_mode": r[3],
            "latency_s": r[4],
            "cost_usd": r[5],
            "tokens": r[6],
            "llm_calls": r[7],
            "tool_calls": r[8],
            "checks_passed": passed,
            "checks_total": total,
            "check_pass_rate": (passed / total) if total else None,
            "failed_checks": json.loads(r[11] or "[]"),
            "confidence": r[12],
            "confidence_gate": r[13],
            "decision_count": r[14],
            "span_count": r[15],
            "contract_sha256": (r[16] or "")[:16],
            "phoenix_on": bool(r[17]),
        })
    return out


def aggregate(limit: int = 200) -> dict:
    """Chart-ready summary of the most recent runs (newest first in ``runs``, chronological in ``series``)."""
    runs = list_runs(limit)
    if not runs:
        return {"n": 0, "runs": [], "summary": {}, "decision_mix": {}, "failed_checks": {}, "series": {}}
    chrono = list(reversed(runs))
    lat = [r["latency_s"] for r in runs if r["latency_s"] is not None]
    costs = [r["cost_usd"] for r in runs if r["cost_usd"] is not None]
    tokens = [r["tokens"] for r in runs if r["tokens"] is not None]
    rates = [r["check_pass_rate"] for r in runs if r["check_pass_rate"] is not None]
    confs = [r["confidence"] for r in runs if r["confidence"] is not None]
    failed: Counter = Counter()
    for r in runs:
        failed.update(r["failed_checks"] or [])
    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    return {
        "n": len(runs),
        "runs": runs,
        "summary": {
            "latency_s_mean": _mean([round(x, 4) for x in lat]) if lat else None,
            "latency_s_max": max(lat) if lat else None,
            "cost_usd_sum": round(sum(costs), 6) if costs else 0.0,
            "cost_usd_mean": _mean(costs),
            "tokens_mean": _mean(tokens),
            "llm_calls_mean": _mean([r["llm_calls"] or 0 for r in runs]),
            "tool_calls_mean": _mean([r["tool_calls"] or 0 for r in runs]),
            "check_pass_rate_mean": _mean(rates),
            "confidence_mean": _mean(confs),
        },
        "decision_mix": dict(Counter(r["status"] or "unknown" for r in runs)),
        "failed_checks": dict(failed),
        "series": {
            "created_at": [r["created_at"] for r in chrono],
            "latency_s": [r["latency_s"] for r in chrono],
            "cost_usd": [r["cost_usd"] for r in chrono],
            "tokens": [r["tokens"] for r in chrono],
            "llm_calls": [r["llm_calls"] for r in chrono],
            "tool_calls": [r["tool_calls"] for r in chrono],
            "check_pass_rate": [r["check_pass_rate"] for r in chrono],
            "confidence": [r["confidence"] for r in chrono],
        },
    }


def clear_runs() -> int:
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        c.execute("DELETE FROM runs")
        return int(n)
