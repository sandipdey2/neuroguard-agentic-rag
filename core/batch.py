"""Agentic workflows at scale (PARTIAL): run a cohort of cases concurrently under one shared rate limit.

* ``asyncio.Semaphore(max_concurrency)`` bounds concurrent runs (each run has its own MCP subprocess).
* ``TokenBucket`` (llm_calls_per_minute) is awaited before EVERY LLM call of every run, so the shared
  Vocareum key is never burst past the contract limit.
* Each case still gets its own contract, budget, audit chain, receipt and episode - nothing is shared but the limiter.

Why partial: this is single-process concurrency for a small cohort (contract max_cases = 6 in the app),
not a distributed queue; there is no retry/resume across process restarts.
"""
from __future__ import annotations

import asyncio
import time

from .config import load_contract


class TokenBucket:
    def __init__(self, rate_per_min: float, burst: int | None = None):
        self.rate = rate_per_min / 60.0
        self.capacity = burst or max(1, int(rate_per_min // 6))
        self.tokens = float(self.capacity)
        self.t = time.monotonic()
        self.waited_s = 0.0
        self._lock: asyncio.Lock | None = None

    async def acquire(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                w = (1 - self.tokens) / self.rate
                self.waited_s += w
                await asyncio.sleep(w)


async def arun_batch(cases: list[dict], max_concurrency: int | None = None, rate_per_min: float | None = None,
                     progress=None) -> dict:
    """cases: [{profile, question, note?, audience?}] -> {results, wall_s, sum_latency_s, speedup, ...}."""
    from .pipeline import arun_case
    b = load_contract()["batch"]
    n = min(len(cases), b["max_cases"])
    conc = max(1, min(max_concurrency or b["max_concurrency"], b["max_concurrency"]))
    bucket = TokenBucket(rate_per_min or b["llm_calls_per_minute"])
    sem = asyncio.Semaphore(conc)
    t0 = time.perf_counter()

    async def one(i, c):
        async with sem:
            s = time.perf_counter()
            r = await arun_case(c["profile"], c["question"], c.get("note", ""), c.get("audience", "clinician"),
                                session_id=f"batch-{i}", limiter=bucket)
            if progress:
                try:
                    progress(i, r)
                except Exception:
                    pass
            return {"i": i, "status": r["status"], "latency_s": r["latency_s"], "queued_s": round(s - t0, 2),
                    "llm_calls": r.get("llm_calls"), "cost_usd": (r.get("cost") or {}).get("cost_usd"),
                    "confidence": (r.get("confidence") or {}).get("score"), "risk_band": (r.get("risk") or {}).get("risk_band"),
                    "run_id": r["run_id"], "result": r}

    rows = await asyncio.gather(*(one(i, c) for i, c in enumerate(cases[:n])))
    wall = time.perf_counter() - t0
    tot = sum(r["latency_s"] for r in rows)
    return {"n": n, "max_concurrency": conc, "rate_per_min": bucket.rate * 60, "wall_s": round(wall, 2), "sum_latency_s": round(tot, 2),
            "speedup": round(tot / wall, 2) if wall else None, "limiter_wait_s": round(bucket.waited_s, 2),
            "finalized": sum(r["status"] == "finalized" for r in rows), "rows": rows}


def run_batch(cases: list[dict], max_concurrency: int | None = None, rate_per_min: float | None = None, progress=None,
              timeout: float = 1500) -> dict:
    from .pipeline import _background_loop
    fut = asyncio.run_coroutine_threadsafe(arun_batch(cases, max_concurrency, rate_per_min, progress), _background_loop())
    return fut.result(timeout=timeout)
