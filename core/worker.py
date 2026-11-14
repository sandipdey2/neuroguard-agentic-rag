"""Queue worker: takes case jobs off the durable queue and runs them through the multi-agent pipeline.

Used by the FastAPI service (an in-process worker thread, ``NG_API_INLINE_WORKER=1`` default), by
``python -m core.worker`` (a separate worker container in docker-compose / Kubernetes) and by the scheduled batch CLI.
Stores only the de-identified case summary, the report markdown and the signed receipt as the job result.
"""
from __future__ import annotations

import os
import socket
import threading
import time

from . import a2a
from .chat import case_summary
from .job_queue import get_queue


def job_result(res: dict) -> dict:
    return {"summary": case_summary(res), "report": res.get("report") or "", "receipt": res.get("receipt"),
            "rag_mode": res.get("rag_mode"), "a2a_events": len(res.get("a2a") or [])}


def process_one(queue=None, run=None, worker: str | None = None, topic: str = "cases") -> dict | None:
    q = queue or get_queue()
    job = q.claim(topic, worker or f"{socket.gethostname()}:{os.getpid()}")
    if job is None:
        return None
    if run is None:
        from .pipeline import run_case as run
    p = job["payload"]
    try:
        res = run(p["profile"], p.get("question") or "Explain this patient's stroke risk and what to discuss at the next visit.",
                  p.get("note", ""), p.get("audience", "clinician"), p.get("session_id") or f"api-{job['job_id'][:8]}",
                  bool(p.get("consent", False)), rag_mode=p.get("rag_mode"))
        q.complete(job["job_id"], job_result(res))
        q.publish("case_results", {"job_id": job["job_id"], "status": res.get("status"), "run_id": res.get("run_id")})
    except Exception as exc:              # job is requeued until max_attempts, never lost
        q.fail(job["job_id"], f"{type(exc).__name__}: {exc}")
    return q.get(job["job_id"])


def run_worker(stop: threading.Event | None = None, poll_s: float = 1.0, queue=None, run=None, publish_a2a: bool = True) -> None:
    q = queue or get_queue()
    if publish_a2a:
        a2a.set_sink(q.publish)
    stop = stop or threading.Event()
    while not stop.is_set():
        if process_one(q, run) is None:
            stop.wait(poll_s)


def start_background_worker(queue=None, run=None) -> tuple[threading.Thread, threading.Event]:
    stop = threading.Event()
    t = threading.Thread(target=run_worker, kwargs={"stop": stop, "queue": queue, "run": run}, name="neuroguard-worker", daemon=True)
    t.start()
    return t, stop


if __name__ == "__main__":       # python -m core.worker
    print("NeuroGuard worker: queue backend =", get_queue().backend, flush=True)
    try:
        run_worker()
    except KeyboardInterrupt:
        time.sleep(0)
