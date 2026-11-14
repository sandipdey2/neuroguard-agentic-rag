"""Durable message queue for API / webhook / scheduled jobs and A2A event publishing.

Default backend: SQLite (WAL) - ``jobs`` + ``events`` tables in ``data/runtime/neuroguard_queue.sqlite``; jobs survive
process restarts. With ``DATABASE_URL`` the same tables live in PostgreSQL (claims use ``FOR UPDATE SKIP LOCKED``), and
with ``NG_QUEUE_URL=redis://...`` a Redis list/hash backend is used instead (optional; tests use fakeredis).

Job lifecycle: ``enqueue`` -> queued -> ``claim`` (atomic, sets a lease) -> running -> ``complete`` (done) or ``fail``
(requeued until ``max_attempts``, then failed). A running job whose lease expired (worker crashed) is claimable again,
so a crash never loses a job. ``publish`` appends an event to a topic (the A2A bus publishes every task/send and
task/completed here when a sink is installed - see core/a2a.py::set_sink).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from .config import RUNTIME, load_contract
from .db import backend as db_backend, connect

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT UNIQUE NOT NULL, topic TEXT NOT NULL, payload TEXT NOT NULL,
  status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, lease_until REAL, worker TEXT, result TEXT, error TEXT,
  source TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL)""",
]
COLS = ["job_id", "topic", "payload", "status", "attempts", "lease_until", "worker", "result", "error", "source", "created_at", "updated_at"]


def _cfg() -> dict:
    return {"lease_s": 420, "max_attempts": 2, **load_contract().get("queue", {})}


def _row(r) -> dict:
    d = dict(zip(COLS, r))
    d["payload"] = json.loads(d["payload"])
    d["result"] = json.loads(d["result"]) if d["result"] else None
    return d


class SQLQueue:
    """SQLite by default, PostgreSQL when DATABASE_URL is set (same SQL through core/db.py)."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or os.environ.get("NG_QUEUE_DB", RUNTIME / "neuroguard_queue.sqlite"))
        cfg = _cfg()
        self.lease_s, self.max_attempts = float(cfg["lease_s"]), int(cfg["max_attempts"])
        self.backend = db_backend()
        with self._conn() as c:
            for s in SCHEMA:
                c.execute(s)

    def _conn(self):
        return connect(self.path)

    def enqueue(self, topic: str, payload: dict, source: str = "api", job_id: str | None = None) -> str:
        job_id = job_id or uuid.uuid4().hex
        now = time.time()
        with self._conn() as c:
            c.execute("INSERT INTO jobs(job_id, topic, payload, status, attempts, source, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                      (job_id, topic, json.dumps(payload, default=str), "queued", 0, source, now, now))
        return job_id

    def claim(self, topic: str, worker: str = "worker") -> dict | None:
        now = time.time()
        cond = ("topic=? AND attempts<? AND (status='queued' OR (status='running' AND lease_until<?))")
        with self._conn() as c:
            if self.backend == "sqlite":
                c.execute("BEGIN IMMEDIATE")
                r = c.execute(f"SELECT job_id FROM jobs WHERE {cond} ORDER BY id LIMIT 1", (topic, self.max_attempts, now)).fetchone()
                if not r:
                    return None
                c.execute("UPDATE jobs SET status='running', attempts=attempts+1, lease_until=?, worker=?, updated_at=? WHERE job_id=?",
                          (now + self.lease_s, worker, now, r[0]))
                job_id = r[0]
            else:
                r = c.execute(f"UPDATE jobs SET status='running', attempts=attempts+1, lease_until=?, worker=?, updated_at=? WHERE job_id = "
                              f"(SELECT job_id FROM jobs WHERE {cond} ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING job_id",
                              (now + self.lease_s, worker, now, topic, self.max_attempts, now)).fetchone()
                if not r:
                    return None
                job_id = r[0]
        return self.get(job_id)

    def complete(self, job_id: str, result: dict) -> None:
        with self._conn() as c:
            c.execute("UPDATE jobs SET status='done', result=?, lease_until=NULL, updated_at=? WHERE job_id=?",
                      (json.dumps(result, default=str), time.time(), job_id))

    def fail(self, job_id: str, error: str) -> str:
        job = self.get(job_id)
        status = "queued" if job and job["attempts"] < self.max_attempts else "failed"
        with self._conn() as c:
            c.execute("UPDATE jobs SET status=?, error=?, lease_until=NULL, updated_at=? WHERE job_id=?",
                      (status, str(error)[:500], time.time(), job_id))
        return status

    def get(self, job_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute(f"SELECT {', '.join(COLS)} FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _row(r) if r else None

    def counts(self) -> dict:
        with self._conn() as c:
            return {s: n for s, n in c.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()}

    def publish(self, topic: str, event: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO events(topic, payload, created_at) VALUES (?,?,?)", (topic, json.dumps(event, default=str), time.time()))

    def events(self, topic: str, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT payload, created_at FROM events WHERE topic=? ORDER BY id DESC LIMIT ?", (topic, int(limit))).fetchall()
        return [{**json.loads(p), "_ts": ts} for p, ts in rows][::-1]


class RedisQueue:
    """Optional Redis backend: list per topic for queued ids, hash per job, capped list per event topic."""

    def __init__(self, client):
        self.r = client
        cfg = _cfg()
        self.lease_s, self.max_attempts = float(cfg["lease_s"]), int(cfg["max_attempts"])
        self.backend = "redis"

    def _save(self, job: dict):
        self.r.set(f"ng:job:{job['job_id']}", json.dumps(job, default=str))

    def enqueue(self, topic, payload, source="api", job_id=None):
        job_id = job_id or uuid.uuid4().hex
        now = time.time()
        self._save({"job_id": job_id, "topic": topic, "payload": payload, "status": "queued", "attempts": 0, "lease_until": None,
                    "worker": None, "result": None, "error": None, "source": source, "created_at": now, "updated_at": now})
        self.r.rpush(f"ng:q:{topic}", job_id)
        return job_id

    def claim(self, topic, worker="worker"):
        for jid in [x.decode() if isinstance(x, bytes) else x for x in self.r.lrange(f"ng:running:{topic}", 0, -1)]:
            job = self.get(jid)                 # recover expired leases first
            if job and job["status"] == "running" and (job["lease_until"] or 0) < time.time():
                self.r.lrem(f"ng:running:{topic}", 0, jid)
                if job["attempts"] < self.max_attempts:
                    self.r.rpush(f"ng:q:{topic}", jid)
        jid = self.r.lmove(f"ng:q:{topic}", f"ng:running:{topic}", "LEFT", "RIGHT")
        if not jid:
            return None
        jid = jid.decode() if isinstance(jid, bytes) else jid
        job = self.get(jid)
        job.update(status="running", attempts=job["attempts"] + 1, lease_until=time.time() + self.lease_s, worker=worker, updated_at=time.time())
        self._save(job)
        return job

    def complete(self, job_id, result):
        job = self.get(job_id)
        job.update(status="done", result=result, lease_until=None, updated_at=time.time())
        self._save(job)
        self.r.lrem(f"ng:running:{job['topic']}", 0, job_id)

    def fail(self, job_id, error):
        job = self.get(job_id)
        status = "queued" if job["attempts"] < self.max_attempts else "failed"
        job.update(status=status, error=str(error)[:500], lease_until=None, updated_at=time.time())
        self._save(job)
        self.r.lrem(f"ng:running:{job['topic']}", 0, job_id)
        if status == "queued":
            self.r.rpush(f"ng:q:{job['topic']}", job_id)
        return status

    def get(self, job_id):
        v = self.r.get(f"ng:job:{job_id}")
        return json.loads(v) if v else None

    def counts(self):
        out = {}
        for k in self.r.scan_iter("ng:job:*"):
            s = json.loads(self.r.get(k))["status"]
            out[s] = out.get(s, 0) + 1
        return out

    def publish(self, topic, event):
        self.r.rpush(f"ng:events:{topic}", json.dumps({**event, "_ts": time.time()}, default=str))
        self.r.ltrim(f"ng:events:{topic}", -1000, -1)

    def events(self, topic, limit=50):
        return [json.loads(x) for x in self.r.lrange(f"ng:events:{topic}", -int(limit), -1)]


_queue = None


def get_queue():
    global _queue
    if _queue is None:
        url = os.environ.get("NG_QUEUE_URL", "").strip()
        if url.startswith("redis"):
            import redis
            _queue = RedisQueue(redis.Redis.from_url(url))
        else:
            _queue = SQLQueue()
    return _queue


def set_queue(q) -> None:
    global _queue
    _queue = q
