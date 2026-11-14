"""Durable job queue: SQLite lifecycle, lease recovery, retries, events; Redis backend with fakeredis (+ real Redis if set)."""
import os
import time

import fakeredis
import pytest

from core.job_queue import RedisQueue, SQLQueue


def _lifecycle(q):
    jid = q.enqueue("cases", {"profile": {"age": 67}})
    assert q.get(jid)["status"] == "queued"
    job = q.claim("cases", "w1")
    assert job["job_id"] == jid and job["status"] == "running" and job["attempts"] == 1
    assert q.claim("cases", "w2") is None                       # leased: nobody else gets it
    q.complete(jid, {"summary": {"status": "finalized"}})
    assert q.get(jid)["status"] == "done" and q.get(jid)["result"]["summary"]["status"] == "finalized"
    q.publish("a2a", {"event": "task/send", "from": "orchestrator"})
    assert q.events("a2a")[-1]["event"] == "task/send"


def _retry(q):
    jid = q.enqueue("cases", {"n": 1})
    q.claim("cases")
    assert q.fail(jid, "boom") == "queued"                      # attempt 1 of 2 -> requeued
    q.claim("cases")
    assert q.fail(jid, "boom again") == "failed"
    assert q.claim("cases") is None


def test_sqlite_queue(tmp_path):
    q = SQLQueue(tmp_path / "q.sqlite")
    _lifecycle(q)
    _retry(q)
    assert q.counts() == {"done": 1, "failed": 1}


def test_sqlite_queue_is_durable_and_recovers_expired_lease(tmp_path):
    p = tmp_path / "q.sqlite"
    q = SQLQueue(p)
    q.lease_s = 0.05
    jid = q.enqueue("cases", {"n": 1})
    q.claim("cases", "crashed-worker")
    time.sleep(0.1)
    q2 = SQLQueue(p)                                              # "restart": new process, same file
    job = q2.claim("cases", "w2")
    assert job and job["job_id"] == jid and job["attempts"] == 2 and job["worker"] == "w2"


def test_redis_queue_fakeredis():
    q = RedisQueue(fakeredis.FakeRedis())
    _lifecycle(q)
    _retry(q)


RURL = os.environ.get("NG_TEST_REDIS_URL")


@pytest.mark.skipif(not RURL, reason="set NG_TEST_REDIS_URL to run against a real Redis server")
def test_redis_queue_real_server():
    import redis
    r = redis.Redis.from_url(RURL)
    r.flushdb()
    q = RedisQueue(r)
    _lifecycle(q)
    _retry(q)


def test_a2a_bus_publishes_to_queue(tmp_path):
    from core import a2a
    q = SQLQueue(tmp_path / "q.sqlite")
    a2a.set_sink(q.publish)
    try:
        bus = a2a.MessageBus()
        t = bus.new_task("orchestrator", "intake", "screen the note (contains question text)")
        bus.complete(t, "intake", "intake_summary", {"red_flags": []})
    finally:
        a2a.set_sink(None)
    ev = q.events("a2a")
    assert [e["event"] for e in ev] == ["task/send", "task/completed"]
    assert ev[0]["text"] is None and ev[0]["contextId"] == bus.context_id     # free text is not published
