"""Cache layer (memory / Redis via fakeredis / fallback) and the database backend switch (SQLite / PostgreSQL)."""
import os

import fakeredis
import pytest

from core import cache as cache_mod
from core.cache import Cache, MemoryCache, RedisCache, make_key
from core.db import backend, describe, translate


def test_memory_cache_ttl_and_lru():
    m = MemoryCache(max_items=2)
    m.set("a", "1", ttl=100)
    m.set("b", "2", ttl=100)
    m.set("c", "3", ttl=100)
    assert m.get("a") is None and m.get("c") == "3"
    m.set("d", "4", ttl=-1)
    assert m.get("d") is None


def test_redis_cache_with_fakeredis():
    c = Cache(RedisCache(fakeredis.FakeRedis()))
    calls = []
    f = lambda: calls.append(1) or {"v": 42}
    assert c.cached("retrieval", ["q", 3], f) == {"v": 42}
    assert c.cached("retrieval", ["q", 3], f) == {"v": 42}
    assert len(calls) == 1 and c.stats()["hits"] == 1 and c.backend == "redis"


def test_redis_outage_falls_back():
    class Down:
        backend = "redis"

        def get(self, k):
            raise ConnectionError("down")

        def set(self, k, v, ttl=None):
            raise ConnectionError("down")
    c = Cache(Down())
    c.set_json("k", {"x": 1})
    assert c.get_json("k") == {"x": 1} and c.errors == 2


def test_get_cache_without_redis_url_is_memory(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    cache_mod.set_cache(None)
    assert cache_mod.get_cache().backend == "memory"
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")       # nothing listens there -> graceful fallback
    cache_mod.set_cache(None)
    assert cache_mod.get_cache().backend == "memory"
    cache_mod.set_cache(None)


def test_pubmed_offline_snapshot_and_cache():
    from core import pubmed
    cache_mod.set_cache(Cache(RedisCache(fakeredis.FakeRedis())))
    r = pubmed.search("hypertension", 2, offline=True)
    assert r["served_from"].startswith("snapshot") and len(r["results"]) == 2
    assert all(x["url"].startswith("https://pubmed.ncbi.nlm.nih.gov/") for x in r["results"])
    assert "error" in pubmed.search("<script>", 2)
    cache_mod.get_cache().set_json(make_key("pubmed", ["smoking", 1]), {"query": "smoking", "results": [{"pmid": "1"}]})
    assert pubmed.search("smoking", 1)["served_from"] == "cache (redis)"
    cache_mod.set_cache(None)


def test_sql_translation():
    s, wid = translate("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY AUTOINCREMENT, x REAL)")
    assert "BIGSERIAL PRIMARY KEY" in s and "DOUBLE PRECISION" in s
    s, wid = translate("INSERT OR REPLACE INTO episodes(run_id, a, b) VALUES (?,?,?)")
    assert s.startswith("INSERT INTO episodes") and "ON CONFLICT (run_id) DO UPDATE SET a=EXCLUDED.a, b=EXCLUDED.b" in s and wid
    assert "%s" in s and "?" not in s


def test_default_backend_is_sqlite():
    assert backend() == "sqlite" and describe().startswith("sqlite")


PG = os.environ.get("NG_TEST_PG_URL")


@pytest.mark.skipif(not PG, reason="set NG_TEST_PG_URL to a PostgreSQL URL to run the real-server backend test")
def test_postgres_backend_stores(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", PG)
    from core import episodes, memory
    from core.chat_memory import ConversationMemory
    from core.job_queue import SQLQueue
    assert backend() == "postgres" and "***" in describe()
    cid = memory.write_case("pg-test", "60-69", 0.12, "high", ["age"], "h" * 64)
    assert cid and memory.read_cases("pg-test")[0]["risk_band"] == "high"
    assert memory.clear_session("pg-test") >= 1
    episodes.write_episode({"run_id": "pg-run-1", "factors": ["age"], "outcome": "finalized"})
    episodes.write_episode({"run_id": "pg-run-1", "factors": ["age", "glucose"], "outcome": "finalized"})   # upsert
    assert any(e["run_id"] == "pg-run-1" for e in episodes.recall(["age", "glucose"], k=5))
    cm = ConversationMemory()
    cm.add("pg-chat", "user", "Call 555-123-4567 please", consent=True)
    assert "[REDACTED-PHONE]" in cm.history("pg-chat")[-1]["text"]
    assert cm.clear("pg-chat") == 1
    q = SQLQueue()
    jid = q.enqueue("cases-pg", {"x": 1})
    job = q.claim("cases-pg", "w1")
    assert job["job_id"] == jid and job["status"] == "running"
    q.complete(jid, {"ok": True})
    assert q.get(jid)["status"] == "done"
