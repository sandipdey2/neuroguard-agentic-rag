"""Cache layer (data & memory layer): Redis when REDIS_URL is set, otherwise an in-process TTL/LRU cache.

Used for deterministic, re-usable results only - never for anything that holds PHI:
* retrieval cache: ``semantic_search`` / ``search_guidelines`` results keyed by (query, k, mode)
* external-API cache: PubMed E-utilities responses (core/pubmed.py), so repeated questions do not hit NCBI
* LLM response cache (opt-in, ``NG_LLM_CACHE=1``): exact-match cache of (model, temperature, top_p, messages) for
  temperature-0 calls - off by default so evaluation numbers stay honest.

Keys are SHA-256 hashes of a namespace + canonical JSON; values are JSON. A Redis outage silently falls back to the
in-process cache (logged in ``stats()``), so the app never fails because the cache is down. Tests use fakeredis.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict

DEFAULT_TTL_S = 24 * 3600


def make_key(namespace: str, payload) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return f"ng:{namespace}:" + hashlib.sha256(raw.encode()).hexdigest()


class MemoryCache:
    backend = "memory"

    def __init__(self, max_items: int = 2048):
        self._d: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()
        self.max_items = max_items

    def get(self, key: str):
        with self._lock:
            v = self._d.get(key)
            if v is None:
                return None
            exp, data = v
            if exp and exp < time.time():
                self._d.pop(key, None)
                return None
            self._d.move_to_end(key)
            return data

    def set(self, key: str, value: str, ttl: int | None = DEFAULT_TTL_S):
        with self._lock:
            self._d[key] = ((time.time() + ttl) if ttl else 0.0, value)
            self._d.move_to_end(key)
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)

    def delete(self, key: str):
        with self._lock:
            self._d.pop(key, None)

    def clear(self):
        with self._lock:
            self._d.clear()


class RedisCache:
    backend = "redis"

    def __init__(self, client):
        self.r = client

    def get(self, key):
        v = self.r.get(key)
        return v.decode() if isinstance(v, bytes) else v

    def set(self, key, value, ttl=DEFAULT_TTL_S):
        self.r.set(key, value, ex=ttl or None)

    def delete(self, key):
        self.r.delete(key)

    def clear(self):
        for k in self.r.scan_iter("ng:*"):
            self.r.delete(k)


class Cache:
    """Facade with hit/miss counters and automatic fallback from Redis to memory."""

    def __init__(self, primary=None):
        self.fallback = MemoryCache()
        self.primary = primary
        self.hits = self.misses = self.errors = 0

    @property
    def backend(self) -> str:
        return self.primary.backend if self.primary is not None else "memory"

    def _store(self):
        return self.primary if self.primary is not None else self.fallback

    def get_json(self, key: str):
        try:
            v = self._store().get(key)
        except Exception:
            self.errors += 1
            v = self.fallback.get(key)
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(v)

    def set_json(self, key: str, value, ttl: int | None = DEFAULT_TTL_S):
        data = json.dumps(value, default=str)
        try:
            self._store().set(key, data, ttl)
        except Exception:
            self.errors += 1
            self.fallback.set(key, data, ttl)

    def cached(self, namespace: str, payload, fn, ttl: int | None = DEFAULT_TTL_S):
        key = make_key(namespace, payload)
        v = self.get_json(key)
        if v is not None:
            return v
        v = fn()
        self.set_json(key, v, ttl)
        return v

    def stats(self) -> dict:
        return {"backend": self.backend, "hits": self.hits, "misses": self.misses, "errors": self.errors}


_cache: Cache | None = None
_lock = threading.Lock()


def _redis_client(url: str):
    import redis
    c = redis.Redis.from_url(url, socket_timeout=1.5, socket_connect_timeout=1.5)
    c.ping()
    return c


def get_cache() -> Cache:
    global _cache
    with _lock:
        if _cache is None:
            url = os.environ.get("REDIS_URL", "").strip()
            primary = None
            if url:
                try:
                    primary = RedisCache(_redis_client(url))
                except Exception:      # Redis down / not installed -> in-process cache, never an app failure
                    primary = None
            _cache = Cache(primary)
        return _cache


def set_cache(cache: Cache | None) -> None:
    """Testing hook (e.g. ``set_cache(Cache(RedisCache(fakeredis.FakeRedis())))``)."""
    global _cache
    _cache = cache


def llm_cache_enabled() -> bool:
    return os.environ.get("NG_LLM_CACHE") == "1"


def install_llm_cache() -> bool:
    """Opt-in (NG_LLM_CACHE=1) exact-match LangChain LLM cache backed by this cache (Redis or memory)."""
    if not llm_cache_enabled():
        return False
    from langchain_core.caches import BaseCache
    from langchain_core.globals import set_llm_cache
    from langchain_core.load import dumps, loads

    class NGLLMCache(BaseCache):
        def lookup(self, prompt, llm_string):
            v = get_cache().get_json(make_key("llm", [prompt, llm_string]))
            return [loads(g) for g in v] if v else None

        def update(self, prompt, llm_string, return_val):
            get_cache().set_json(make_key("llm", [prompt, llm_string]), [dumps(g) for g in return_val])

        def clear(self, **kwargs):
            get_cache().fallback.clear()

    set_llm_cache(NGLLMCache())
    return True
