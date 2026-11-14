"""Second official public data API: NCBI PubMed E-utilities (esearch + esummary), keyless.

Served by the MCP tool ``pubmed_search``. Returns bibliographic METADATA only (PMID, title, journal, year, authors'
count, PubMed URL) for peer-reviewed literature related to the patient's risk factors - never abstracts as
citations; the report still cites only the verbatim CDC/NHLBI G-xx excerpts. Real data only: no synthetic records.

Controls (enforced here, in the MCP server policy layer and in the client pre-tool-use hook)
* fixed host ``eutils.ncbi.nlm.nih.gov`` over https, only the ``esearch.fcgi`` and ``esummary.fcgi`` endpoints;
* NCBI etiquette: ``tool=neuroguard`` parameter, <= 3 requests/second process-wide (NCBI's keyless limit),
  8 s timeout, query <= 200 chars, k <= 5, 3 calls per run (contract ``pubmed_calls_max_per_run``);
* cache: responses are cached (core/cache.py: Redis or memory, 24 h) and a snapshot of REAL responses fetched from
  NCBI for the standard risk-factor queries is committed in ``data/pubmed_cache.json`` as the offline fallback.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from .cache import get_cache, make_key

HOST = "eutils.ncbi.nlm.nih.gov"
BASE = f"https://{HOST}/entrez/eutils/"
ENDPOINTS = ("esearch.fcgi", "esummary.fcgi")
SNAPSHOT = Path(__file__).resolve().parent.parent / "data" / "pubmed_cache.json"
MAX_RPS = 3.0
TIMEOUT_S = 8.0

_lock = threading.Lock()
_last: list[float] = []


def _rate_limit() -> float:
    """Sliding one-second window: at most MAX_RPS requests per second in this process. Returns seconds waited."""
    waited = 0.0
    with _lock:
        while True:
            now = time.monotonic()
            while _last and now - _last[0] >= 1.0:
                _last.pop(0)
            if len(_last) < MAX_RPS:
                _last.append(now)
                return waited
            w = 1.0 - (now - _last[0]) + 0.01
            waited += w
            time.sleep(w)


def normalize_query(q: str) -> str:
    q = re.sub(r"[^A-Za-z0-9 \-]", " ", str(q or ""))
    return re.sub(r"\s+", " ", q).strip().lower()[:200]


def check_query(q: str) -> tuple[bool, str]:
    if not isinstance(q, str) or not q.strip():
        return False, "empty query"
    if len(q) > 200:
        return False, "query longer than 200 characters"
    if re.search(r"https?://|[<>{}`$\\]", q):
        return False, "URLs and special characters are not allowed in a PubMed query"
    return True, "ok"


def _get(endpoint: str, params: dict) -> dict:
    assert endpoint in ENDPOINTS
    import httpx
    _rate_limit()
    with httpx.Client(timeout=TIMEOUT_S, follow_redirects=False,
                      headers={"User-Agent": "NeuroGuard-AI/2.4 (+https://github.com/sandipdey2/neuroguard-agentic-rag)"}) as cl:
        r = cl.get(BASE + endpoint, params={**params, "retmode": "json", "tool": "neuroguard"})
        r.raise_for_status()
        return r.json()


def _snapshot() -> dict:
    try:
        return json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def fetch_live(query: str, k: int) -> dict:
    term = f"({query}) AND stroke"
    es = _get("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": k, "sort": "relevance"})
    ids = es.get("esearchresult", {}).get("idlist", [])[:k]
    count = int(es.get("esearchresult", {}).get("count", 0) or 0)
    items = []
    if ids:
        sm = _get("esummary.fcgi", {"db": "pubmed", "id": ",".join(ids)}).get("result", {})
        for pmid in ids:
            d = sm.get(pmid) or {}
            items.append({"pmid": pmid, "title": d.get("title", "").strip(), "journal": d.get("fulljournalname") or d.get("source"),
                          "year": (d.get("pubdate") or "")[:4], "authors": len(d.get("authors") or []),
                          "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
    return {"query": query, "term": term, "total_hits": count, "results": items}


def search(query: str, k: int = 3, offline: bool | None = None) -> dict:
    ok, why = check_query(query)
    if not ok:
        return {"query": query, "error": f"refused: {why}"}
    q, k = normalize_query(query), max(1, min(int(k), 5))
    base = {"source": "NCBI PubMed E-utilities (esearch + esummary)", "host": HOST,
            "note": "bibliographic metadata for context only; cite G-xx guideline excerpts in the report"}
    cache = get_cache()
    hit = cache.get_json(make_key("pubmed", [q, k]))
    if hit:
        return {**hit, **base, "served_from": f"cache ({cache.backend})"}
    offline = offline if offline is not None else os.environ.get("NG_WEB_OFFLINE") == "1"
    err = None
    if not offline:
        try:
            res = fetch_live(q, k)
            cache.set_json(make_key("pubmed", [q, k]), res)
            return {**res, **base, "served_from": "live"}
        except Exception as exc:          # network down / NCBI throttled -> real cached snapshot
            err = f"{type(exc).__name__}: {str(exc)[:120]}"
    snap = _snapshot()
    key = q if q in snap else next((s for s in snap if set(s.split()) & set(q.split())), None)
    if key:
        return {**snap[key], "results": snap[key]["results"][:k], **base, "served_from": "snapshot data/pubmed_cache.json",
                "snapshot_query": key, "live_error": err or "offline mode"}
    return {"query": q, "results": [], **base, "served_from": "none", "error": err or "offline and no snapshot for this query"}
