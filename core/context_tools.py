"""Context-retrieval tools for "agentic search in context engineering" (served by the MCP server).

Sources: local files (knowledge/ + skills/), database (read-only SQLite cohort + semantic search),
web (allowlisted CDC / NIH / WHO pages only) and long-term memory. Every validator here is used TWICE:
by the harness pre-tool-use hook (core/permissions.py) and again inside the MCP server (defense in depth).

Highest-risk tools and their controls
* shell      - no shell interpreter (argv list, shell=False); binaries allowlisted (grep, wc, head, ls);
               metacharacters, absolute paths, ``..`` and dangerous options rejected; cwd = read-only docs dir;
               empty environment (no network config, no secrets); 3 s timeout; 8 KB output cap.
* web_fetch  - https only; host must be (a subdomain of) cdc.gov, nih.gov or who.int; no IP literals, ports,
               credentials or non-default schemes; every redirect hop is re-checked; DNS answers that resolve to
               private / loopback / link-local addresses are refused (SSRF); 8 s timeout; 300 KB cap;
               offline or blocked -> graceful fallback to the local excerpt, never an exception.
* execute_query - single SELECT/WITH statement, sqlite opened read-only (mode=ro) with an authorizer that
               denies every write / attach / pragma, 50-row cap and a progress-handler time limit.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import shlex
import socket
import sqlite3
import subprocess
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

from .config import ROOT, RUNTIME

KNOWLEDGE = ROOT / "knowledge"
SKILLS = ROOT / "skills"
FILE_ROOTS = {"knowledge": KNOWLEDGE, "skills": SKILLS}
WEB_ALLOW = ("cdc.gov", "nih.gov", "who.int")
SHELL_BINS = {"grep", "wc", "head", "ls"}
SHELL_META = re.compile(r"[;&|`$<>(){}\[\]\\!*?~\n\r\t]")
SHELL_BAD_OPTS = {"-r", "-R", "--recursive", "-f", "--file", "--include", "--exclude-from", "-P", "--perl-regexp", "--devices", "-D"}
SQL_START = re.compile(r"^\s*(select|with)\b", re.I)
SQL_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|load_extension)\b", re.I)


# ----------------------------------------------------------------------------- validators (hook + server)
def check_shell(command: str) -> tuple[bool, str, list[str]]:
    if not isinstance(command, str) or not command.strip():
        return False, "empty command", []
    if len(command) > 200:
        return False, "command longer than 200 characters", []
    if SHELL_META.search(command):
        return False, "shell metacharacters are not allowed (no pipes, redirects, globbing, substitution or chaining)", []
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return False, f"unparseable command: {exc}", []
    if argv[0] not in SHELL_BINS:
        return False, f"binary {argv[0]!r} is not in the allowlist {sorted(SHELL_BINS)}", []
    for a in argv[1:]:
        if a.startswith("-"):
            if a.split("=", 1)[0] in SHELL_BAD_OPTS:
                return False, f"option {a!r} is not allowed", []
            continue
        if a.startswith("/") or ".." in Path(a).parts or a.startswith("~"):
            return False, f"path {a!r} escapes the read-only docs directory", []
    return True, "ok", argv


def check_url(url: str) -> tuple[bool, str]:
    try:
        u = urlparse(url)
    except Exception:
        return False, "unparseable URL"
    if u.scheme != "https":
        return False, "only https URLs are allowed"
    if u.username or u.password or "@" in (u.netloc or ""):
        return False, "credentials in URLs are not allowed"
    if u.port not in (None, 443):
        return False, "non-default ports are not allowed"
    host = (u.hostname or "").lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
        return False, "IP-literal hosts are not allowed"
    except ValueError:
        pass
    if not any(host == d or host.endswith("." + d) for d in WEB_ALLOW):
        return False, f"host {host!r} is not on the allowlist {list(WEB_ALLOW)}"
    return True, "ok"


def check_sql(sql: str) -> tuple[bool, str]:
    if not isinstance(sql, str) or len(sql) > 600:
        return False, "query missing or longer than 600 characters"
    body = sql.strip().rstrip(";")
    if ";" in body:
        return False, "only a single statement is allowed"
    if not SQL_START.match(body):
        return False, "only SELECT / WITH queries are allowed"
    if SQL_FORBIDDEN.search(body):
        return False, "write / schema / pragma keywords are not allowed"
    return True, "ok"


# ----------------------------------------------------------------------------- local files
def _files():
    for label, root in FILE_ROOTS.items():
        for p in sorted(root.rglob("*.md")):
            yield label, root, p


def search_files(query: str, max_results: int = 8) -> dict:
    """Case-insensitive keyword search over knowledge/*.md and skills/*/SKILL.md; returns file, line and text."""
    terms = [t for t in re.findall(r"[a-z0-9-]+", query.lower()) if len(t) > 2]
    hits = []
    for label, root, p in _files():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            ll = line.lower()
            score = sum(t in ll for t in terms)
            if score:
                hits.append({"file": f"{label}/{p.relative_to(root).as_posix()}", "line": i, "score": score, "text": line.strip()[:240]})
    hits.sort(key=lambda h: (-h["score"], h["file"], h["line"]))
    return {"query": query, "files_searched": sum(1 for _ in _files()), "results": hits[:max_results]}


def run_shell(command: str) -> dict:
    ok, why, argv = check_shell(command)
    if not ok:
        return {"command": command, "error": f"refused: {why}"}
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(argv, cwd=KNOWLEDGE, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}, capture_output=True,
                            text=True, timeout=3, shell=False, stdin=subprocess.DEVNULL)
        out = cp.stdout[:8000]
        return {"command": command, "cwd": "knowledge/", "exit_code": cp.returncode, "stdout": out,
                "stderr": cp.stderr[:400], "truncated": len(cp.stdout) > 8000, "ms": round((time.perf_counter() - t0) * 1000)}
    except subprocess.TimeoutExpired:
        return {"command": command, "error": "timeout after 3 s"}
    except FileNotFoundError:
        return {"command": command, "error": f"{argv[0]} is not installed"}


# ----------------------------------------------------------------------------- database
def cohort_db() -> Path:
    path = RUNTIME / "cohort.sqlite"
    if not path.exists():
        from .data import load_stroke
        df, _ = load_stroke()
        RUNTIME.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with sqlite3.connect(tmp) as c:
            df.to_sql("patients", c, index=False, if_exists="replace")
        os.replace(tmp, path)
    return path


def _authorizer(action, *args):
    return sqlite3.SQLITE_OK if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION) else sqlite3.SQLITE_DENY


def execute_query(sql: str, max_rows: int = 50) -> dict:
    ok, why = check_sql(sql)
    if not ok:
        return {"sql": sql, "error": f"refused: {why}"}
    c = sqlite3.connect(f"file:{cohort_db()}?mode=ro", uri=True)
    try:
        c.set_authorizer(_authorizer)
        deadline = time.monotonic() + 2.0
        c.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        cur = c.execute(sql.strip().rstrip(";"))
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
        return {"sql": sql, "columns": cols, "rows": [list(r) for r in rows[:max_rows]], "row_count": min(len(rows), max_rows),
                "truncated": len(rows) > max_rows, "table": "patients (Stroke Prediction Dataset, cleaned, read-only)"}
    except sqlite3.Error as exc:
        return {"sql": sql, "error": f"sqlite: {exc}"}
    finally:
        c.close()


# ----------------------------------------------------------------------------- web (allowlisted)
class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer", "noscript"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer", "noscript") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.out.append(data.strip())


FAKE_IP_NET = ipaddress.ip_network("198.18.0.0/15")     # RFC 2544 range used by fake-IP DNS proxies in some sandboxes


def _public_host(host: str) -> bool:
    """DNS-level SSRF check (second layer after the hostname allowlist). Loopback, link-local, RFC 1918, ULA,
    multicast and unspecified answers are always refused. The 198.18.0.0/15 benchmark range is accepted only when
    NG_FAKEIP_DNS=1 (sandboxes whose resolver returns fake IPs for every host); it is refused by default."""
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.version == 4 and ip in FAKE_IP_NET and os.environ.get("NG_FAKEIP_DNS") == "1":
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False
    return True


def web_fetch(url: str, max_chars: int = 3000, offline: bool | None = None) -> dict:
    ok, why = check_url(url)
    if not ok:
        return {"url": url, "error": f"refused: {why}"}
    if offline if offline is not None else os.environ.get("NG_WEB_OFFLINE") == "1":
        return {"url": url, "status": "offline", "error": "web access disabled (offline mode)", "fallback": "use semantic_search excerpts"}
    import httpx
    hops, cur = [], url
    try:
        with httpx.Client(timeout=8, follow_redirects=False, headers={"User-Agent": "NeuroGuard-AI/2.2 (+decision-support demo)"}) as cl:
            for _ in range(4):
                ok, why = check_url(cur)
                host = urlparse(cur).hostname or ""
                if not ok or not _public_host(host):
                    return {"url": url, "error": f"refused redirect/host {cur!r}: {why if not ok else 'non-public address'}", "hops": hops}
                r = cl.get(cur)
                hops.append({"url": cur, "status": r.status_code})
                if r.is_redirect and r.headers.get("location"):
                    cur = urljoin(cur, r.headers["location"])
                    continue
                body = r.content[:300_000].decode(r.encoding or "utf-8", errors="replace")
                p = _Text()
                p.feed(body)
                text = re.sub(r"\s+", " ", " ".join(p.out))
                return {"url": url, "final_url": cur, "status": r.status_code, "hops": hops, "text": text[:max_chars],
                        "chars": len(text), "note": "web text is context only; cite G-xx excerpts"}
        return {"url": url, "error": "too many redirects", "hops": hops}
    except (httpx.HTTPError, OSError) as exc:
        return {"url": url, "status": "offline", "error": f"network unavailable: {type(exc).__name__}", "hops": hops,
                "fallback": "use semantic_search excerpts"}


def web_search(query: str, k: int = 2, offline: bool | None = None) -> dict:
    """'Search' = rank the allowlisted source pages of the guideline corpus by the query, then fetch the best page live.
    No general web search engine is used, so the agent can only reach CDC / NIH / WHO pages."""
    from .rag import get_index
    hits = get_index().search(query, k=max(1, min(k, 3)), mode="hybrid")
    seen, pages = set(), []
    for h in hits:
        if h["url"] in seen:
            continue
        seen.add(h["url"])
        f = web_fetch(h["url"], max_chars=1500, offline=offline)
        pages.append({"id": h["id"], "title": h["title"], "url": h["url"], "live_status": f.get("status"),
                      "live_text": f.get("text", "")[:1500], "error": f.get("error"),
                      "local_excerpt": h["text"]})        # graceful degradation: the local excerpt is always returned
    return {"query": query, "allowlist": list(WEB_ALLOW), "results": pages,
            "degraded": any(p["error"] for p in pages)}
