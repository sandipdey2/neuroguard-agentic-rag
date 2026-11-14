"""Database backend switch for the data & memory layer: SQLite by default, PostgreSQL when DATABASE_URL is set.

Every store (long-term case memory, episodic memory, conversation memory, durable job queue) opens its connection
through ``connect(sqlite_path)``. Without ``DATABASE_URL`` that is a plain ``sqlite3`` connection on the store's own
file (the Streamlit Cloud default - nothing to install). With ``DATABASE_URL=postgresql://user:pass@host:5432/db``
(docker-compose / Kubernetes profile) it returns a thin wrapper over ``psycopg`` 3 that accepts the same SQLite-style
statements, so the stores need no second code path:

* ``?`` placeholders -> ``%s``; ``INTEGER PRIMARY KEY AUTOINCREMENT`` -> ``BIGSERIAL PRIMARY KEY``; ``REAL`` -> ``DOUBLE PRECISION``
* ``INSERT OR REPLACE`` / ``INSERT OR IGNORE`` -> ``INSERT ... ON CONFLICT`` (conflict key per table in ``CONFLICT_KEYS``)
* ``cursor.lastrowid`` works for inserts (``RETURNING id`` is appended); ``rowcount`` / ``fetchone`` / ``fetchall`` as usual.

Credentials stay in the environment / secret store; the URL is never logged (``describe()`` masks the password).
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

CONFLICT_KEYS = {"episodes": "run_id", "kv_cache": "k"}


def database_url() -> str | None:
    url = os.environ.get("DATABASE_URL", "").strip()
    return url if url.startswith(("postgres://", "postgresql://")) else None


def backend() -> str:
    return "postgres" if database_url() else "sqlite"


def describe() -> str:
    url = database_url()
    if not url:
        return "sqlite (local file per store)"
    u = urlparse(url)
    host = u.hostname
    if not host:  # unix-socket URI: postgresql://user@/db?host=/path
        from urllib.parse import parse_qs
        host = (parse_qs(u.query).get("host") or ["localhost"])[0]
    return f"postgres://{u.username or ''}:***@{host}:{u.port or 5432}{u.path}"


_INS = re.compile(r"^\s*INSERT\s+OR\s+(REPLACE|IGNORE)\s+INTO\s+(\w+)\s*\(([^)]*)\)", re.I | re.S)


def translate(sql: str) -> tuple[str, bool]:
    """SQLite dialect -> PostgreSQL. Returns (sql, wants_returning_id)."""
    s = sql.replace("?", "%s")
    s = re.sub(r"INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY", s, flags=re.I)
    s = re.sub(r"\bREAL\b", "DOUBLE PRECISION", s)
    s = re.sub(r"\bBLOB\b", "BYTEA", s)
    m = _INS.match(s)
    if m:
        mode, table, cols = m.group(1).upper(), m.group(2), [c.strip() for c in m.group(3).split(",")]
        s = re.sub(r"^\s*INSERT\s+OR\s+(REPLACE|IGNORE)\s+INTO", "INSERT INTO", s, count=1, flags=re.I)
        key = CONFLICT_KEYS.get(table)
        if key and mode == "REPLACE":
            s += f" ON CONFLICT ({key}) DO UPDATE SET " + ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != key)
        else:
            s += " ON CONFLICT DO NOTHING"
    wants_id = bool(re.match(r"^\s*INSERT\s", s, re.I)) and "RETURNING" not in s.upper()
    return s, wants_id


class _PgCursor:
    def __init__(self, cur, lastrowid=None):
        self._cur, self.lastrowid = cur, lastrowid
        self.rowcount = cur.rowcount

    def fetchall(self):
        return [tuple(r) for r in self._cur.fetchall()] if self._cur.description else []

    def fetchone(self):
        r = self._cur.fetchone() if self._cur.description else None
        return tuple(r) if r is not None else None

    def fetchmany(self, n):
        return [tuple(r) for r in self._cur.fetchmany(n)] if self._cur.description else []

    @property
    def description(self):
        return self._cur.description


class PgConnection:
    """sqlite3-like facade over a psycopg 3 connection (only what the stores use)."""

    def __init__(self, url: str):
        import psycopg
        self._c = psycopg.connect(url, autocommit=False, connect_timeout=5)

    def execute(self, sql: str, params=()):
        s, wants_id = translate(sql)
        cur = self._c.cursor()
        if s.lstrip().upper().startswith("CREATE"):
            cur.execute(s)
            return _PgCursor(cur)
        if wants_id and _has_id(s):
            try:
                cur.execute("SAVEPOINT ng_ins")
                cur.execute(s + " RETURNING id", tuple(params))
                row = cur.fetchone()
                cur.execute("RELEASE SAVEPOINT ng_ins")
                c2 = _PgCursor(cur, row[0] if row else None)
                c2.rowcount = 1 if row else 0
                return c2
            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT ng_ins")
                raise
        cur.execute(s, tuple(params))
        return _PgCursor(cur)

    def executescript(self, script: str):
        for stmt in [x for x in script.split(";") if x.strip()]:
            self.execute(stmt)

    def commit(self):
        self._c.commit()

    def close(self):
        self._c.close()

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        try:
            (self._c.rollback if et else self._c.commit)()
        finally:
            self._c.close()
        return False


_ID_TABLES = {"cases", "episodes", "chat_turns", "jobs", "events"}


def _has_id(sql: str) -> bool:
    m = re.match(r"^\s*INSERT\s+INTO\s+(\w+)", sql, re.I)
    return bool(m) and m.group(1) in _ID_TABLES


class _SqliteConn(sqlite3.Connection):
    """sqlite3 connection that also closes when used as a context manager (sqlite3's own only commits)."""

    def __exit__(self, et, ev, tb):
        r = super().__exit__(et, ev, tb)
        self.close()
        return r


def connect(sqlite_path: Path | str):
    url = database_url()
    if url:
        return PgConnection(url)
    p = Path(sqlite_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p, timeout=10, factory=_SqliteConn)
    c.execute("PRAGMA journal_mode=WAL")
    return c
