"""Long-term memory: SQLite case history per pseudonymous browser session.

Stored only with explicit consent and only de-identified summaries (age band, risk, band, factor names,
report hash) - never the free-text note or the question. Agents reach it exclusively through the MCP tools
memory_read / memory_write, whose session_id is pinned by the pre-tool-use hook. The UI offers a clear button.
On Streamlit Community Cloud the file lives on the container's ephemeral disk and is lost on restart.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from .config import RUNTIME

SCHEMA = """CREATE TABLE IF NOT EXISTS cases (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, created_at REAL NOT NULL,
  age_band TEXT, risk_probability REAL, risk_band TEXT, top_factors TEXT, report_sha256 TEXT)"""


def db_path() -> Path:
    return Path(os.environ.get("NG_MEMORY_DB", RUNTIME / "neuroguard_memory.sqlite"))


def _conn() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p, timeout=10)
    c.execute(SCHEMA)
    return c


def write_case(session_id: str, age_band: str, risk_probability: float, risk_band: str, top_factors: list, report_sha256: str) -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO cases(session_id, created_at, age_band, risk_probability, risk_band, top_factors, report_sha256) "
                        "VALUES (?,?,?,?,?,?,?)", (session_id, time.time(), age_band, float(risk_probability), risk_band,
                                                   json.dumps(top_factors), report_sha256))
        return int(cur.lastrowid)


def read_cases(session_id: str, limit: int = 3) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT id, created_at, age_band, risk_probability, risk_band, top_factors FROM cases "
                         "WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, int(limit))).fetchall()
    return [{"case_id": r[0], "created_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(r[1])), "age_band": r[2],
             "risk_probability": r[3], "risk_band": r[4], "top_factors": json.loads(r[5] or "[]")} for r in rows]


def clear_session(session_id: str) -> int:
    with _conn() as c:
        return c.execute("DELETE FROM cases WHERE session_id=?", (session_id,)).rowcount
