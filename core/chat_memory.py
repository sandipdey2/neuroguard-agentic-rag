"""Conversation episodic memory: de-identified question/answer history per session, recalled for follow-ups.

* every turn is PHI-redacted with the same deterministic redactor as the input guardrail before it is kept;
* consent-gated: with consent the turns are persisted (SQLite ``chat_turns`` table, or PostgreSQL via DATABASE_URL);
  without consent they live only in this process for the session and disappear on restart;
* bounded: at most ``max_turns_kept`` turns per session (older ones are deleted), ``max_chars_per_turn`` per turn,
  and only the last ``recall_turns`` turns are injected into the follow-up agent's context window;
* never shared across sessions (session id is the only key) and clearable from the UI / API.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from pathlib import Path

from .config import RUNTIME, load_contract
from .db import connect
from .guardrails import redact

SCHEMA = """CREATE TABLE IF NOT EXISTS chat_turns (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, created_at REAL NOT NULL, role TEXT NOT NULL,
  text TEXT NOT NULL, run_id TEXT, phi_redactions INTEGER)"""


def _cfg() -> dict:
    return {"max_turns_kept": 20, "recall_turns": 6, "max_chars_per_turn": 800, **load_contract().get("conversation_memory", {})}


def db_path() -> Path:
    return Path(os.environ.get("NG_CHAT_DB", os.environ.get("NG_MEMORY_DB", RUNTIME / "neuroguard_memory.sqlite")))


class ConversationMemory:
    def __init__(self):
        self._mem: dict[str, list[dict]] = defaultdict(list)
        self._lock = threading.Lock()
        self.cfg = _cfg()

    def _conn(self):
        c = connect(db_path())
        c.execute(SCHEMA)
        return c

    def add(self, session_id: str, role: str, text: str, consent: bool, run_id: str | None = None) -> dict:
        if role not in ("user", "assistant"):
            raise ValueError("role must be user or assistant")
        red, found = redact(text or "")
        red = red[: self.cfg["max_chars_per_turn"]]
        turn = {"role": role, "text": red, "run_id": run_id, "phi_redactions": len(found), "created_at": time.time(),
                "persisted": bool(consent)}
        if consent:
            with self._conn() as c:
                c.execute("INSERT INTO chat_turns(session_id, created_at, role, text, run_id, phi_redactions) VALUES (?,?,?,?,?,?)",
                          (session_id, turn["created_at"], role, red, run_id, len(found)))
                ids = [r[0] for r in c.execute("SELECT id FROM chat_turns WHERE session_id=? ORDER BY id DESC", (session_id,)).fetchall()]
                for old in ids[self.cfg["max_turns_kept"]:]:
                    c.execute("DELETE FROM chat_turns WHERE id=?", (old,))
        else:
            with self._lock:
                self._mem[session_id].append(turn)
                del self._mem[session_id][: -self.cfg["max_turns_kept"]]
        return turn

    def history(self, session_id: str, limit: int | None = None) -> list[dict]:
        limit = limit or self.cfg["max_turns_kept"]
        with self._conn() as c:
            rows = c.execute("SELECT role, text, run_id, phi_redactions, created_at FROM chat_turns WHERE session_id=? "
                             "ORDER BY id DESC LIMIT ?", (session_id, int(limit))).fetchall()
        stored = [{"role": r[0], "text": r[1], "run_id": r[2], "phi_redactions": r[3], "created_at": r[4], "persisted": True} for r in rows]
        with self._lock:
            volatile = list(self._mem.get(session_id, []))
        turns = sorted(stored + volatile, key=lambda t: t["created_at"])
        return turns[-limit:]

    def recall(self, session_id: str) -> list[dict]:
        return self.history(session_id, self.cfg["recall_turns"])

    def as_context(self, session_id: str) -> str:
        turns = self.recall(session_id)
        return "\n".join(f"{t['role'].upper()}: {t['text']}" for t in turns) or "(no earlier turns)"

    def clear(self, session_id: str) -> int:
        with self._lock:
            n = len(self._mem.pop(session_id, []))
        with self._conn() as c:
            n += c.execute("DELETE FROM chat_turns WHERE session_id=?", (session_id,)).rowcount
        return n


_memory: ConversationMemory | None = None


def get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory
