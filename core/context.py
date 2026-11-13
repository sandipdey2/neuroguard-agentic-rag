"""Context-window management with real token counts (tiktoken, o200k_base = the gpt-4o / gpt-4o-mini encoding).

gpt-4o-mini has a 128k-token context window, but cost, latency and "lost in the middle" effects make a small,
explicit budget better. The contract's ``context`` section sets per-use budgets:
  * ``max_tool_output_tokens``  - each MCP tool result is truncated (head kept, marker appended) before it enters
                                  an agent's message history;
  * ``writer_evidence_tokens``  - the evidence payload for the report prompt is assembled most-relevant-first and
                                  excerpts that do not fit are reduced to their findings only;
  * ``agent_context_tokens``    - the whole message list of a tool-using agent; the oldest tool observations are
                                  compacted first if it grows past the budget.
Every assembly is measured and logged (trace type ``context``) so the UI shows tokens used vs budget.
"""
from __future__ import annotations

from functools import lru_cache

import tiktoken

MODEL_CONTEXT_WINDOW = 128_000


@lru_cache(maxsize=1)
def _enc():
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:            # offline fallback: tiktoken downloads encodings on first use
        return None


def count(text: str) -> int:
    enc = _enc()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text or "", disallowed_special=()))


def truncate(text: str, max_tokens: int) -> tuple[str, bool]:
    enc = _enc()
    if count(text) <= max_tokens:
        return text, False
    if enc is None:
        return text[: max_tokens * 4] + " ...[truncated to fit the context budget]", True
    ids = enc.encode(text, disallowed_special=())
    return enc.decode(ids[:max_tokens]) + " ...[truncated to fit the context budget]", True


def messages_tokens(msgs) -> int:
    total = 0
    for m in msgs:
        c = getattr(m, "content", m)
        total += 4 + count(c if isinstance(c, str) else str(c))     # ~4 tokens of chat-format overhead per message
    return total


def assemble_evidence(items: list[dict], budget: int) -> tuple[list[dict], dict]:
    """items are already ordered most-relevant-first; keep full text while it fits, then findings only."""
    out, used, reduced = [], 0, []
    for it in items:
        full = count(str(it))
        if used + full <= budget:
            out.append(it)
            used += full
        else:
            slim = {k: v for k, v in it.items() if k != "text"}
            out.append(slim)
            used += count(str(slim))
            reduced.append(it.get("id"))
    return out, {"budget": budget, "used": used, "excerpts": len(items), "reduced_to_findings": reduced}
