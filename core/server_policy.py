"""Server-side policy engine of the MCP server (defence in depth behind the client pre-tool-use hook).

The client hook (core/permissions.py) knows WHICH agent is calling and enforces per-agent least privilege. The MCP
server cannot trust its client, so it re-checks every ``tools/call`` itself against harness_contract.json before the
tool body runs:

1. registry allowlist - the tool must be granted to at least one agent / RAG mode in the contract;
2. argument policy    - numeric limits are clamped to ``tool_arg_limits`` (k, limit, max_hops, max_results ...);
3. validators         - shell argv allowlist, SSRF URL allowlist, read-only SQL, PubMed query, calculator length;
4. per-run quotas     - one server process serves exactly one run (stdio subprocess), so per-process counters are
                        per-run quotas: shell, web and PubMed calls;
5. write policy       - memory_write needs a pinned session id; episode_write accepts only whitelisted fields (in the tool).

A refusal is returned to the caller as data (``{"error": "refused by server policy: ..."}``) and counted; the
``registry://tools`` resource publishes every tool with its policy so clients and auditors can read it.
"""
from __future__ import annotations

import functools
import json
from collections import Counter

from .config import load_contract

CLAMPS = {  # tool -> {arg: (contract limit key, default max)}
    "search_guidelines": {"k": ("search_guidelines.k_max", 5)},
    "semantic_search": {"k": ("semantic_search.k_max", 5)},
    "memory_read": {"limit": ("memory_read.limit_max", 5)},
    "graph_traverse": {"max_hops": ("graph_traverse.max_hops_max", 3), "k": ("graph_traverse.k_max", 8)},
    "episode_recall": {"k": ("episode_recall.k_max", 3)},
    "search_files": {"max_results": ("search_files.max_results_max", 10)},
    "web_search": {"k": ("web_search.k_max", 3)},
    "pubmed_search": {"k": ("pubmed_search.k_max", 5)},
}
QUOTAS = {"shell": ("shell_calls_max_per_run", 4), "web_search": ("web_calls_max_per_run", 3),
          "web_fetch": ("web_calls_max_per_run", 3), "pubmed_search": ("pubmed_calls_max_per_run", 3)}
QUOTA_GROUP = {"web_search": "web", "web_fetch": "web", "shell": "shell", "pubmed_search": "pubmed"}


class ServerPolicy:
    def __init__(self, contract: dict | None = None):
        self.c = contract or load_contract()
        self.lim = self.c["tool_arg_limits"]
        self.counts: Counter = Counter()
        self.refusals: list[dict] = []

    def granted_tools(self) -> dict[str, list[str]]:
        """tool -> agents / modes that are granted it in the contract."""
        g: dict[str, list[str]] = {}
        for agent, spec in self.c["agents"].items():
            for t in spec.get("tools", []):
                g.setdefault(t, []).append(agent)
        for mode, spec in self.c.get("rag_modes", {}).items():
            if isinstance(spec, dict):
                for t in spec.get("evidence_researcher_tools", []):
                    g.setdefault(t, []).append(f"evidence_researcher[{mode}]")
        return g

    def check(self, tool: str, args: dict) -> tuple[bool, str, dict]:
        from . import context_tools as ct
        from . import pubmed
        args = dict(args)
        if tool not in self.granted_tools():
            return False, f"tool {tool!r} is not granted to any agent in harness_contract.json", args
        for arg, (key, dflt) in CLAMPS.get(tool, {}).items():
            if arg in args and args[arg] is not None:
                try:
                    args[arg] = max(1, min(int(args[arg]), int(self.lim.get(key, dflt))))
                except (TypeError, ValueError):
                    return False, f"argument {arg!r} must be an integer", args
        if tool in QUOTAS:
            key, dflt = QUOTAS[tool]
            grp = QUOTA_GROUP[tool]
            if self.counts[grp] >= int(self.lim.get(key, dflt)):
                return False, f"per-run {grp} quota ({self.lim.get(key, dflt)}) exhausted", args
        if tool == "shell":
            ok, why, _ = ct.check_shell(str(args.get("command", "")))
            if not ok:
                return False, f"shell: {why}", args
        if tool == "web_fetch":
            ok, why = ct.check_url(str(args.get("url", "")))
            if not ok:
                return False, f"url: {why}", args
        if tool == "execute_query":
            ok, why = ct.check_sql(str(args.get("sql", "")))
            if not ok:
                return False, f"sql: {why}", args
        if tool == "pubmed_search":
            ok, why = pubmed.check_query(str(args.get("query", "")))
            if not ok:
                return False, f"pubmed: {why}", args
        if tool == "calculator" and len(str(args.get("expression", ""))) > int(self.lim.get("calculator.expression_max_chars", 120)):
            return False, "calculator expression too long", args
        if tool in ("memory_write", "memory_read", "get_memory") and not str(args.get("session_id", "")).strip():
            return False, "session_id must be pinned by the client harness", args
        if tool in QUOTAS:
            self.counts[QUOTA_GROUP[tool]] += 1
        self.counts["allowed"] += 1
        return True, "allowed by server policy", args

    def guard(self, tool: str):
        """Decorator for an MCP tool function: policy check first, refusal returned as JSON data."""
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(**kwargs):
                ok, why, args = self.check(tool, kwargs)
                if not ok:
                    self.counts["refused"] += 1
                    self.refusals.append({"tool": tool, "reason": why})
                    return json.dumps({"error": f"refused by server policy: {why}", "tool": tool})
                return fn(**args)
            return wrapper
        return deco

    def registry(self, tool_docs: dict[str, str]) -> dict:
        g = self.granted_tools()
        rows = []
        for name in sorted(tool_docs):
            rows.append({"tool": name, "description": (tool_docs[name] or "").strip().splitlines()[0][:200] if tool_docs[name] else "",
                         "granted_to": g.get(name, []),
                         "clamps": {a: f"<= {self.lim.get(k, d)}" for a, (k, d) in CLAMPS.get(name, {}).items()},
                         "per_run_quota": (self.lim.get(QUOTAS[name][0], QUOTAS[name][1]) if name in QUOTAS else None),
                         "write_policy": self.c.get("write_tools", {}).get(name),
                         "high_risk_controls": self.c.get("high_risk_tools", {}).get(name)})
        return {"contract_version": self.c["contract_version"], "tools": rows,
                "enforcement": ["client pre-tool-use hook (per agent)", "server policy engine (this server, per call)"],
                "counters": dict(self.counts), "recent_refusals": self.refusals[-10:]}
