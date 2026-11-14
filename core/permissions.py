"""Permission engine, pre-tool-use hook and hash-chained audit log.

Least privilege: each agent can only see (bind_tools) and only call the MCP tools listed for it in
harness_contract.json. The pre-tool-use hook re-checks every call at run time (defence in depth against a
hallucinated tool name or an injected instruction), enforces argument limits and the tool budget, pins
session-scoped arguments that an agent must never choose itself, and blocks writes without consent.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field


@dataclass
class AuditLog:
    entries: list = field(default_factory=list)

    def log(self, actor: str, action: str, target: str, decision: str, **detail) -> dict:
        prev = self.entries[-1]["hash"] if self.entries else "0" * 64
        e = {"seq": len(self.entries) + 1, "ts": round(time.time(), 3), "actor": actor, "action": action,
             "target": target, "decision": decision, "detail": detail, "prev": prev}
        e["hash"] = hashlib.sha256(json.dumps(e, sort_keys=True, default=str).encode()).hexdigest()
        self.entries.append(e)
        return e

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for e in self.entries:
            body = {k: v for k, v in e.items() if k != "hash"}
            if e["prev"] != prev or hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != e["hash"]:
                return False
            prev = e["hash"]
        return True


@dataclass
class Decision:
    allow: bool
    reason: str
    args: dict


class PermissionEngine:
    def __init__(self, contract: dict, audit: AuditLog, session_id: str, consent: bool):
        self.c = contract
        self.audit = audit
        self.session_id = session_id
        self.consent = consent
        self.tool_calls = 0
        self.pinned: dict[str, dict] = {}     # harness-set arguments an agent can never choose itself (per tool)
        self.rag_mode = contract.get("rag_modes", {}).get("default", "agentic")
        self.counts: dict[str, int] = {}

    def allowed_tools(self, agent: str) -> list[str]:
        """Least privilege per agent; the Evidence Researcher's tool set depends on the run's RAG mode."""
        if agent == "evidence_researcher" and "rag_modes" in self.c:
            if self.rag_mode == "classical":
                return []                                   # classical RAG: retrieval is done by the fixed pipeline
            mode = self.c["rag_modes"].get(self.rag_mode, {})
            if "evidence_researcher_tools" in mode:
                return list(mode["evidence_researcher_tools"])
        return list(self.c["agents"].get(agent, {}).get("tools", []))

    def _arg_check(self, agent: str, tool: str, args: dict) -> str | None:
        """Argument checks for the context-retrieval tools; shell and web are the highest-risk (same validators run in the server)."""
        from . import context_tools as ct
        lim = self.c["tool_arg_limits"]
        if tool == "shell":
            if self.counts.get("shell", 0) >= lim.get("shell_calls_max_per_run", 4):
                return "shell call budget for this run exhausted"
            ok, why, _ = ct.check_shell(str(args.get("command", "")))
            return None if ok else f"shell command refused: {why}"
        if tool in ("web_fetch", "web_search"):
            if self.counts.get("web", 0) >= lim.get("web_calls_max_per_run", 3):
                return "web call budget for this run exhausted"
            if tool == "web_fetch":
                ok, why = ct.check_url(str(args.get("url", "")))
                return None if ok else f"URL refused (SSRF guard): {why}"
        if tool == "pubmed_search":
            from . import pubmed
            if self.counts.get("pubmed", 0) >= lim.get("pubmed_calls_max_per_run", 3):
                return "PubMed call budget for this run exhausted"
            ok, why = pubmed.check_query(str(args.get("query", "")))
            if not ok:
                return f"PubMed query refused: {why}"
        if tool == "execute_query":
            ok, why = ct.check_sql(str(args.get("sql", "")))
            return None if ok else f"SQL refused: {why}"
        if tool == "load_skill":
            from .skills import catalog_for
            if args.get("name") not in {m["name"] for m in catalog_for(agent)}:
                return f"skill {args.get('name')!r} is not in {agent}'s catalog"
        return None

    def allowed_prompt(self, agent: str, prompt: str) -> bool:
        return prompt in self.c["agents"].get(agent, {}).get("prompts", [])

    def allowed_resource(self, agent: str, uri: str) -> bool:
        for pat in self.c["agents"].get(agent, {}).get("resources", []):
            if pat == uri or (pat.endswith("{id}") and uri.startswith(pat[: -len("{id}")])):
                return True
        return False

    def pre_tool_use(self, agent: str, tool: str, args: dict) -> Decision:
        args = dict(args or {})
        lim = self.c["tool_arg_limits"]
        if tool not in self.allowed_tools(agent):
            d = Decision(False, f"{agent} is not permitted to call {tool}", args)
        elif self.tool_calls >= self.c["budget"]["max_tool_calls"]:
            d = Decision(False, "tool-call budget exhausted", args)
        elif tool == "calculator" and len(str(args.get("expression", ""))) > lim["calculator.expression_max_chars"]:
            d = Decision(False, "calculator expression too long", args)
        elif self.c["write_tools"].get(tool) == "requires_user_consent" and not self.consent:
            d = Decision(False, "write blocked: user has not consented to case-history storage", args)
        elif (why := self._arg_check(agent, tool, args)) is not None:
            d = Decision(False, why, args)
        else:
            if tool == "search_guidelines":
                args["k"] = max(1, min(int(args.get("k", 4)), lim["search_guidelines.k_max"]))
            if tool in ("memory_read", "memory_write"):
                args["session_id"] = self.session_id          # pinned: agents can never read another session
            if tool == "memory_read":
                args["limit"] = max(1, min(int(args.get("limit", 3)), lim["memory_read.limit_max"]))
            if tool == "graph_traverse":
                args["max_hops"] = max(1, min(int(args.get("max_hops", 2)), lim["graph_traverse.max_hops_max"]))
                args["k"] = max(1, min(int(args.get("k", 6)), lim["graph_traverse.k_max"]))
            if tool == "episode_recall":
                args["k"] = max(1, min(int(args.get("k", 3)), lim["episode_recall.k_max"]))
            if tool == "semantic_search":
                args["k"] = max(1, min(int(args.get("k", 4)), lim["semantic_search.k_max"]))
            if tool == "search_files":
                args["max_results"] = max(1, min(int(args.get("max_results", 8)), lim["search_files.max_results_max"]))
            if tool == "web_search":
                args["k"] = max(1, min(int(args.get("k", 2)), lim["web_search.k_max"]))
            if tool == "pubmed_search":
                args["k"] = max(1, min(int(args.get("k", 3)), lim.get("pubmed_search.k_max", 5)))
                self.counts["pubmed"] = self.counts.get("pubmed", 0) + 1
            if tool == "get_memory":
                args["session_id"] = self.session_id          # pinned
            if tool == "shell":
                self.counts["shell"] = self.counts.get("shell", 0) + 1
            if tool in ("web_fetch", "web_search"):
                self.counts["web"] = self.counts.get("web", 0) + 1
            args.update(self.pinned.get(tool, {}))
            d = Decision(True, "allowed by contract", args)
        if d.allow:
            self.tool_calls += 1
        self.audit.log(agent, "tool_call", tool, "allow" if d.allow else "deny", reason=d.reason,
                       args={k: v for k, v in d.args.items() if k != "session_id"})
        return d
