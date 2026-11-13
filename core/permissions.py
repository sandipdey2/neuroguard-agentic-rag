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

    def allowed_tools(self, agent: str) -> list[str]:
        return list(self.c["agents"].get(agent, {}).get("tools", []))

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
        elif tool in self.c["write_tools"] and not self.consent:
            d = Decision(False, "write blocked: user has not consented to case-history storage", args)
        else:
            if tool == "search_guidelines":
                args["k"] = max(1, min(int(args.get("k", 4)), lim["search_guidelines.k_max"]))
            if tool in ("memory_read", "memory_write"):
                args["session_id"] = self.session_id          # pinned: agents can never read another session
            if tool == "memory_read":
                args["limit"] = max(1, min(int(args.get("limit", 3)), lim["memory_read.limit_max"]))
            d = Decision(True, "allowed by contract", args)
        if d.allow:
            self.tool_calls += 1
        self.audit.log(agent, "tool_call", tool, "allow" if d.allow else "deny", reason=d.reason,
                       args={k: v for k, v in d.args.items() if k != "session_id"})
        return d
