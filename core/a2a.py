"""A2A-style agent communication (modelled on the open Agent2Agent protocol's data shapes).

Honest scope: this is an in-process message bus, not a networked A2A server. Each agent publishes an Agent
Card (name, description, version, endpoint, skills, capabilities). The orchestrator delegates by sending a
Task whose history holds a Message (role "user", text + data parts); the receiving agent returns the Task
with status "completed" (or "failed") and an Artifact carrying its structured result. Every send/receive is
recorded on the bus and shown in the UI. The endpoints use an ``inproc://`` scheme to make clear that no
network hop exists; swapping the bus for HTTP JSON-RPC would not change the payloads. With ``set_sink`` installed (the API
worker does this) every event is also published to the durable queue's 'a2a' topic (core/job_queue.py); the text
part of task/send is not published (it may quote the question).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .skills import catalog_for

AGENT_DESCRIPTIONS = {
    "intake": "Screens the PHI-redacted note with the BE FAST rule and the note classifier; reads de-identified prior cases.",
    "risk_analyst": "Runs the calibrated stroke-risk model, explains factor contributions and compares with cohort statistics.",
    "evidence_researcher": "Retrieves CDC/NHLBI guideline excerpts for the patient's risk factors and returns cited claims.",
    "clinical_writer": "Generates the decision-support report from the other agents' artifacts using the MCP report prompt.",
    "safety_reviewer": "Runs the deterministic invariant verifier and an LLM red-team pass; can return the draft for one revision.",
}


def agent_card(name: str, contract: dict) -> dict:
    spec = contract["agents"][name]
    return {
        "name": name, "description": AGENT_DESCRIPTIONS[name], "version": contract["contract_version"],
        "url": f"inproc://neuroguard/agents/{name}", "protocol": "a2a-style (in-process)",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text", "data"], "defaultOutputModes": ["data"],
        "skills": [{"id": s["name"], "name": s["name"], "description": s["description"]} for s in catalog_for(name)],
        "mcp": {"tools": spec["tools"], "resources": spec["resources"], "prompts": spec["prompts"]},
    }


_SINK = None


def set_sink(fn) -> None:
    """Publish every bus event to an external sink too (e.g. ``job_queue.get_queue().publish`` on topic 'a2a')."""
    global _SINK
    _SINK = fn


def _emit(ev: dict) -> None:
    if _SINK is not None:
        try:
            _SINK("a2a", ev)
        except Exception:      # an unavailable queue must never break a run
            pass


def _id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class MessageBus:
    context_id: str = field(default_factory=_id)
    log: list = field(default_factory=list)

    def new_task(self, sender: str, receiver: str, text: str, data: dict | None = None, skill: str | None = None) -> dict:
        task_id = _id()
        msg = {"kind": "message", "messageId": _id(), "role": "user", "taskId": task_id, "contextId": self.context_id,
               "parts": [{"kind": "text", "text": text}] + ([{"kind": "data", "data": data}] if data else []),
               "metadata": {"from": sender, "to": receiver, "skill": skill}}
        task = {"kind": "task", "id": task_id, "contextId": self.context_id, "status": {"state": "submitted"},
                "history": [msg], "artifacts": []}
        self.log.append({"ts": round(time.time(), 3), "event": "task/send", "from": sender, "to": receiver,
                         "taskId": task_id, "skill": skill, "text": text[:400]})
        _emit({**self.log[-1], "contextId": self.context_id, "text": None})
        return task

    def complete(self, task: dict, agent: str, name: str, data: dict, state: str = "completed") -> dict:
        art = {"artifactId": _id(), "name": name, "parts": [{"kind": "data", "data": data}]}
        task["artifacts"].append(art)
        task["status"] = {"state": state, "timestamp": round(time.time(), 3)}
        task["history"].append({"kind": "message", "messageId": _id(), "role": "agent", "taskId": task["id"],
                                "contextId": self.context_id, "parts": [{"kind": "text", "text": f"{name} ready"}],
                                "metadata": {"from": agent}})
        self.log.append({"ts": round(time.time(), 3), "event": f"task/{state}", "from": agent, "to": "orchestrator",
                         "taskId": task["id"], "artifact": name, "artifact_keys": sorted(data)[:12]})
        _emit({**self.log[-1], "contextId": self.context_id})
        return task
