"""Agent as a tool + conversation memory, end to end over the REAL MCP server with a scripted (mocked) LLM."""
import asyncio

from langchain_core.messages import AIMessage

from core.agent_tools import SubFinding, SubOut


class ScriptedLLM:
    """Stand-in for ChatOpenAI: bind_tools / with_structured_output return objects that pop scripted replies in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def _next(self):
        self.calls += 1
        return self.script.pop(0)

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, msgs):
        return self._next()

    def with_structured_output(self, schema, include_raw=False):
        outer = self

        class S:
            async def ainvoke(self, msgs):
                parsed = outer._next()
                return {"raw": AIMessage(content=""), "parsed": parsed} if include_raw else parsed
        return S()


def tc(name, args, i):
    return AIMessage(content="Thought: need evidence", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


def test_followup_agent_calls_evidence_subagent_as_tool(monkeypatch, tmp_path):
    monkeypatch.setenv("NG_CHAT_DB", str(tmp_path / "chat.sqlite"))
    import core.chat_memory as cm
    cm._memory = None
    from core.chat import afollowup
    script = [
        tc("consult_evidence_researcher", {"question": "blood pressure and stroke", "factors": ["hypertension"]}, 1),  # follow-up agent
        tc("search_guidelines", {"query": "high blood pressure stroke risk", "k": 3}, 2),                            # sub-agent ReAct
        tc("pubmed_search", {"query": "hypertension", "k": 2}, 3),
        AIMessage(content="done"),
        SubOut(findings=[SubFinding(claim="High blood pressure is a leading cause of stroke.", citation_id="G-03"),
                         SubFinding(claim="Invented.", citation_id="G-99")]),                                         # G-99 never retrieved
        AIMessage(content="High blood pressure is a leading cause of stroke, so discuss blood pressure control [G-03]."),
    ]
    llm = ScriptedLLM(script)
    case = {"run_id": "r1", "risk_percent": 15.0, "risk_band": "high", "top_factors": ["age"]}
    out = asyncio.run(afollowup("What should we discuss about blood pressure?", case, "aat-session", consent=True, llm=llm))
    assert out["status"] == "answered" and out["citations"] == ["G-03"] and "G-03" in out["retrieved"]
    assert "not a medical device" in out["answer"]
    events = [(e["event"], e["from"], e.get("to")) for e in out["a2a"]]
    assert ("task/send", "followup_agent", "evidence_subagent") in events and any(e[0] == "task/completed" for e in events)
    tools = [e["action"].split("(")[0] for e in out["trace"] if e.get("type") == "react_step"]
    assert tools == ["search_guidelines", "pubmed_search"]
    assert out["literature"] and out["literature"][0]["url"].startswith("https://pubmed.ncbi.nlm.nih.gov/")
    sub = [e for e in out["trace"] if e.get("type") == "agent_output"][0]["output"]
    assert sub["dropped_unretrieved"] == 1                                    # hallucinated citation dropped by code
    assert out["receipt"]["algorithm"] == "HMAC-SHA256" and out["llm_calls"] == 6
    # conversation memory: the next follow-up sees this turn (PHI-redacted)
    from core.chat_memory import get_memory
    hist = get_memory().recall("aat-session")
    assert hist[-2]["role"] == "user" and "blood pressure" in hist[-2]["text"] and hist[-1]["persisted"]


def test_followup_blocked_by_guardrail_before_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("NG_CHAT_DB", str(tmp_path / "chat.sqlite"))
    from core.chat import afollowup
    llm = ScriptedLLM([])
    out = asyncio.run(afollowup("What dose of aspirin should he take?", None, "s", llm=llm))
    assert out["status"] == "blocked" and out["guardrail"] == "medication_dosing" and llm.calls == 0


def test_withheld_when_output_checks_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("NG_CHAT_DB", str(tmp_path / "chat.sqlite"))
    from core.chat import afollowup
    llm = ScriptedLLM([AIMessage(content="He should take aspirin 81 mg daily.")])
    out = asyncio.run(afollowup("What should we discuss about her stroke risk next?", None, "s2", llm=llm))
    assert out["status"] == "withheld" and "aspirin" not in out["answer"].lower()
