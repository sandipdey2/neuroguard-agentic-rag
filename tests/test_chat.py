"""Chat routing, conversation memory and case-request parsing (LLM pipeline mocked)."""
from core.chat import case_summary, handle_message, parse_case_request, summary_text
from core.chat_memory import ConversationMemory


def test_parse_case_request():
    p = parse_case_request("assess row 9046")
    assert p["age"] == 67.0 and p["gender"] == "Male" and p["heart_disease"] == 1
    p = parse_case_request("age=54 gender=female hypertension=1 glucose=180.5 bmi=31 smoking=former")
    assert p == {**p, "age": 54.0, "gender": "Female", "hypertension": 1, "avg_glucose_level": 180.5, "bmi": 31.0,
                 "smoking_status": "formerly smoked"}
    assert parse_case_request("what about blood pressure?") is None


def test_conversation_memory_redacts_bounds_and_consent(tmp_path, monkeypatch):
    monkeypatch.setenv("NG_CHAT_DB", str(tmp_path / "chat.sqlite"))
    m = ConversationMemory()
    m.cfg["max_turns_kept"] = 4
    m.add("s1", "user", "Patient Mr. John Smith, phone 555-123-4567", consent=False)
    t = m.history("s1")[0]
    assert "555-123-4567" not in t["text"] and "John Smith" not in t["text"] and not t["persisted"]
    for i in range(6):
        m.add("s2", "user", f"question {i}", consent=True)
    h = m.history("s2")
    assert len(h) == 4 and h[-1]["text"] == "question 5" and all(x["persisted"] for x in h)
    assert ConversationMemory().history("s2")[-1]["text"] == "question 5"     # persisted across instances
    assert ConversationMemory().history("s1") == []                           # no consent -> not persisted
    assert m.history("other") == []
    assert len(m.recall("s2")) <= m.cfg["recall_turns"]
    assert m.clear("s2") == 4 and m.history("s2") == []


def test_handle_message_case_then_followup(fake_run):
    out = handle_message("assess row 9046", "chat-test", run=fake_run)
    assert out["kind"] == "case" and "15.0%" in out["reply"] and "[G-03]" in out["reply"]
    seen = {}

    def follow(q, case, sid, consent):
        seen.update(q=q, case=case)
        return {"answer": "Blood pressure is a leading factor [G-03].", "receipt": {"signature": "ab" * 32}}
    out2 = handle_message("what should we discuss about blood pressure?", "chat-test", case=out["case"], follow=follow)
    assert out2["kind"] == "followup" and seen["case"]["risk_band"] == "high"


def test_summary_text_blocked():
    s = case_summary({"status": "blocked", "input_guardrail": {"reason": "emergency"}, "receipt": {"signature": "f" * 64}})
    assert "blocked" in summary_text(s)
