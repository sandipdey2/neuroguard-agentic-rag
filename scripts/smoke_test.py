"""End-to-end check with the real LLM:  python scripts/smoke_test.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.pipeline import run_case  # noqa: E402

P = {"age": 67, "gender": "Male", "hypertension": 0, "heart_disease": 1, "ever_married": "Yes", "work_type": "Private",
     "Residence_type": "Urban", "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked"}
r = run_case(P, "Explain this patient's stroke risk and what to discuss at the next visit.",
             "Follow-up visit. Denies weakness or speech problems. Reports occasional dizziness when standing.",
             session_id="smoke", consent=True)
print("status:", r["status"], "| latency_s:", r["latency_s"], "| llm calls:", r["llm_calls"], "| tool calls:", r["tool_calls"], "| tokens:", r.get("tokens"))
print("route:", r.get("route"))
for t in r["trace"]:
    if t["type"] == "route":
        print(f"  orchestrator -> {t['next']:20s} skill={t['skill']} override={t['override']}")
    elif t["type"] == "mcp_tool_call":
        print(f"    [{t['agent']}] MCP {t['tool']} {t['decision']} {json.dumps(t['args'])[:100]}")
    elif t["type"] in ("review", "gate"):
        print("  ", t["type"], {k: v for k, v in t.items() if k in ("verdict", "status", "failed_checks", "issues", "approval_reasons", "memory")})
    elif t["type"] in ("skill_loaded", "mcp_resource_read", "mcp_prompt_get"):
        print("    ", t["type"], t.get("skill") or t.get("uri") or t.get("prompt"))
print("analyst:", r.get("analyst")); print(r.get("report", "")[:3000])
print("receipt:", {k: r["receipt"][k] for k in ("status", "failed_checks", "signature", "key_id")})
