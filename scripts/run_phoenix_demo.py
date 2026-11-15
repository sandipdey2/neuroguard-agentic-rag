"""Run one NeuroGuard case with Arize Phoenix tracing and print how to open the UI.

Prerequisite (Phoenix packages are optional and stay out of requirements.txt):

    # Phoenix UI/server in a separate venv (arize-phoenix 21 is not compatible with this repo's mcp 1.30)
    python -m venv .venv-phoenix && source .venv-phoenix/bin/activate
    pip install arize-phoenix==21.0.0
    phoenix serve                    # UI at http://localhost:6006

    # App venv
    pip install -r requirements.txt -r requirements-phoenix.txt
    export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
    # or: export NG_PHOENIX=1
    export OPENAI_API_KEY=PASTE_YOUR_KEY_HERE
    export OPENAI_API_BASE=https://openai.vocareum.com/v1
    python scripts/run_phoenix_demo.py

The script does not print secrets. The Streamlit Monitoring tab works even when Phoenix is not running.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROFILE = {"age": 67.0, "gender": "Male", "hypertension": 0, "heart_disease": 1, "ever_married": "Yes",
           "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 228.69, "bmi": 36.6,
           "smoking_status": "formerly smoked"}
QUESTION = "Explain this patient's stroke risk and what to discuss at the next visit."


def main() -> int:
    if not os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") and os.environ.get("NG_PHOENIX") != "1":
        os.environ["NG_PHOENIX"] = "1"
    from core.observability import phoenix_status, reset_tracing
    reset_tracing()
    status = phoenix_status()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Export it first (the value is never printed).")
        return 2
    print(f"Phoenix project: {status['project']}")
    print(f"Collector: {status['endpoint']}")
    print(f"UI: {status['ui']}")
    from core.pipeline import run_case
    t0 = time.time()
    res = run_case(PROFILE, QUESTION, audience="clinician", session_id="phoenix-demo", consent=False, rag_mode="agentic")
    elapsed = round(time.time() - t0, 1)
    cost = (res.get("cost") or {}).get("cost_usd")
    print(f"Run {res.get('run_id')} status={res.get('status')} latency_s={res.get('latency_s')} "
          f"(wall {elapsed}s) llm={res.get('llm_calls')} tools={res.get('tool_calls')} cost_usd={cost}")
    # Best-effort: list spans from a local Phoenix if the client is installed.
    try:
        from phoenix.client import Client
        client = Client(base_url=status["ui"])
        df = client.spans.get_spans_dataframe(project_identifier="neuroguard", limit=500)
        names = []
        if df is not None and len(df):
            col = "name" if "name" in df.columns else next((c for c in df.columns if c.endswith("name")), None)
            names = sorted({str(v) for v in (df[col].tolist() if col else [])})
        print(f"Phoenix spans returned: {0 if df is None else len(df)}")
        if names:
            print("Span names:", ", ".join(names[:40]) + (" ..." if len(names) > 40 else ""))
    except Exception as exc:
        print(f"Could not query the Phoenix API yet ({type(exc).__name__}). "
              f"If `phoenix serve` is running, open {status['ui']} and look at project 'neuroguard'.")
    print("Open the Phoenix UI at", status["ui"], "and the Streamlit tab 📈 Monitoring for stored run metrics.")
    return 0 if res.get("status") in {"finalized", "not_finalized", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
