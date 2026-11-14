"""Cron-ready scheduled batch trigger: enqueue real dataset rows (or a JSON file of cases) and process them.

    python scripts/scheduled_batch.py --rows 9046 31112 --rag-mode classical          # enqueue + run now
    python scripts/scheduled_batch.py --cases eval/cases.json --limit 3 --enqueue-only  # leave for the worker
    crontab:  0 6 * * 1  cd /app && python scripts/scheduled_batch.py --top-age 3 >> /var/log/neuroguard.log 2>&1

Exit code 0 = every processed job finished (done), 1 = at least one failed, 2 = bad arguments / no API key.
Prints one JSON line per job (job id, status, run id, risk band, receipt prefix) - no PHI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None, run=None) -> int:
    from core.chat import parse_case_request
    from core.config import get_credentials
    from core.job_queue import get_queue
    from core.worker import process_one
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", nargs="*", type=int, default=[], help="dataset row ids")
    ap.add_argument("--top-age", type=int, default=0, help="N oldest distinct-age rows of the public dataset")
    ap.add_argument("--cases", type=Path, help="JSON list of {profile, question?, audience?}")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--rag-mode", choices=["classical", "agentic", "agentic_search"])
    ap.add_argument("--enqueue-only", action="store_true")
    a = ap.parse_args(argv)
    payloads = [{"profile": parse_case_request(f"row {r}")} for r in a.rows]
    if a.top_age:
        from core.data import load_stroke
        df, _ = load_stroke()
        payloads += [{"profile": parse_case_request(f"row {int(i)}", df)} for i in df.sort_values("age", ascending=False).drop_duplicates("age").head(a.top_age)["id"]]
    if a.cases:
        payloads += [c for c in json.loads(a.cases.read_text()) if "profile" in c]
    payloads = payloads[: a.limit]
    if not payloads:
        print("no cases given", file=sys.stderr)
        return 2
    if not a.enqueue_only and run is None and not get_credentials()[0]:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 2
    q = get_queue()
    ids = [q.enqueue("cases", {**p, "rag_mode": a.rag_mode, "session_id": "scheduled"}, source="schedule") for p in payloads]
    if a.enqueue_only:
        print(json.dumps({"enqueued": ids}))
        return 0
    bad = 0
    for _ in ids:
        job = process_one(q, run)
        if job is None:
            break
        s = (job.get("result") or {}).get("summary") or {}
        print(json.dumps({"job_id": job["job_id"], "status": job["status"], "run_status": s.get("status"), "run_id": s.get("run_id"),
                          "risk_band": s.get("risk_band"), "receipt": str(s.get("receipt_signature"))[:16]}))
        bad += job["status"] != "done"
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
