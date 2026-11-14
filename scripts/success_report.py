"""Compare observed results with the success criteria pre-declared in harness_contract.json["success_criteria"].

    python scripts/success_report.py [--pytest-passed N --pytest-total M]
Reads eval/summary.json (notebook), eval/logs/e2e_v23.json (live end-to-end runs) and writes
eval/success_criteria.json + eval/success_criteria.md. A missing observation is reported as "not measured", never guessed.
"""
from __future__ import annotations

import argparse
import json
import operator
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq}


def lookup(src: str, summary: dict, e2e: dict, pytest_rate):
    kind, _, path = src.partition(" ")
    if kind == "pytest":
        return pytest_rate
    if kind == "e2e":
        vals = [r[path] for r in e2e.get("pipeline_runs", []) if r.get(path) is not None]
        return round(max(vals), 6) if vals else None          # worst case across the live runs
    cur = summary
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main(argv=None) -> int:
    from core.config import load_contract
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytest-passed", type=int)
    ap.add_argument("--pytest-total", type=int)
    a = ap.parse_args(argv)
    sc = load_contract()["success_criteria"]
    summary = json.loads((ROOT / "eval" / "summary.json").read_text())
    e2e_p = ROOT / "eval" / "logs" / "e2e_v23.json"
    e2e = json.loads(e2e_p.read_text()) if e2e_p.exists() else {}
    rate = round(a.pytest_passed / a.pytest_total, 3) if a.pytest_total else None
    rows = []
    for t in sc["targets"]:
        obs = lookup(t["source"], summary, e2e, rate)
        met = None if obs is None else bool(OPS[t["op"]](obs, t["target"]))
        rows.append({**t, "observed": obs, "met": met})
    out = {"declared": sc["declared"], "rows": rows, "met": sum(r["met"] is True for r in rows),
           "missed": [r["metric"] for r in rows if r["met"] is False], "not_measured": [r["metric"] for r in rows if r["met"] is None]}
    (ROOT / "eval" / "success_criteria.json").write_text(json.dumps(out, indent=1))
    md = ["| Metric | Target | Observed | Met |", "|---|---|---|---|"]
    md += [f"| {r['metric']} | {r['op']} {r['target']} | {r['observed'] if r['observed'] is not None else 'not measured'} | "
           f"{'✅' if r['met'] else ('❌' if r['met'] is False else '—')} |" for r in rows]
    (ROOT / "eval" / "success_criteria.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"met {out['met']}/{len(rows)}; missed {out['missed']}; not measured {out['not_measured']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
