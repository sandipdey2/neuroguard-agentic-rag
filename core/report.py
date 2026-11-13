"""Downloadable report: Markdown and a PDF rendered with fpdf2 (pure Python, works on Streamlit Cloud)."""
from __future__ import annotations

import json
import re

from fpdf import FPDF

_MAP = {"\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2022": "-",
        "\u2265": ">=", "\u2264": "<=", "\u00a0": " ", "\u2026": "...", "\u03c3": "sigma", "\u2192": "->"}


def _latin(t: str) -> str:
    for k, v in _MAP.items():
        t = t.replace(k, v)
    return t.encode("latin-1", "replace").decode("latin-1")


def full_markdown(result: dict, approval: dict | None = None) -> str:
    final = result.get("final") or {}
    lines = ["# NeuroGuard AI - stroke-risk decision-support report", "",
             f"Run `{result['run_id']}` | status **{result['status']}** | model {result['receipt']['model']} | "
             f"contract sha256 `{result['contract_sha256'][:16]}...`", ""]
    if final.get("needs_human_approval"):
        a = (approval or {}).get("decision", "PENDING")
        lines += [f"> Clinician sign-off required ({', '.join(final.get('approval_reasons', []))}): **{a}**", ""]
    if result["status"] != "finalized":
        lines += ["> DRAFT WITHHELD BY THE FINALIZE GATE - failed checks: " + ", ".join((final.get("verification") or {}).get("failed", [])), ""]
    lines += [result.get("report", ""), "", "## Sources"]
    for cid, d in sorted((result.get("retrieved") or {}).items()):
        lines.append(f"- [{cid}] {d['publisher']}, \"{d['title']}\", {d['url']} (accessed {d['accessed']}; {d['license']})")
    lines += ["", "## Verification checks"]
    for k, v in ((final.get("verification") or {}).get("checks") or {}).items():
        lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines += ["", f"Receipt signature (HMAC-SHA256): `{result['receipt']['signature']}`", "",
              "*NeuroGuard AI is clinical decision support for education and demonstration. It is not a medical device, "
              "has no regulatory clearance, and must not be used for diagnosis or treatment decisions.*"]
    return "\n".join(lines)


def to_pdf(md: str) -> bytes:
    pdf = FPDF(format="Letter")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_margins(16, 14, 16)
    for raw in md.splitlines():
        line = _latin(raw.rstrip())
        clean = re.sub(r"\*\*|`", "", line)
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 15); pdf.multi_cell(0, 8, clean[2:]); pdf.ln(1)
        elif line.startswith("## "):
            pdf.ln(2); pdf.set_font("Helvetica", "B", 12); pdf.multi_cell(0, 7, clean[3:])
        elif line.startswith(("- ", "* ")):
            pdf.set_font("Helvetica", "", 10); pdf.set_x(20); pdf.multi_cell(0, 5, "- " + clean[2:])
        elif line.startswith("> "):
            pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(160, 30, 30); pdf.multi_cell(0, 5, clean[2:]); pdf.set_text_color(0, 0, 0)
        elif not line.strip():
            pdf.ln(2)
        else:
            pdf.set_font("Helvetica", "I" if line.startswith("*") else "", 10); pdf.multi_cell(0, 5, clean.strip("*"))
        pdf.set_x(16)
    return bytes(pdf.output())


def receipt_json(receipt: dict) -> str:
    return json.dumps(receipt, indent=1, sort_keys=True, default=str)
