"""Multi-modal input (PARTIAL): read a lab report image or PDF into structured, validated lab values.

Image  -> gpt-4o-mini vision (same Vocareum endpoint, temperature 0) with a structured ``LabExtract`` schema.
PDF    -> local text extraction with pypdf (no LLM), then deterministic regex parsing; image-only PDFs are refused.
Both   -> values are range-checked (same plausibility limits as the patient form), the raw text is PHI-redacted,
          and the clinician must confirm before anything is applied to the form. Nothing here diagnoses.

Why partial: only three fields (average glucose mg/dL, BMI, blood pressure) are read, there is no OCR for
scanned PDFs, and vision accuracy was tested on a small rendered set (see evaluation.ipynb), not on real
scanned lab reports. The extracted values feed the existing text pipeline; the agents themselves stay text-only.
"""
from __future__ import annotations

import base64
import io
import re
from typing import Optional

from pydantic import BaseModel, Field

from .guardrails import redact

LIMITS = {"avg_glucose_level": (40.0, 400.0), "bmi": (10.0, 80.0), "systolic_bp": (60, 260), "diastolic_bp": (30, 160)}
MAX_BYTES = 4_000_000


class LabExtract(BaseModel):
    avg_glucose_level: Optional[float] = Field(default=None, description="glucose in mg/dL (convert mmol/L x 18.0); null if absent")
    bmi: Optional[float] = Field(default=None, description="body-mass index kg/m2; null if absent")
    systolic_bp: Optional[int] = Field(default=None, description="systolic blood pressure mmHg; null if absent")
    diastolic_bp: Optional[int] = Field(default=None, description="diastolic blood pressure mmHg; null if absent")
    notes: str = Field(default="", description="one short sentence on what the document is; no names or identifiers")


def validate(ex: dict) -> tuple[dict, list[str]]:
    """Drop out-of-range values (never guess) and derive the hypertension hint (>=140/90 mmHg)."""
    out, warnings = {}, []
    for k, (lo, hi) in LIMITS.items():
        v = ex.get(k)
        if v is None:
            continue
        if lo <= float(v) <= hi:
            out[k] = round(float(v), 1) if k in ("avg_glucose_level", "bmi") else int(v)
        else:
            warnings.append(f"{k}={v} outside plausible range {lo}-{hi}; ignored")
    if "systolic_bp" in out and "diastolic_bp" in out:
        out["hypertension_hint"] = int(out["systolic_bp"] >= 140 or out["diastolic_bp"] >= 90)
    return out, warnings


_RX = {
    "avg_glucose_level": r"glucose[^0-9\n]{0,30}(\d{2,3}(?:\.\d+)?)\s*(mg/dl|mmol/l)?",
    "bmi": r"\bbmi[^0-9\n]{0,20}(\d{2}(?:\.\d+)?)",
    "bp": r"(?:blood pressure|\bbp)[^0-9\n]{0,20}(\d{2,3})\s*/\s*(\d{2,3})",
}


def parse_text(text: str) -> dict:
    t = text.lower()
    ex: dict = {}
    m = re.search(_RX["avg_glucose_level"], t)
    if m:
        v = float(m.group(1))
        ex["avg_glucose_level"] = v * 18.0 if (m.group(2) or "").startswith("mmol") else v
    m = re.search(_RX["bmi"], t)
    if m:
        ex["bmi"] = float(m.group(1))
    m = re.search(_RX["bp"], t)
    if m:
        ex["systolic_bp"], ex["diastolic_bp"] = int(m.group(1)), int(m.group(2))
    return ex


def from_pdf(data: bytes) -> dict:
    from pypdf import PdfReader
    if len(data) > MAX_BYTES:
        return {"ok": False, "error": "file too large"}
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join((p.extract_text() or "") for p in reader.pages[:5])
    if len(text.strip()) < 20:
        return {"ok": False, "error": "no text layer (scanned PDF) - OCR is not supported; upload an image instead"}
    vals, warn = validate(parse_text(text))
    red, kinds = redact(text[:1500])
    return {"ok": True, "source": "pdf", "method": "pypdf text + regex (no LLM)", "values": vals, "warnings": warn,
            "redacted_excerpt": red[:600], "phi_redactions": kinds}


def from_image(data: bytes, mime: str = "image/png", llm=None, ctx=None) -> dict:
    """gpt-4o-mini vision; ``llm`` defaults to the contract model. Usage is counted on ``ctx`` when given."""
    from langchain_core.messages import HumanMessage, SystemMessage
    if len(data) > MAX_BYTES:
        return {"ok": False, "error": "file too large"}
    if llm is None:
        from .config import get_llm
        llm = get_llm()
    b64 = base64.b64encode(data).decode()
    msgs = [SystemMessage("Read ONLY these lab values from the image: glucose (mg/dL; convert mmol/L x 18), BMI, blood pressure. "
                          "Return null for anything not clearly visible. Never output names, dates of birth or ID numbers."),
            HumanMessage(content=[{"type": "text", "text": "Extract the lab values."},
                                  {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}}])]
    raw = llm.with_structured_output(LabExtract, include_raw=True).invoke(msgs)
    if ctx is not None and raw.get("raw") is not None:
        ctx.record_llm("multimodal_intake", raw["raw"])
    parsed = raw.get("parsed")
    if parsed is None:
        return {"ok": False, "error": "vision model returned no structured output"}
    ex = parsed.model_dump()
    vals, warn = validate(ex)
    usage = getattr(raw.get("raw"), "usage_metadata", None) or {}
    return {"ok": True, "source": "image", "method": "gpt-4o-mini vision, structured LabExtract (detail=low)", "values": vals,
            "warnings": warn, "notes": redact(ex.get("notes", ""))[0][:200], "tokens": usage.get("total_tokens")}


def to_note(res: dict) -> str:
    """One de-identified line to append to the clinical note (the clinician confirms it first)."""
    v = res.get("values", {})
    parts = []
    if "avg_glucose_level" in v:
        parts.append(f"glucose {v['avg_glucose_level']} mg/dL")
    if "bmi" in v:
        parts.append(f"BMI {v['bmi']}")
    if "systolic_bp" in v:
        parts.append(f"BP {v['systolic_bp']}/{v.get('diastolic_bp', '?')} mmHg")
    return f"Lab report ({res.get('source')}): " + ", ".join(parts) + "." if parts else ""
