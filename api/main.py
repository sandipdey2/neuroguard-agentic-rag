"""NeuroGuard HTTP API (FastAPI) - the API gateway in front of the LangGraph + MCP agent system.

    uvicorn api.main:app --host 0.0.0.0 --port 8000        (docs: /docs, OpenAPI schema: /openapi.json)

Endpoints
  GET  /health                         liveness + backends (no auth)
  POST /v1/cases                       enqueue a case on the durable queue; ?wait=true blocks until it finishes (<= 330 s)
  GET  /v1/cases/{job_id}              job status + de-identified summary, report and signed receipt when done
  GET  /v1/receipts/{job_id}/verify    HMAC-SHA256 verification of that job's receipt (needs the same NG_RECEIPT_KEY)
  POST /v1/webhooks/case               webhook trigger: body signed with X-NeuroGuard-Signature: sha256=<HMAC(NG_WEBHOOK_SECRET, body)>
  POST /v1/chat                        chat turn (case request or follow-up with conversation memory)
  GET  /v1/events/{topic}              recent queue events (a2a, case_results)

Auth: ``X-API-Key`` must equal the NG_API_KEY environment variable (constant-time compare). Without NG_API_KEY the
API refuses every protected call unless NG_API_ALLOW_ANON=1 (local development only). Keys are never logged.
CORS: NG_CORS_ORIGINS (comma separated) for browser / mobile web clients. Streamlit Community Cloud cannot host this
service (it only runs ``streamlit run``); use Docker / docker-compose / Kubernetes / any container host.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import load_contract
from core.db import describe as db_describe
from core.harness import verify_receipt
from core.job_queue import get_queue

VERSION = "2.3.0"
WAIT_MAX_S = 330


class Profile(BaseModel):
    age: float = Field(ge=0, le=120, examples=[67])
    gender: Literal["Female", "Male", "Other"] = "Male"
    hypertension: int = Field(0, ge=0, le=1)
    heart_disease: int = Field(0, ge=0, le=1)
    ever_married: Literal["Yes", "No"] = "Yes"
    work_type: Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"] = "Private"
    Residence_type: Literal["Urban", "Rural"] = "Urban"
    avg_glucose_level: float = Field(ge=40, le=400, examples=[228.69])
    bmi: Optional[float] = Field(None, ge=10, le=80, examples=[36.6])
    smoking_status: Literal["never smoked", "formerly smoked", "smokes", "Unknown"] = "never smoked"


class CaseRequest(BaseModel):
    profile: Profile
    question: str = Field("Explain this patient's stroke risk and what to discuss at the next visit.", max_length=500)
    note: str = Field("", max_length=4000)
    audience: Literal["clinician", "patient"] = "clinician"
    rag_mode: Optional[Literal["classical", "agentic", "agentic_search"]] = None
    session_id: Optional[str] = Field(None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    consent: bool = False


class ChatRequest(BaseModel):
    message: str = Field(max_length=1000)
    session_id: str = Field(max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    case_job_id: Optional[str] = None
    consent: bool = False


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("NG_API_KEY", "")
    if not expected:
        if os.environ.get("NG_API_ALLOW_ANON") == "1":
            return
        raise HTTPException(503, "NG_API_KEY is not configured on the server")
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(401, "missing or invalid X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop = None
    if os.environ.get("NG_API_INLINE_WORKER", "1") == "1" and not getattr(app.state, "no_worker", False):
        from core.worker import start_background_worker
        _, stop = start_background_worker()
    yield
    if stop is not None:
        stop.set()


app = FastAPI(title="NeuroGuard AI API", version=VERSION, lifespan=lifespan,
              description="Stroke-risk decision support (not a medical device): LangGraph multi-agent system over a real MCP "
                          "server, durable job queue, signed receipts. Educational use; public data only.")
_origins = [o.strip() for o in os.environ.get("NG_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"],
                       allow_headers=["X-API-Key", "Content-Type", "X-NeuroGuard-Signature"], allow_credentials=False)


def _public(job: dict) -> dict:
    out = {k: job[k] for k in ("job_id", "status", "attempts", "source", "created_at", "updated_at", "error")}
    if job.get("result"):
        out.update(job["result"])
    return out


@app.get("/health")
def health():
    c = load_contract()
    q = get_queue()
    from core.cache import get_cache
    return {"status": "ok", "version": VERSION, "model": c["model"], "contract_version": c["contract_version"],
            "queue": {"backend": q.backend, "counts": q.counts()}, "database": db_describe(), "cache": get_cache().backend,
            "auth": "api-key" if os.environ.get("NG_API_KEY") else ("anonymous (dev)" if os.environ.get("NG_API_ALLOW_ANON") == "1" else "unconfigured")}


def _wait(job_id: str, timeout: float) -> dict:
    q = get_queue()
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = q.get(job_id)
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.5)
    return q.get(job_id)


@app.post("/v1/cases", status_code=202, dependencies=[Depends(require_key)])
def create_case(req: CaseRequest, wait: bool = Query(False, description="block until the run finishes (max 330 s)")):
    job_id = get_queue().enqueue("cases", req.model_dump(), source="api")
    if wait:
        return _public(_wait(job_id, WAIT_MAX_S))
    return {"job_id": job_id, "status": "queued", "poll": f"/v1/cases/{job_id}"}


@app.get("/v1/cases/{job_id}", dependencies=[Depends(require_key)])
def get_case(job_id: str):
    job = get_queue().get(job_id)
    if not job:
        raise HTTPException(404, "unknown job id")
    return _public(job)


@app.get("/v1/receipts/{job_id}/verify", dependencies=[Depends(require_key)])
def verify(job_id: str):
    job = get_queue().get(job_id)
    if not job or not (job.get("result") or {}).get("receipt"):
        raise HTTPException(404, "no receipt for this job (unknown id or not finished)")
    rc = job["result"]["receipt"]
    return {"job_id": job_id, "run_id": rc.get("run_id"), "valid": verify_receipt(rc), "key_id": rc.get("key_id"),
            "algorithm": rc.get("algorithm"), "contract_sha256": rc.get("contract_sha256"), "status": rc.get("status")}


@app.post("/v1/webhooks/case", status_code=202)
async def webhook(request: Request, x_neuroguard_signature: str | None = Header(default=None)):
    """Webhook trigger (e.g. an EHR sandbox or a form tool). Authenticated by an HMAC signature of the raw body."""
    secret = os.environ.get("NG_WEBHOOK_SECRET", "")
    body = await request.body()
    if not secret:
        raise HTTPException(503, "NG_WEBHOOK_SECRET is not configured")
    good = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not x_neuroguard_signature or not hmac.compare_digest(x_neuroguard_signature, good):
        raise HTTPException(401, "bad webhook signature")
    try:
        req = CaseRequest.model_validate(json.loads(body))
    except Exception as exc:
        raise HTTPException(422, f"invalid case payload: {str(exc)[:200]}")
    job_id = get_queue().enqueue("cases", req.model_dump(), source="webhook")
    return {"job_id": job_id, "status": "queued", "poll": f"/v1/cases/{job_id}"}


@app.post("/v1/chat", dependencies=[Depends(require_key)])
def chat(req: ChatRequest):
    from core.chat import handle_message
    case = None
    if req.case_job_id:
        job = get_queue().get(req.case_job_id)
        case = ((job or {}).get("result") or {}).get("summary")
    out = handle_message(req.message, req.session_id, case=case, consent=req.consent)
    return {"kind": out["kind"], "reply": out["reply"], "case": out.get("case"),
            "receipt": (out.get("followup") or {}).get("receipt") or ((out.get("result") or {}).get("receipt"))}


@app.get("/v1/events/{topic}", dependencies=[Depends(require_key)])
def events(topic: Literal["a2a", "case_results"], limit: int = Query(50, ge=1, le=500)):
    return {"topic": topic, "events": get_queue().events(topic, limit)}
