# Calling NeuroGuard from a web or mobile app (REST API)

**Honest scope:** NeuroGuard has **no native iOS or Android app**. The FastAPI service (`api/main.py`) is *mobile-ready*:
a JSON REST API with an OpenAPI schema, CORS for browser and hybrid apps, API-key auth, a non-blocking job model (submit,
then poll), and signed receipts that a client can check. Any mobile stack that can send HTTPS requests
(Swift `URLSession`, Kotlin/OkHttp, React Native / Expo `fetch`, Flutter `http`) can use it. The Streamlit UI also has
a responsive layout (columns wrap and headings and metrics shrink below 640 px), so it can be used in a phone browser.

Decision support only, not a medical device. Send de-identified profiles only.

## Start the API

```bash
pip install -r requirements.txt -r requirements-api.txt
export OPENAI_API_KEY=PASTE_YOUR_KEY_HERE OPENAI_API_BASE=https://openai.vocareum.com/v1
export NG_API_KEY=PASTE_A_LONG_RANDOM_API_KEY_HERE NG_RECEIPT_KEY=PASTE_A_LONG_RANDOM_RECEIPT_KEY_HERE
export NG_CORS_ORIGINS=https://your-web-or-mobile-app.example     # comma-separated, exact origins (no wildcard by default)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

* Interactive docs (Swagger UI): `http://localhost:8000/docs`. The OpenAPI JSON is at `/openapi.json`, so you can generate a typed client with `openapi-generator`.
* By default an in-process worker runs the queued jobs (`NG_API_INLINE_WORKER=1`). In production, run `python -m core.worker` as separate replicas.
* Without `NG_API_KEY` every protected endpoint returns 503. For local development only, `NG_API_ALLOW_ANON=1` disables auth.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | version, model, contract, queue / database / cache backend (no auth) |
| POST | `/v1/cases` | queue a case; `?wait=true` blocks until it finishes (max 330 s) |
| GET | `/v1/cases/{job_id}` | job status plus result summary (risk, band, report, citations, cost, latency) |
| GET | `/v1/receipts/{job_id}/verify` | re-verify the HMAC-signed run receipt server-side |
| POST | `/v1/chat` | follow-up question about a case (conversation memory, agent-as-tool evidence look-up) |
| POST | `/v1/webhooks/case` | server-to-server trigger, authenticated by `X-NeuroGuard-Signature: sha256=<HMAC of body>` |
| GET | `/v1/events/{topic}` | recent `a2a` or `case_results` events from the durable queue |

## curl

```bash
API=http://localhost:8000; KEY=PASTE_A_LONG_RANDOM_API_KEY_HERE
# 1) submit (returns 202 + job_id immediately)
curl -s -X POST "$API/v1/cases" -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d '{
  "profile": {"age": 67, "gender": "Male", "hypertension": 0, "heart_disease": 1, "ever_married": "Yes",
              "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 228.69, "bmi": 36.6,
              "smoking_status": "formerly smoked"},
  "audience": "clinician", "rag_mode": "agentic", "session_id": "mobile-demo-1"}'
# 2) poll until status is done / failed
curl -s "$API/v1/cases/JOB_ID" -H "X-API-Key: $KEY"
# 3) verify the signed receipt
curl -s "$API/v1/receipts/JOB_ID/verify" -H "X-API-Key: $KEY"
# 4) follow-up chat about that case
curl -s -X POST "$API/v1/chat" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"message": "What should we discuss about blood pressure?", "session_id": "mobile-demo-1", "case_job_id": "JOB_ID"}'
```

## JavaScript `fetch` (browser, React Native / Expo)

```js
const API = "https://api.your-domain.example";            // must be listed in NG_CORS_ORIGINS for browser apps
const KEY = "PASTE_A_LONG_RANDOM_API_KEY_HERE";            // in a real app, keep it on your backend or use per-user tokens

async function assess(profile) {
  const headers = { "X-API-Key": KEY, "Content-Type": "application/json" };
  const r = await fetch(`${API}/v1/cases`, { method: "POST", headers,
    body: JSON.stringify({ profile, audience: "patient", session_id: "app-user-42" }) });
  if (r.status !== 202) throw new Error(`submit failed: ${r.status}`);
  const { job_id } = await r.json();
  for (let i = 0; i < 120; i++) {                            // poll every 3 s, up to 6 min
    const job = await (await fetch(`${API}/v1/cases/${job_id}`, { headers })).json();
    if (job.status === "done" || job.status === "failed") return job;
    await new Promise(res => setTimeout(res, 3000));
  }
  throw new Error("timed out");
}
```

Notes for mobile developers:

* **Do not ship a shared API key inside a public app binary.** Put a thin backend (or an API gateway with per-user auth) in front. `X-API-Key` is a service credential.
* A run takes about 30-90 s (several LLM calls). Use submit-and-poll, or `?wait=true` only from a backend.
* Blocked inputs (emergency signs, dosing questions, prompt injection) come back as a completed job with the refusal or 9-1-1 message and a signed receipt. Show that text verbatim.
