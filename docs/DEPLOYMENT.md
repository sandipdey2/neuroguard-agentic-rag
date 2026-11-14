# Deployment

NeuroGuard is built, evaluated and demoed with **gpt-4o-mini through the Vocareum OpenAI-compatible endpoint**
(temperature 0, top_p 1.0). v2.4 keeps deployment deliberately simple for the capstone: two ways to run the same code.

| Where | What runs | Use |
|---|---|---|
| Streamlit Community Cloud (live) | `app.py`: UI, pipeline, MCP server subprocess, 💬 Chat tab | the public live demo: https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/ |
| Your machine | `uvicorn api.main:app` (FastAPI + inline queue worker), optional `streamlit run app.py` | the REST API, webhook and cron trigger |

**Streamlit Community Cloud cannot host the FastAPI service.** It runs a single Streamlit process and exposes no other
port. The live demo therefore shows the UI and the chat tab; run the API locally.

## 1. Streamlit Community Cloud (live demo)

1. Fork or push the repo to GitHub.
2. On https://share.streamlit.io choose **Create app**, pick the repo, branch `main`, main file `app.py`.
3. In **Advanced settings > Secrets** paste (with your own values):
   ```toml
   OPENAI_API_KEY = "PASTE_YOUR_KEY_HERE"
   OPENAI_API_BASE = "https://openai.vocareum.com/v1"
   NG_RECEIPT_KEY = "PASTE_A_LONG_RANDOM_STRING"
   ```
4. Deploy. The first start trains the models (about 20 s); a full run then takes about 1-2 minutes.

## 2. REST API locally with uvicorn

```bash
pip install -r requirements-api.txt                       # FastAPI, uvicorn (+ optional redis / psycopg clients)
export OPENAI_API_KEY=PASTE_YOUR_KEY_HERE OPENAI_API_BASE=https://openai.vocareum.com/v1
export NG_API_KEY=PASTE_A_LONG_RANDOM_API_KEY_HERE NG_WEBHOOK_SECRET=PASTE_A_LONG_RANDOM_WEBHOOK_SECRET_HERE
uvicorn api.main:app --port 8000                          # OpenAPI docs: http://localhost:8000/docs
python -m core.worker                                     # optional extra worker process on the same queue
python scripts/scheduled_batch.py --rows 9046 31112       # cron trigger; add to crontab for nightly cohorts
```

Example call:

```bash
curl -s -X POST "http://localhost:8000/v1/cases?wait=true" -H "X-API-Key: $NG_API_KEY" -H "Content-Type: application/json" \
     -d '{"profile": {"age": 67, "gender": "Male", "hypertension": 0, "heart_disease": 1, "ever_married": "Yes",
          "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 228.69, "bmi": 36.6,
          "smoking_status": "formerly smoked"}}'
```

Browser clients on another origin need `NG_CORS_ORIGINS` (comma-separated allowlist).

## Backends (all optional, SQLite / in-process by default)

| Variable | Effect | Fallback when unset or unreachable |
|---|---|---|
| `DATABASE_URL=postgresql://...` | case memory, episodes, chat memory and the job queue use PostgreSQL (`core/db.py`, queue claims use `FOR UPDATE SKIP LOCKED`) | SQLite files under `NG_RUNTIME_DIR` (WAL mode) |
| `REDIS_URL=redis://...` | shared cache for retrieval, PubMed and (opt-in `NG_LLM_CACHE=1`) LLM responses (`core/cache.py`) | in-process TTL/LRU cache; Redis errors fall back automatically |
| `NG_QUEUE_URL=redis://...` | job queue on Redis lists (`LMOVE`) instead of SQL | SQL queue |
| `NG_VECTOR_BACKEND=memory` | skip FAISS | FAISS `IndexFlatIP` persisted under `NG_RUNTIME_DIR/vector_store` |

Tested: SQLite always; a real PostgreSQL 16.2 server and a real Redis 6.2 server on the development box
(`test_cache_db.py::test_postgres_backend_real`, `test_queue.py::test_redis_queue_real`, enabled with
`NG_TEST_PG_URL` / `NG_TEST_REDIS_URL`), and fakeredis in CI.

## Secrets

Only placeholders live in the repo: `.env.example` and `.streamlit/secrets.toml.example`.
The MCP server subprocess never receives `OPENAI_API_KEY`, `NG_RECEIPT_KEY` or GitHub tokens.
