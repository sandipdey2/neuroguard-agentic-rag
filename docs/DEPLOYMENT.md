# Deployment profiles

NeuroGuard is built, evaluated and demoed with **gpt-4o-mini through the Vocareum OpenAI-compatible endpoint**
(temperature 0, top_p 1.0). Everything below is packaging around that same code.

| Profile | What runs | Use |
|---|---|---|
| Streamlit Community Cloud | `app.py` only (UI, pipeline, MCP subprocess, in-app chat tab) | the public live demo |
| Single host, no Docker | `uvicorn api.main:app` (+ inline worker), optional `streamlit run app.py`, `python -m bots.slack_bot` | development, small pilots |
| Docker Compose | api, worker, ui, redis, postgres (+ `slack` profile; optional `vllm` profile) | one VM / on-prem server |
| Kubernetes | `deploy/k8s/*.yaml`: namespace + ConfigMap, Secret **template**, api / worker / ui / slack-bot Deployments, Services, PostgreSQL StatefulSet, Redis, HPA, Ingress | clusters |

**Streamlit Community Cloud cannot host the FastAPI service, the worker or the Slack bot.** It runs a single Streamlit
process and exposes no other port or background service. The live demo therefore shows the UI and the chat tab. The API runs
via Docker, Kubernetes, the `Procfile` (Render / Railway / Heroku-style hosts) or any VM.

## Backends (all optional, with fallbacks)

| Variable | Effect | Fallback when unset or unreachable |
|---|---|---|
| `DATABASE_URL=postgresql://...` | case memory, episodes, chat memory and the job queue use PostgreSQL (`core/db.py`, queue claims use `FOR UPDATE SKIP LOCKED`) | SQLite files under `NG_RUNTIME_DIR` (WAL mode) |
| `REDIS_URL=redis://...` | shared cache for retrieval, PubMed and (opt-in `NG_LLM_CACHE=1`) LLM responses (`core/cache.py`) | in-process TTL/LRU cache; Redis errors fall back automatically |
| `NG_QUEUE_URL=redis://...` | job queue on Redis lists (`LMOVE`) instead of SQL | SQL queue |
| `NG_VECTOR_BACKEND=memory` | skip FAISS | FAISS `IndexFlatIP` persisted under `NG_RUNTIME_DIR/vector_store` |

Tested in this release: SQLite, plus a real PostgreSQL 16.2 server and a real Redis 6.2 server on the development box
(tests `test_cache_db.py::test_postgres_backend_real` and `test_queue.py::test_redis_queue_real`, enabled with
`NG_TEST_PG_URL` / `NG_TEST_REDIS_URL`), and fakeredis in CI. The live FastAPI end-to-end run used the PostgreSQL queue and the Redis cache.

## Docker Compose

```bash
cp .env.example .env        # fill in placeholders; never commit .env
docker compose up -d --build                     # api :8000, ui :8501, worker, redis, postgres
docker compose --profile slack up -d slack-bot   # optional Slack bot (socket mode, no public URL needed)
docker compose up -d --scale worker=3            # more workers on the same queue
```

## Kubernetes

```bash
cp deploy/k8s/01-secret.template.yaml /tmp/secret.yaml   # fill in real values OUTSIDE the repo
kubectl apply -f deploy/k8s/00-namespace-config.yaml -f /tmp/secret.yaml
kubectl apply -f deploy/k8s/02-app.yaml -f deploy/k8s/03-data.yaml -f deploy/k8s/04-hpa.yaml -f deploy/k8s/05-ingress.yaml
```

The image is built from the repo `Dockerfile` (python:3.13-slim, non-root user 10001). Push it to your registry and
set the image name in `02-app.yaml`. The HPA scales the `worker` Deployment (2-8 replicas) and `api` (2-6) on CPU.
The manifests pass `kubeconform -strict` (18/18 resources) and `docker compose config` passes for every profile; see
`eval/logs/deploy_validation.log`. They have **not** been applied to a live cluster in this release, and the images were not
built on the development box (no Docker daemon there).

## Optional on-prem LLM path: vLLM (not the default)

For sites that cannot send de-identified text to a hosted API, `docker-compose.yml` (profile `vllm`) and
`deploy/k8s/optional/vllm.yaml` describe a self-hosted **vLLM** server with an OpenAI-compatible `/v1` API. It is
**optional, off by default, and not used in any test, evaluation or the live demo.** To try it, point `OPENAI_API_BASE` at
`http://vllm:8000/v1`, set the served model name in `harness_contract.json` (this changes the contract hash), and
**re-run the evaluation notebook** before relying on it: every number in this repo was measured with gpt-4o-mini.

## Secrets

Only placeholders live in the repo: `.env.example`, `.streamlit/secrets.toml.example`, `deploy/k8s/01-secret.template.yaml`.
The MCP server subprocess never receives `OPENAI_API_KEY`, `NG_RECEIPT_KEY` or GitHub tokens.
