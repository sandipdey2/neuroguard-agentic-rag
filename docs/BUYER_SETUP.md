# NeuroGuard AI - buyer setup guide

**Repository:** https://github.com/sandipdey2/neuroguard-agentic-rag · **Live demo:** https://neuroguard-agentic-rag-4ywylkye6e2rgzavdb5xrb.streamlit.app/ (try it before installing)

Time needed: about 15 minutes. You need Python 3.13 or 3.14, a GitHub account (for Streamlit Community Cloud)
and an OpenAI-compatible API key (OpenAI, or a course gateway such as Vocareum).

## 1. Unzip and install
```bash
unzip neuroguard-ai-2.3.0.zip && cd neuroguard-ai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the stroke dataset (not bundled)
The Stroke Prediction Dataset is published for **educational use**. Read its terms on
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset, then run:
```bash
python scripts/fetch_data.py      # downloads and checksums the public CSV into data/raw/
```
For any commercial deployment, replace it with data you are licensed to use (same 12 columns) and retrain:
the model, statistics and cohort tool all rebuild automatically from `data/raw/`.

## 3. Add your key
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit OPENAI_API_KEY, OPENAI_API_BASE (https://api.openai.com/v1 for OpenAI) and NG_RECEIPT_KEY
```
The key is read only from Streamlit secrets or environment variables and is never passed to the MCP server.

## 4. Check and run
```bash
python scripts/verify_mcp.py      # MCP tools/resources/prompts over stdio, no API key needed
python scripts/smoke_test.py      # one full multi-agent run with the real LLM
streamlit run app.py
```

## 5. Deploy to Streamlit Community Cloud
1. Push the folder to a GitHub repository (the `.gitignore` already excludes secrets and caches).
2. On https://share.streamlit.io choose **Create app**, pick the repo, branch `main`, main file `app.py`.
3. In **Advanced settings > Secrets** paste the contents of your `secrets.toml`. Python 3.13 or 3.14 both work.
4. Deploy. The first start trains the models (about 20 s); later runs take roughly 70-120 s each (mean 87.2 s in the evaluation).

## 6. Optional: REST API, workers, chat and Slack (v2.3)
```bash
pip install -r requirements-api.txt                     # FastAPI, uvicorn, redis, psycopg
export NG_API_KEY=PASTE_A_LONG_RANDOM_API_KEY_HERE NG_WEBHOOK_SECRET=PASTE_A_LONG_RANDOM_WEBHOOK_SECRET_HERE
uvicorn api.main:app --port 8000                        # OpenAPI docs: http://localhost:8000/docs
python scripts/scheduled_batch.py --rows 9046 31112     # cron trigger; add to crontab for nightly cohorts
```
- Web or mobile clients: see `docs/MOBILE_CLIENT.md` (curl, JS `fetch`, CORS via `NG_CORS_ORIGINS`). There is no native mobile app.
- Shared storage: set `DATABASE_URL` (PostgreSQL) and/or `REDIS_URL`; without them everything falls back to local SQLite and an in-process cache.
- **Slack bot:** create a Slack app, enable Socket Mode, add the slash command `/neuroguard` and the `commands` + `chat:write` scopes,
  install it to your workspace, then `pip install -r requirements-bots.txt` and
  `export SLACK_BOT_TOKEN=xoxb-PASTE_YOUR_TOKEN_HERE SLACK_APP_TOKEN=xapp-PASTE_YOUR_TOKEN_HERE` and run `python -m bots.slack_bot`.
- **Docker Compose / Kubernetes:** `cp .env.example .env`, fill in the placeholders, `docker compose up -d --build`; manifests in `deploy/k8s/`. See `docs/DEPLOYMENT.md`.
  Streamlit Community Cloud hosts only the UI, not the API, worker or bot.
- Tests: `pip install -r requirements-dev.txt && pytest` (no API key needed). To run CI on GitHub, copy `ci/github-actions-ci.yml` to `.github/workflows/ci.yml`.

## 7. Customise
- Budgets, turn caps, per-agent tool permissions, required checks and success criteria: `harness_contract.json` (re-run `python scripts/success_report.py` after an evaluation).
- Agent behaviour: `skills/*/SKILL.md` (front matter = catalog entry; body = instructions loaded on demand).
- Guideline corpus: `data/guidelines.json` (keep `id`, `url`, `license` for every chunk).
- Report template and red-team prompt: MCP prompts in `mcp_server/neuroguard_mcp_server.py`.
- Re-run `evaluation.ipynb` after any change and keep the published numbers in sync with its outputs.

## Important limits
NeuroGuard AI is **not a medical device**, has no FDA or other regulatory clearance, and must not be used for
diagnosis, treatment or emergency triage. Do not enter real patient identifiers unless you have a compliant
deployment (BAA, access control, logging, retention policy). Streamlit Community Cloud is not HIPAA-compliant.
