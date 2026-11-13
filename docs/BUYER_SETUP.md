# NeuroGuard AI - buyer setup guide

Time needed: about 15 minutes. You need Python 3.13 or 3.14, a GitHub account (for Streamlit Community Cloud)
and an OpenAI-compatible API key (OpenAI, or a course gateway such as Vocareum).

## 1. Unzip and install
```bash
unzip neuroguard-ai-2.0.0.zip && cd neuroguard-ai
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
4. Deploy. The first start trains the models (about 20 s); later runs take roughly 50-90 s each.

## 6. Customise
- Budgets, turn caps, per-agent tool permissions and required checks: `harness_contract.json`.
- Agent behaviour: `skills/*/SKILL.md` (front matter = catalog entry; body = instructions loaded on demand).
- Guideline corpus: `data/guidelines.json` (keep `id`, `url`, `license` for every chunk).
- Report template and red-team prompt: MCP prompts in `mcp_server/neuroguard_mcp_server.py`.
- Re-run `evaluation.ipynb` after any change and keep the published numbers in sync with its outputs.

## Important limits
NeuroGuard AI is **not a medical device**, has no FDA or other regulatory clearance, and must not be used for
diagnosis, treatment or emergency triage. Do not enter real patient identifiers unless you have a compliant
deployment (BAA, access control, logging, retention policy). Streamlit Community Cloud is not HIPAA-compliant.
