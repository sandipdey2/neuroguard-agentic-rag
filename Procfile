web: uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
worker: python -m core.worker
ui: streamlit run app.py --server.port ${UI_PORT:-8501} --server.address 0.0.0.0
slack: python -m bots.slack_bot
