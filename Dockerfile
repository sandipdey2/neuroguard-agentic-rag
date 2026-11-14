# NeuroGuard AI - one image for the API, the queue worker, the Streamlit UI and the Slack bot (command differs per service).
# docker build -t neuroguard:2.3.0 .     (secrets are passed at run time as env vars / Kubernetes secrets, never baked in)
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NG_RUNTIME_DIR=/data/runtime
WORKDIR /app
COPY requirements.txt requirements-api.txt requirements-bots.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt -r requirements-bots.txt
COPY . .
RUN useradd -m -u 10001 neuroguard && mkdir -p /data/runtime && chown -R neuroguard /data /app
USER neuroguard
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
