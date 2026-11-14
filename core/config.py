"""Configuration: paths, LLM settings and secrets. Keys come from Streamlit secrets or the environment only."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RUNTIME = Path(os.environ.get("NG_RUNTIME_DIR", ROOT / "data" / "runtime"))  # git-ignored: model cache + SQLite memory
CONTRACT_PATH = ROOT / "harness_contract.json"

DEFAULT_BASE_URL = "https://openai.vocareum.com/v1"
MODEL = "gpt-4o-mini"
TEMPERATURE = 0
TOP_P = 1.0


def _from_streamlit(name: str) -> str | None:
    try:
        import streamlit as st
        return st.secrets.get(name)  # raises when no secrets file exists
    except Exception:
        return None


def secret(name: str, default: str | None = None) -> str | None:
    return _from_streamlit(name) or os.environ.get(name) or default


def get_credentials() -> tuple[str | None, str]:
    return secret("OPENAI_API_KEY"), secret("OPENAI_API_BASE", DEFAULT_BASE_URL)


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def get_llm(http_async_client=None):
    """gpt-4o-mini through the OpenAI-compatible Vocareum endpoint.

    Pass an ``httpx.AsyncClient`` created inside the running event loop. Without it, langchain-openai reuses a
    process-wide async client bound to the first event loop, which fails with "Event loop is closed" later on.
    """
    from langchain_openai import ChatOpenAI
    key, base = get_credentials()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (Streamlit secrets or environment variable).")
    c = load_contract()
    from .cache import install_llm_cache
    install_llm_cache()                  # opt-in exact-match LLM cache (NG_LLM_CACHE=1, Redis or memory)
    extra = {"http_async_client": http_async_client} if http_async_client is not None else {}
    return ChatOpenAI(model=c["model"], temperature=c["temperature"], top_p=c.get("top_p", TOP_P), api_key=key, base_url=base,
                      timeout=c["llm_timeout_s"], max_retries=2, **extra)
