"""MCP client: launches the NeuroGuard server over stdio with the official SDK and wraps it for the agents.

The server runs as a child process started with ``sys.executable`` (the same interpreter as the app, so it
works on Streamlit Community Cloud). The OpenAI key and other secrets are removed from its environment.
``load_mcp_tools`` (langchain-mcp-adapters) turns MCP tool schemas into LangChain tools for bind_tools;
resources and prompts are read directly through the ClientSession.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .memory import db_path

SERVER = Path(__file__).resolve().parent.parent / "mcp_server" / "neuroguard_mcp_server.py"
_SECRET_PREFIXES = ("OPENAI", "NG_RECEIPT", "GH_", "GITHUB", "AWS", "ANTHROPIC")


def server_params() -> StdioServerParameters:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_SECRET_PREFIXES)}
    env["NG_MEMORY_DB"] = str(db_path())
    return StdioServerParameters(command=sys.executable, args=[str(SERVER)], env=env)


@asynccontextmanager
async def mcp_session():
    async with stdio_client(server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session
