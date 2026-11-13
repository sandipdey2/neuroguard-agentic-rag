"""Skills with progressive disclosure.

At start-up only the YAML front matter (name, description, agents) of every skills/*/SKILL.md is read and
offered to the orchestrator as a catalog. When the orchestrator delegates to an agent it may select one of
that agent's skills; only then is the full Markdown body read from disk and appended to the agent's system
prompt. The trace records which skill was loaded and how many characters were injected.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import ROOT

SKILLS_DIR = ROOT / "skills"


def _front_matter(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path} has no front matter")
    meta = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    meta["agents"] = [a.strip() for a in meta.get("agents", "").split(",") if a.strip()]
    meta["path"] = str(path)
    return meta


@lru_cache(maxsize=1)
def catalog() -> dict[str, dict]:
    """name -> {name, description, agents, path}; bodies are NOT loaded here."""
    return {m["name"]: m for m in (_front_matter(p) for p in sorted(SKILLS_DIR.glob("*/SKILL.md")))}


def catalog_for(agent: str) -> list[dict]:
    return [{"name": m["name"], "description": m["description"]} for m in catalog().values() if agent in m["agents"]]


def load_body(name: str, agent: str) -> str:
    meta = catalog().get(name)
    if meta is None or agent not in meta["agents"]:
        raise KeyError(f"skill {name!r} is not available to {agent}")
    text = Path(meta["path"]).read_text(encoding="utf-8")
    return text.split("---", 2)[2].strip()
