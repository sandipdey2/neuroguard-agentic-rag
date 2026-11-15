# Lint + type steps in CI (applied)

As of commit b8fb7f7 (2026-11-15), `.github/workflows/ci.yml` runs `Lint (ruff)` and `Types (mypy)` after pytest and the MCP server check, on Python 3.13 and 3.14. First green run with these steps: https://github.com/sandipdey2/neuroguard-agentic-rag/actions/runs/43707597000.

The automation token used for this repo cannot edit `.github/workflows/*` (no `workflow` scope), so the steps were pasted once in the GitHub web editor. They are kept below as a reference of what is now in the workflow.

```yaml
      - name: Lint (ruff)
        run: |
          pip install ruff==0.17.5
          ruff check .
      - name: Types (mypy)
        run: |
          pip install mypy==2.4.1
          mypy
```

`mypy` with no arguments reads `mypy.ini`, which scopes the check to the harness / gate / receipt / learning core
(`core/harness.py`, `core/pipeline.py`, `core/episodes.py`, `core/config.py`, `core/permissions.py`,
`core/confidence.py`, `core/monitoring.py`) and `scripts/garden.py`, with `ignore_missing_imports = True`.
Widening that scope is a listed improvement, not a v2.6.1 claim.

## Run locally
```bash
pip install -r requirements-dev.txt     # includes ruff==0.17.5 and mypy==2.4.1
ruff check .
mypy
```
