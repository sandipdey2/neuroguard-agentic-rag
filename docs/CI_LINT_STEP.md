# Add the lint + type step to CI (manual, one paste)

v2.6.0 adds `ruff` (lint) and `mypy` (types) with their configs in `ruff.toml` and `mypy.ini`. Both pass locally on
Python 3.13 and 3.14. The automation token used for this repo cannot edit `.github/workflows/*` (no `workflow` scope),
so the step has to be pasted once in the GitHub web editor.

## How
1. Open https://github.com/sandipdey2/neuroguard-agentic-rag/edit/main/.github/workflows/ci.yml
2. Paste the two steps below **at the end of `steps:`**, after `MCP server over stdio`, with the same indentation
   (6 spaces before each `- name:`).
3. Commit to `main`. The next CI run shows `Lint (ruff)` and `Types (mypy)` for both Python versions.

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
Widening that scope is a listed improvement, not a v2.6.0 claim.

## Run locally
```bash
pip install -r requirements-dev.txt     # includes ruff==0.17.5 and mypy==2.4.1
ruff check .
mypy
```
