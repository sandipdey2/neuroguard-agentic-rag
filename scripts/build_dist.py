"""Build the clean product zip for Gumroad:  python scripts/build_dist.py  ->  dist/neuroguard-ai-<version>.zip

Includes code, skills, guideline corpus (public domain), MTSamples subset (CC0), docs, notebook, diagram and paper.
EXCLUDES the Stroke Prediction Dataset (educational-use terms) - buyers run scripts/fetch_data.py and must accept
its terms themselves - plus secrets, runtime caches, git metadata and virtual environments.
"""
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "2.0.1"
INCLUDE = ["app.py", "harness_contract.json", "requirements.txt", "requirements-dev.txt", "README.md", "LICENSE", "CHANGELOG.md",
           "DATA_SOURCES.md", "module_summary.md", "module_summary.pdf", "architecture_diagram.png", "architecture_diagram.pdf",
           "evaluation.ipynb", ".streamlit/config.toml", ".streamlit/secrets.toml.example", ".gitignore",
           "data/guidelines.json", "data/raw/mtsamples_subset.csv"]
INCLUDE_DIRS = ["core", "mcp_server", "skills", "scripts", "docs", "eval"]
EXCLUDE_PARTS = {"__pycache__", "runtime", ".git", "dist", ".ipynb_checkpoints"}
EXCLUDE_NAMES = {"secrets.toml", "healthcare-dataset-stroke-data.csv", ".env"}


def files():
    for f in INCLUDE:
        yield ROOT / f
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & EXCLUDE_PARTS) and p.name not in EXCLUDE_NAMES and p.suffix != ".pyc":
                yield p


def main():
    out = ROOT / "dist" / f"neuroguard-ai-{VERSION}.zip"
    out.parent.mkdir(exist_ok=True)
    manifest = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files():
            rel = p.relative_to(ROOT).as_posix()
            data = p.read_bytes()
            for marker in ("voc" + "-1", "sk" + "-proj", "gh" + "p_", "github" + "_pat_"):  # split so this file does not match itself
                assert marker.encode() not in data, f"possible secret in {rel}"
            z.writestr(f"neuroguard-ai/{rel}", data)
            manifest.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
        z.writestr("neuroguard-ai/MANIFEST.sha256", "\n".join(manifest) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB, {len(manifest)} files)")


if __name__ == "__main__":
    main()
