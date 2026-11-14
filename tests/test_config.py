"""Configuration hygiene: only placeholders in the committed env / secrets templates."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_env_example_has_placeholders_only():
    for line in (ROOT / ".env.example").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            assert "PASTE_" in v or v.startswith("https://") or v.startswith("http://"), k


def test_streamlit_secrets_example_has_placeholders_only():
    text = (ROOT / ".streamlit" / "secrets.toml.example").read_text()
    assert "PASTE_" in text and "sk-" not in text
