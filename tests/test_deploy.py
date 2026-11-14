"""Deployment manifests: YAML parses, vLLM is optional (profile / not in the main set), secrets are placeholders only."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_compose_services_and_optional_vllm():
    c = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    s = c["services"]
    assert {"api", "worker", "ui", "slack-bot", "postgres", "redis", "vllm"} <= set(s)
    assert s["vllm"]["profiles"] == ["vllm"] and s["slack-bot"]["profiles"] == ["slack"]
    assert "vocareum" in (ROOT / "docker-compose.yml").read_text()


def test_k8s_manifests_parse_and_secret_is_template():
    kinds = []
    for f in sorted((ROOT / "deploy" / "k8s").glob("*.yaml")):
        kinds += [d["kind"] for d in yaml.safe_load_all(f.read_text()) if d]
    assert {"Deployment", "Service", "ConfigMap", "Secret", "HorizontalPodAutoscaler", "StatefulSet", "Ingress"} <= set(kinds)
    sec = yaml.safe_load((ROOT / "deploy" / "k8s" / "01-secret.template.yaml").read_text())
    assert all("PASTE_" in v for v in sec["stringData"].values())
    main_text = "".join(f.read_text() for f in (ROOT / "deploy" / "k8s").glob("*.yaml"))
    assert "vllm/vllm-openai" not in main_text                      # vLLM only under deploy/k8s/optional/


def test_env_example_has_placeholders_only():
    for line in (ROOT / ".env.example").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            assert "PASTE_" in v or v.startswith("https://"), k
