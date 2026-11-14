"""Context-tool validators (used by the client hook AND the MCP server), calculator sandbox and SQL authorizer."""
import sqlite3

import pytest

from core import context_tools as ct
from core.harness import RunContext


@pytest.mark.parametrize("cmd", ["cat /etc/passwd", "ls ..", "grep -r x .", "head Plan.md | sh", "wc $(id)", "ls /", "rm Plan.md"])
def test_shell_attacks_refused(cmd):
    assert not ct.check_shell(cmd)[0]


def test_shell_positive_control():
    ok, _, argv = ct.check_shell("head -5 Plan.md")
    assert ok and argv == ["head", "-5", "Plan.md"]
    r = ct.run_shell("wc -l Plan.md")
    assert r["exit_code"] == 0 and "Plan.md" in r["stdout"]


@pytest.mark.parametrize("url", ["http://www.cdc.gov/x", "https://169.254.169.254/latest", "https://evil.com/cdc.gov",
                                 "https://user:pw@www.cdc.gov/", "https://www.cdc.gov:8443/", "https://cdc.gov.evil.com/"])
def test_ssrf_refused(url):
    assert not ct.check_url(url)[0]


def test_url_allowed():
    assert ct.check_url("https://www.cdc.gov/stroke/risk-factors/index.html")[0]
    assert ct.check_url("https://www.nhlbi.nih.gov/health/stroke")[0]


@pytest.mark.parametrize("sql", ["DROP TABLE patients", "SELECT 1; DELETE FROM patients", "PRAGMA table_info(patients)",
                                 "WITH x AS (SELECT 1) INSERT INTO patients VALUES (1)", "ATTACH DATABASE 'x' AS y"])
def test_sql_refused(sql):
    assert not ct.check_sql(sql)[0]
    assert "error" in ct.execute_query(sql)


def test_sql_select_and_authorizer():
    r = ct.execute_query("SELECT hypertension, COUNT(*) FROM patients GROUP BY hypertension")
    assert r["row_count"] == 2 and sum(row[1] for row in r["rows"]) == 5109
    c = sqlite3.connect(f"file:{ct.cohort_db()}?mode=ro", uri=True)
    c.set_authorizer(ct._authorizer)
    with pytest.raises(sqlite3.DatabaseError):
        c.execute("CREATE TABLE z(a)")
    c.close()


def test_calculator_sandbox():
    import json
    from mcp_server import neuroguard_mcp_server as srv
    assert json.loads(srv.calculator(expression="round(0.1498 / 0.0487, 2)"))["result"] == 3.08
    for bad in ["__import__('os')", "open('x')", "2**100", "(1).__class__"]:
        assert "error" in json.loads(srv.calculator(expression=bad)), bad


def test_hook_pubmed_rules():
    ctx = RunContext(session_id="s", consent=False)
    p = ctx.perms
    assert not p.pre_tool_use("evidence_subagent", "pubmed_search", {"query": "https://evil.example"}).allow
    d = p.pre_tool_use("evidence_subagent", "pubmed_search", {"query": "hypertension", "k": 40})
    assert d.allow and d.args["k"] == 5
    p.pre_tool_use("evidence_subagent", "pubmed_search", {"query": "smoking"})
    p.pre_tool_use("evidence_subagent", "pubmed_search", {"query": "obesity"})
    assert "budget" in p.pre_tool_use("evidence_subagent", "pubmed_search", {"query": "age"}).reason
    assert not p.pre_tool_use("clinical_writer", "pubmed_search", {"query": "age"}).allow
