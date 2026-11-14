"""Scheduled batch CLI (cron trigger) and the pipeline's state-machine labels."""
import json


def test_scheduled_batch_cli(fake_run, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NG_QUEUE_DB", str(tmp_path / "sched.sqlite"))
    from core import job_queue
    job_queue.set_queue(None)
    from scripts.scheduled_batch import main
    assert main(["--rows", "9046", "31112"], run=fake_run) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 2 and all(x["status"] == "done" for x in lines)
    assert main(["--rows", "9046", "--enqueue-only"]) == 0
    assert main([]) == 2
    job_queue.set_queue(None)


def test_state_machine_roles():
    from core.agents import NODE_ROLES
    assert NODE_ROLES["planner"] == "planner" and NODE_ROLES["orchestrator"] == "executor"
    assert NODE_ROLES["safety_reviewer"] == NODE_ROLES["gate"] == "validator"
