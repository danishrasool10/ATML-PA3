"""The Kaggle orchestrator on synthetic data: stage subprocesses, background live worker (fake backend), CPU-core
split, markers/skip on rerun, and failure isolation (a failed stage skips only its dependants)."""
from __future__ import annotations

from pa3ttc.io import read_json


def test_stage_auto(env, monkeypatch):
    from pa3ttc import orchestrate as O

    monkeypatch.setenv("PA3_SESSION_START", str(__import__("time").time()))
    O.setup_env(str(env["out"]), budget_hours=1.0)
    O.STATE["gpus"] = 1                                   # pretend a GPU so the live worker is launched
    O.STATE["gen_cores"], O.STATE["off_cores"] = [0], [0]
    res = O.stage_auto()
    assert res["state"] == "done", res
    st = read_json(env["out"] / "status.json")["steps"]
    assert st["live_generation"]["state"] == "done" and st["live_eval"]["state"] == "done"
    assert (env["out"] / "report_assets" / "summary.md").exists()
    # rerun: finished stages are skipped, live calls are not regenerated
    n = len((env["out"] / "live" / "responses.jsonl").read_text().splitlines())
    O.STATE["steps"] = {}
    res = O.stage_auto()
    st = read_json(env["out"] / "status.json")["steps"]
    assert st["offline"].get("note") == "already done" and "live_generation" not in st
    assert len((env["out"] / "live" / "responses.jsonl").read_text().splitlines()) == n


def test_failure_isolation(env, monkeypatch):
    from pa3ttc import orchestrate as O

    O.setup_env(str(env["out"]), budget_hours=1.0)
    O.STATE["gpus"] = 0
    monkeypatch.setenv("PA3_TEST_FAIL_STAGE", "scorer")
    res = O.stage_auto()
    st = read_json(env["out"] / "status.json")["steps"]
    assert st["scorer"]["state"] == "failed" and st["signals"]["state"] == "done"
    assert st["offline"]["state"] == "skipped" and st["analysis"]["state"] == "skipped"
    assert st["live_generation"]["state"] == "skipped"
    assert res["state"] == "failed" and res["failed"] == ["scorer"] and "offline" in res["skipped"]
