"""Orchestrator control flow to the very end (heavy steps stubbed): status bookkeeping must not crash after
training/eval are done (regression: `_status() got multiple values for argument 'state'`), and a paused run
is reported as paused."""
from __future__ import annotations

import json


def _fake_env(tmp_path, monkeypatch, train_state: str):
    from pa3vlm import orchestrate as O

    R = tmp_path / "out"
    (R / "data" / "label_audit").mkdir(parents=True)
    (R / "data" / "label_audit" / "summary.json").write_text("{}")
    for r in ("vt196", "vt49"):
        (R / "runs" / r / "final_eval").mkdir(parents=True)
        (R / "runs" / r / "status.json").write_text(json.dumps({"state": train_state}))
        (R / "preflight" / r).mkdir(parents=True)
        (R / "preflight" / r / "preflight.json").write_text("{}")
        if train_state == "done":
            (R / "runs" / r / "final_eval" / "summary.json").write_text("{}")
    O.STATE.clear()
    O.STATE.update(root=R, scratch=tmp_path / "scratch", logs=R / "logs", steps={}, deadline=O.Deadline(5))
    calls = []
    monkeypatch.setattr(O, "_run", lambda step, cmd, tries=2, fatal=True: calls.append(step) or True)
    monkeypatch.setattr(O, "_features_done", lambda: True)
    monkeypatch.setattr(O, "disk_report", lambda *a, **k: 0.0)
    monkeypatch.setattr(O, "run_on_gpus", lambda jobs, **kw: calls.append([j[0] for j in jobs]) or
                        {j[0]: (3 if train_state == "paused" else 0) for j in jobs})
    return O, R, calls


def test_pipeline_finishes_and_writes_status(tmp_path, monkeypatch):
    O, R, calls = _fake_env(tmp_path, monkeypatch, "done")
    res = O.stage_auto()
    assert res["state"] == "done" and res["main_run_done"]
    st = json.loads((R / "status.json").read_text())
    assert st["steps"]["pipeline"]["state"] == "done" and st["steps"]["train"]["state"] == "done"
    assert "report" in calls


def test_pipeline_paused_status(tmp_path, monkeypatch):
    O, R, calls = _fake_env(tmp_path, monkeypatch, "paused")
    res = O.stage_auto()
    assert res["state"] == "paused"
    assert json.loads((R / "status.json").read_text())["steps"]["pipeline"]["state"] == "paused"
