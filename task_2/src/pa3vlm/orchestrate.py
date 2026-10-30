"""Kaggle pipeline for Task 2 (deadline-aware, resumable across sessions).

Order: [optional CPU tests] -> hub prefetch -> data selection + image download -> label audit ->
SigLIP feature cache (two shards, one per GPU; separate processes, timed separately) -> preflight (3 updates
per stage type for both runs: s/update, memory, FP16 sanity) -> training (GPU0: vt196, GPU1: vt49,
concurrently) -> final evaluation (one run per GPU) -> report assets.  The notebook's last cell zips the
outputs and shows a download link.

Each step is skipped when its outputs already exist (including outputs of an earlier session attached under
/kaggle/input, which are merged into the output root first).  Training pauses itself with a resumable
checkpoint before the session budget runs out; the notebook then ends normally so Kaggle keeps the outputs.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from pa3common.kaggle import Deadline, copytree_merge, discover_prior_roots, disk_report, n_gpus, run_on_gpus, \
    scratch_root, sh
from pa3common.utils import write_json

TASK = Path(__file__).resolve().parents[2]
COMMON = TASK.parent / "common"
PY = sys.executable
OUT_NAME = "pa3_task2_outputs"
STATE: dict = {}


def _env() -> dict:
    return {"PYTHONPATH": os.pathsep.join([str(TASK / "src"), str(COMMON)]), "TOKENIZERS_PARALLELISM": "false"}


def setup_env(output_root: str, budget_hours: float = 11.25) -> dict:
    R = Path(output_root)
    R.mkdir(parents=True, exist_ok=True)
    os.environ["PA3_OUTPUT_ROOT"] = str(R)
    os.environ.update(_env())
    dl = Deadline(budget_hours)
    scratch = scratch_root("pa3_task2_scratch")
    prior = [p for p in discover_prior_roots(OUT_NAME) if Path(p).resolve() != R.resolve()]
    for p in prior:
        print(f"[setup] merging earlier-session outputs from {p}", flush=True)
        copytree_merge(Path(p), R, skip=("report_assets",))
    os.environ["PA3_PRIOR_ROOTS"] = os.pathsep.join(prior)
    STATE.update(deadline=dl, root=R, scratch=scratch, logs=R / "logs", steps={})
    info = {"output_root": str(R), "scratch": str(scratch), "gpus": n_gpus(), "prior_roots": prior, "budget": str(dl)}
    print(json.dumps(info, indent=1), flush=True)
    return info


def _status(step: str, state: str, **kw) -> None:
    STATE["steps"][step] = {"state": state, "time": time.strftime("%H:%M:%S"), **kw}
    write_json(STATE["root"] / "status.json", {"steps": STATE["steps"], "deadline": str(STATE["deadline"])})


def _run(step: str, cmd: list, tries: int = 2, fatal: bool = True) -> bool:
    t0 = time.time()
    for i in range(tries):
        rc = sh(cmd, STATE["logs"] / f"{step}.log", env=_env(), cwd=TASK)
        if rc == 0:
            _status(step, "done", seconds=round(time.time() - t0))
            return True
        print(f"!! {step} failed (rc={rc}), attempt {i + 1}/{tries}", flush=True)
    _status(step, "failed", seconds=round(time.time() - t0))
    if fatal:
        raise RuntimeError(f"{step} failed — see {STATE['logs'] / (step + '.log')}")
    return False


def _features_done() -> bool:
    d = STATE["scratch"] / "features"
    keys = json.loads((STATE["root"] / "data" / "images.json").read_text())
    need = set(keys["train"]) | set(keys["val"])
    have: set = set()
    for m in d.glob("shard*of2.json"):
        have |= set(json.loads(m.read_text())["keys"])
    return need <= have


def _run_status(run: str) -> str:
    p = STATE["root"] / "runs" / run / "status.json"
    return json.loads(p.read_text())["state"] if p.exists() else "todo"


def stage_auto(run_cpu_tests: bool = False, run_49: bool = True) -> dict:
    R, dl = STATE["root"], STATE["deadline"]
    runs = ["vt196"] + (["vt49"] if run_49 else [])
    if run_cpu_tests:
        _run("cpu_tests", [PY, "-m", "pytest", "-q", "-x", str(TASK / "tests")], tries=1)
    _run("prefetch", [PY, "-m", "pa3vlm.prefetch"], tries=3)
    _run("data", [PY, "-m", "pa3vlm.data.splits"], tries=3)
    if not (R / "data" / "label_audit" / "summary.json").exists():
        _run("label_audit", [PY, str(TASK / "scripts" / "label_audit.py")], tries=1, fatal=False)
    disk_report()
    if not _features_done():
        t0 = time.time()
        jobs = [(f"features{i}", [PY, "-m", "pa3vlm.features.siglip_cache", "--shard", str(i), "--nshards", "2"],
                 STATE["logs"] / f"features{i}.log") for i in range(2)]
        rcs = run_on_gpus(jobs, env_extra=_env())
        assert all(v == 0 for v in rcs.values()) and _features_done(), f"feature caching failed: {rcs}"
        write_json(R / "features" / "cache_wall.json", {"wall_seconds": time.time() - t0, "n_gpus": n_gpus()})
        _status("features", "done", seconds=round(time.time() - t0))
    todo = [r for r in runs if _run_status(r) != "done"]
    pf = [(f"preflight_{r}", [PY, "-m", "pa3vlm.train.loop", "--config", r, "--preflight", "3"],
           STATE["logs"] / f"preflight_{r}.log") for r in todo
          if not (R / "preflight" / r / "preflight.json").exists()]
    if pf:
        rcs = run_on_gpus(pf, env_extra=_env())
        _status("preflight", "done" if all(v == 0 for v in rcs.values()) else "failed", rcs=rcs)
        assert all(v == 0 for v in rcs.values()), f"preflight failed: {rcs} (see logs/preflight_*.log)"
    for attempt in range(3):
        todo = [r for r in runs if _run_status(r) != "done"]
        if not todo:
            break
        jobs = [(f"train_{r}", [PY, "-m", "pa3vlm.train.loop", "--config", r], STATE["logs"] / f"train_{r}.log")
                for r in todo]
        rcs = run_on_gpus(jobs, env_extra=_env(), kill_at=dl.end - 300)
        _status("train", "running", attempt=attempt + 1, rcs=rcs)
        if any(v == 3 for v in rcs.values()) or not dl.fits(1800, margin=0):
            break
    states = {r: _run_status(r) for r in runs}
    _status("train", "done" if all(v == "done" for v in states.values()) else "paused", runs=states)
    ev = [(f"final_{r}", [PY, "-m", "pa3vlm.eval.final", "--config", r], STATE["logs"] / f"final_{r}.log")
          for r in runs if states[r] == "done" and not (R / "runs" / r / "final_eval" / "summary.json").exists()]
    if ev and dl.fits(900, margin=120):
        rcs = run_on_gpus(ev, env_extra=_env(), kill_at=dl.end - 240)
        _status("final_eval", "done" if all(v == 0 for v in rcs.values()) else "failed", rcs=rcs)
    if dl.fits(120, margin=60):
        _run("report", [PY, str(TASK / "scripts" / "make_report_assets.py")], tries=1, fatal=False)
    final = {"runs": states, "main_run_done": states.get("vt196") == "done",
             "state": "done" if all((R / "runs" / r / "final_eval" / "summary.json").exists() for r in runs)
             else "paused"}
    _status("pipeline", final["state"], **final)
    return final


def zip_results() -> Path:
    """results_task2.zip = everything in the output root except resumable checkpoints (manual use; the notebook's
    DOWNLOAD cell builds /kaggle/working/results_task2.zip itself)."""
    R = STATE["root"]
    z = R / "results_task2.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(R.rglob("*")):
            if f.is_file() and f != z and "checkpoints" not in f.parts and f.suffix != ".tmp":
                zf.write(f, f.relative_to(R))
    print(f"[zip] {z} ({z.stat().st_size / 2 ** 20:.1f} MiB)", flush=True)
    return z


def clean_scratch() -> None:
    shutil.rmtree(STATE["scratch"], ignore_errors=True)
