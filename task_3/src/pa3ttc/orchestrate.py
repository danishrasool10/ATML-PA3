"""Kaggle pipeline for Task 3 (failure-tolerant, resumable).

data -> splits -> [live generation starts in the background on GPU0, pinned to its own CPU cores] ->
scorer -> signals -> offline (Steps 1-2) -> analysis (Step 3) -> adaptive (Step 5) -> compute (Step 6) ->
wait for live -> live_eval (Step 4) -> qualitative -> report.

Offline stages run as subprocesses pinned to the other cores (approved decision 7) so the generation timing is not
disturbed.  A stage is skipped when its _stages/<stage>.json marker exists (also from an earlier session attached
under /kaggle/input), except `data`, which is rebuilt whenever the scratch cache is missing.  A failed stage is
recorded in status.json and only the stages that depend on it are skipped.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pa3common.kaggle import Deadline, copytree_merge, discover_prior_roots, n_gpus, scratch_root, sh

from .io import write_json

TASK = Path(__file__).resolve().parents[2]
COMMON = TASK.parent / "common"
PY = sys.executable
OUT_NAME = "pa3_task3_outputs"
STATE: dict = {}
DEPS = {"data": [], "splits": ["data"], "scorer": ["data"], "signals": ["splits"], "offline": ["scorer", "signals"],
        "analysis": ["offline"], "adaptive": ["offline"], "compute": ["offline"], "live_eval": ["splits"],
        "qualitative": ["offline"], "report": ["data"]}


def _env(cores: list[int] | None = None) -> dict:
    e = {"PYTHONPATH": os.pathsep.join([str(TASK / "src"), str(COMMON)]), "TOKENIZERS_PARALLELISM": "false"}
    if cores:
        n = str(len(cores))
        e.update({"PA3_CPUS": ",".join(map(str, cores)), "OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n,
                  "OPENBLAS_NUM_THREADS": n})
    return e


def setup_env(output_root: str, budget_hours: float = 11.25) -> dict:
    R = Path(output_root)
    R.mkdir(parents=True, exist_ok=True)
    os.environ["PA3_OUTPUT_ROOT"] = str(R)
    os.environ.update(_env())
    dl = Deadline(budget_hours)
    scratch = scratch_root("pa3_task3_scratch")
    prior = [p for p in discover_prior_roots(OUT_NAME) if Path(p).resolve() != R.resolve()]
    for p in prior:
        print(f"[setup] merging earlier-session outputs from {p}", flush=True)
        copytree_merge(Path(p), R, skip=("report_assets",))
    cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    gpus = n_gpus()
    gen = cores[: max(1, len(cores) // 2)] if gpus and len(cores) >= 2 else []
    off = [c for c in cores if c not in gen] or cores
    STATE.update(deadline=dl, root=R, scratch=scratch, logs=R / "logs", steps={}, gen_cores=gen, off_cores=off,
                 gpus=gpus)
    info = {"output_root": str(R), "scratch": str(scratch), "gpus": gpus, "prior_roots": prior, "budget": str(dl),
            "generation_cores": gen, "offline_cores": off}
    write_json(R / "env.json", info)
    print(json.dumps(info, indent=1), flush=True)
    return info


def _status(step: str, state: str, **kw) -> None:
    STATE["steps"][step] = {"state": state, "time": time.strftime("%H:%M:%S"), **kw}
    write_json(STATE["root"] / "status.json", {"steps": STATE["steps"], "deadline": str(STATE["deadline"])})


def _marker(name: str) -> Path:
    return STATE["root"] / "_stages" / f"{name}.json"


def _cache_ok() -> bool:
    return all((STATE["scratch"] / "cache" / ds / "tokens.npz").exists() for ds in ("math", "mathvista"))


def _stage(name: str) -> bool:
    failed = [d for d in DEPS[name] if STATE["steps"].get(d, {}).get("state") in ("failed", "skipped")]
    if failed:
        _status(name, "skipped", reason=f"dependency failed or skipped: {failed}")
        return False
    if _marker(name).exists() and (name != "data" or _cache_ok()):
        _status(name, "done", note="already done")
        return True
    if not STATE["deadline"].fits(300, margin=120):
        _status(name, "skipped", reason="session budget")
        return False
    t0 = time.time()
    rc = sh([PY, "-m", "pa3ttc.stages", name], STATE["logs"] / f"{name}.log", env=_env(STATE["off_cores"]), cwd=TASK)
    ok = rc == 0 and _marker(name).exists()
    _status(name, "done" if ok else "failed", seconds=round(time.time() - t0), rc=rc)
    if not ok:
        print(f"!! stage {name} failed (rc={rc}); see {STATE['logs'] / (name + '.log')}", flush=True)
    return ok


def _launch_live():
    log = STATE["logs"] / "live_generation.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "a")
    env = {**os.environ, **_env(STATE["gen_cores"]), "CUDA_VISIBLE_DEVICES": "0"}
    p = subprocess.Popen([PY, "-m", "pa3ttc.live.run"], stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=TASK)
    print(f"[live] generation started in the background on GPU0 (pid {p.pid}, cores {STATE['gen_cores']}, log {log})",
          flush=True)
    _status("live_generation", "running", pid=p.pid)
    return p, fh


def _wait_live(p, fh) -> int:
    kill_at = STATE["deadline"].end - 300
    last = 0.0
    while p.poll() is None:
        if time.time() > kill_at:
            p.kill()
            print("[live] session deadline reached -> killed (finished calls are kept; rerun to resume)", flush=True)
            break
        if time.time() - last > 600:
            tail = [l for l in Path(fh.name).read_text(errors="ignore").splitlines() if l.startswith("[live]")][-1:]
            print(f"[live] waiting... {tail}", flush=True)
            last = time.time()
        time.sleep(15)
    fh.close()
    rc = p.wait()
    state = {0: "done", 3: "paused"}.get(rc, "failed")
    _status("live_generation", state, rc=rc)
    return rc


def stage_auto(run_cpu_tests: bool = False, run_live: bool = True) -> dict:
    if run_cpu_tests:
        rc = sh([PY, "-m", "pytest", "-q", "-x", str(TASK / "tests")], STATE["logs"] / "cpu_tests.log", env=_env(),
                cwd=TASK)
        _status("cpu_tests", "done" if rc == 0 else "failed", rc=rc)
    if not _stage("data"):
        log = STATE["logs"] / "data.log"
        tail = log.read_text(errors="ignore").splitlines()[-60:] if log.exists() else []
        print("\n" + "!" * 100 + "\n!! DATA STAGE FAILED — nothing else can run. Last lines of logs/data.log:\n"
              + "\n".join(tail) + "\n" + "!" * 100, flush=True)
    _stage("splits")
    live = None
    live_done = (STATE["root"] / "live" / "status.json").exists() and json.loads(
        (STATE["root"] / "live" / "status.json").read_text()).get("state") == "done"
    if run_live and not live_done and STATE["steps"].get("splits", {}).get("state") == "done":
        if STATE["gpus"] >= 1:
            live = _launch_live()
        else:
            _status("live_generation", "skipped", reason="no GPU visible")
    for s in ("scorer", "signals", "offline", "analysis", "adaptive", "compute"):
        _stage(s)
    if live is not None:
        _wait_live(*live)
        if _marker("live_eval").exists():
            _marker("live_eval").unlink()       # re-evaluate with the new calls
    if (STATE["root"] / "live" / "responses.jsonl").exists():
        _stage("live_eval")
    for s in ("qualitative", "report"):
        if _marker(s).exists():
            _marker(s).unlink()                 # cheap; always rebuilt from the latest outputs
        _stage(s)
    states = {k: v["state"] for k, v in STATE["steps"].items()}
    failed = [k for k, v in states.items() if v == "failed"]
    state = "failed" if failed else ("done" if all(v == "done" for k, v in states.items() if k != "cpu_tests")
                                     else "incomplete")
    final = {"steps": states, "state": state, "failed": failed,
             "skipped": [k for k, v in states.items() if v == "skipped"]}
    _status("pipeline", state, failed=failed)
    return final
