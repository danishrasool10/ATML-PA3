"""Kaggle orchestration for Task 1.  Every step is idempotent and deadline-aware: finished work (in this
session's outputs or earlier sessions' outputs attached under /kaggle/input) is skipped, and nothing is
started that cannot finish before the session budget, so the notebook always ends cleanly and Kaggle
saves its outputs.  Training pauses with a recovery checkpoint instead of being killed.

    AUTO  (default) everything in one session, as far as the budget allows; rerun to continue:
          prefetch -> data/probe/bases/z-cache -> preflight (NCCL, FP16, worst-case memory + ZeRO-1
          fallback) -> Full FT -> eval baseline | FullFT-mid -> L2-SP -> eval FullFT-final | L2-SP
          -> LoRA-4 -> LoRA-16 -> eval LoRA-4 | LoRA-16 -> OPLoRA -> eval OPLoRA | diagnostics
          -> qualitative sheet -> report assets
    S0    extended smoke test (CPU test suite, 20-update runs of every method, lm-eval smoke)
    S1/S2/S3  the same pipeline split into three sessions
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

from pa3common.kaggle import (Deadline, StagePaused, discover_prior_roots, disk_report, is_oom, log_has,
                              n_gpus, run_on_gpus, scratch_root, sh)

TASK = Path(__file__).resolve().parents[2]  # .../task_1
ROOT = TASK.parent  # repository root
COMMON = ROOT / "common"
CFG = TASK / "configs"
MODEL_ID = "Qwen/Qwen3-0.6B"
RUNS = ["full_ft", "full_ft_l2sp", "lora_r4", "lora_r16", "oplora_r16"]
DEADLINE: Deadline | None = None
FINALIZE_RESERVE = 900  # seconds kept free at the end (persist recovery, report, Kaggle output upload)


# ------------------------------------------------------------------ environment
def out_root() -> Path:
    return Path(os.environ.setdefault("PA3_OUTPUT_ROOT", "/kaggle/working/pa3_outputs"))


def setup_env(output_root: str = "/kaggle/working/pa3_outputs", budget_hours: float = 11.25) -> dict:
    global DEADLINE
    os.environ["PA3_OUTPUT_ROOT"] = output_root
    Path(output_root).mkdir(parents=True, exist_ok=True)
    priors = [r for r in discover_prior_roots("pa3_outputs") if Path(r).resolve() != Path(output_root).resolve()]
    os.environ["PA3_PRIOR_ROOTS"] = os.pathsep.join(priors)
    paths = [str(TASK / "src"), str(COMMON)]
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.update({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_DOWNLOAD_TIMEOUT": "120",
                       "HF_HUB_ETAG_TIMEOUT": "60", "PYTHONUNBUFFERED": "1"})
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    scratch = scratch_root()
    for k, v in decisions().get("env", {}).items():
        os.environ[k] = v
    DEADLINE = Deadline(budget_hours)
    q = locate_questions()
    info = {"output_root": output_root, "scratch": str(scratch), "prior_roots": priors,
            "questions_yaml": str(q) if q else None, "gpus": n_gpus(), "budget": str(DEADLINE)}
    print(json.dumps(info, indent=1), flush=True)
    return info


def locate_questions() -> Path | None:
    dst = out_root() / "first_plot_questions.yaml"
    if dst.exists():
        return dst
    cands = sorted(Path("/kaggle/input").glob("**/first_plot_questions.yaml")) if Path("/kaggle/input").exists() else []
    cands += [ROOT / "first_plot_questions.yaml", TASK / "first_plot_questions.yaml"]
    cands += [Path(r) / "first_plot_questions.yaml" for r in os.environ.get("PA3_PRIOR_ROOTS", "").split(os.pathsep) if r]
    for c in cands:
        if c.exists():
            shutil.copy(c, dst)
            return dst
    return None


def find(rel: str) -> Path | None:
    from pa3.utils.common import find_artifact

    return find_artifact(rel)


def decisions() -> dict:
    p = out_root() / "decisions.json"
    if p.exists():
        return json.loads(p.read_text())
    for r in os.environ.get("PA3_PRIOR_ROOTS", "").split(os.pathsep):
        if r and (Path(r) / "decisions.json").exists():
            d = json.loads((Path(r) / "decisions.json").read_text())
            p.write_text(json.dumps(d, indent=1))
            return d
    return {}


def set_decision(key: str, value) -> None:
    d = decisions()
    d[key] = value
    (out_root() / "decisions.json").write_text(json.dumps(d, indent=1))


def logs() -> Path:
    return out_root() / "logs"


def py(*args) -> list:
    return [sys.executable, *args]


def _need(seconds: float, what: str) -> None:
    if DEADLINE is not None and not DEADLINE.fits(seconds, FINALIZE_RESERVE):
        raise StagePaused(f"not enough session time left for {what} (~{seconds / 60:.0f} min needed; {DEADLINE})")


# ------------------------------------------------------------------ common preparation
def prefetch() -> None:
    if decisions().get("prefetched") and find("data/report.json"):
        return
    rc = sh(py("-m", "pa3.prefetch", *(["--skip_train_data"] if find("data/report.json") else [])),
            logs() / "prefetch.log", cwd=TASK)
    assert rc == 0, "downloads failed (check internet is ON in the Kaggle settings); see logs/prefetch.log"
    set_decision("prefetched", True)


def ensure_common() -> None:
    prefetch()
    if not find("data/report.json"):
        rc = sh(py("-m", "pa3.data.prepare", "--out_dir", out_root() / "data"), logs() / "prepare.log", cwd=TASK)
        assert rc == 0, "data preparation failed; see logs/prepare.log"
    if not find("data/probe_ids.json"):
        sh(py("-m", "pa3.diagnostics.probe_set"), logs() / "probe.log", cwd=TASK)
    if not find("bases/oplora_bases_k128.safetensors"):
        rc = sh(py(TASK / "scripts" / "compute_bases.py"), logs() / "bases.log", cwd=TASK)
        assert rc == 0, "OPLoRA basis computation failed; see logs/bases.log"
    if not find("diagnostics/z_cache.pt"):
        rc = sh(py("-m", "pa3.diagnostics.run", "cache"), logs() / "zcache.log", cwd=TASK)
        assert rc == 0, "activation cache failed; see logs/zcache.log"


def nccl_check() -> None:
    if n_gpus() < 2 or "nccl_ok" in decisions():
        return
    cmd = py("-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", COMMON / "pa3common" / "nccl_check.py")
    rc = sh(cmd, logs() / "nccl_check.log", timeout=240)
    if rc != 0:
        print("[preflight] NCCL check failed/hung -> retry with NCCL_P2P_DISABLE=1", flush=True)
        os.environ["NCCL_P2P_DISABLE"] = "1"
        rc = sh(cmd, logs() / "nccl_check.log", timeout=240)
        assert rc == 0, "2-GPU communication does not work on this machine; see logs/nccl_check.log"
        set_decision("env", {**decisions().get("env", {}), "NCCL_P2P_DISABLE": "1"})
    set_decision("nccl_ok", True)


def preflight() -> None:
    """~10 min: NCCL, FP16 numerics, and one worst-case update (32 longest examples) for the two
    memory-heavy runs, choosing ZeRO-1 (approved fallback) before the real runs if needed."""
    nccl_check()
    if "fp16_ok" not in decisions():
        sh(py("-m", "pa3.smoke", "fp16"), logs() / "fp16.log", cwd=TASK)
        p = out_root() / "smoke" / "fp16_sanity.json"
        res = json.loads(p.read_text()) if p.exists() else {"ok": False}
        assert res.get("ok"), f"FP16 forward of Qwen3 is not sane on this GPU: {res}"
        set_decision("fp16_ok", True)
    for run in ("full_ft", "full_ft_l2sp"):
        key = f"{run}_preflight"
        if key in decisions() or find(f"runs/{run}/DONE"):
            continue
        for attempt in ("ddp", "zero1", "zero1+host_ref"):
            if attempt == "zero1":
                set_decision(f"{run}_zero1", True)
            if attempt == "zero1+host_ref":
                if run != "full_ft_l2sp":
                    break
                set_decision(f"{run}_ref_device", "cpu")
            name = f"{run}_preflight"
            rc = _torchrun(run, name, ["max_updates=1", "extra={smoke_longest: true}"], timeout=1800)
            log = logs() / f"train_{name}.log"
            if rc == 0:
                res = json.loads((out_root() / "runs" / name / "smoke_resources.json").read_text())
                set_decision(key, {"mode": attempt, "peak": res["peak_memory_per_rank"]})
                print(f"[preflight] {run}: worst-case update OK with {attempt}: {res['peak_memory_per_rank']}")
                break
            if not is_oom(log):
                raise RuntimeError(f"preflight of {run} failed (not OOM); see {log}")
            print(f"[preflight] {run}: OOM with {attempt}", flush=True)
        else:
            raise RuntimeError(f"{run} does not fit in T4 memory even with ZeRO-1")
        shutil.rmtree(out_root() / "runs" / f"{run}_preflight", ignore_errors=True)


# ------------------------------------------------------------------ training
def _run_overrides(run: str) -> list[str]:
    d = decisions()
    sets = []
    if run in ("full_ft", "full_ft_l2sp"):
        sets.append(f"zero1={str(bool(d.get(f'{run}_zero1', False))).lower()}")
    if run == "full_ft_l2sp":
        sets.append(f"l2sp_ref_device={d.get(f'{run}_ref_device', 'cuda')}")
    return sets


def _torchrun(run: str, name: str, sets: list[str], timeout: float | None = None) -> int:
    nproc = max(1, n_gpus())
    cmd = py("-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", "-m", "pa3.train.loop",
             "--config", CFG / "base.yaml", CFG / f"{run}.yaml", "--set", f"run_name={name}",
             *_run_overrides(run), *sets)
    return sh(cmd, logs() / f"train_{name}.log", timeout=timeout, cwd=TASK)


def _scratch_runs() -> Path:
    return Path(os.environ["PA3_SCRATCH"]) / "runs"


def _complete(rec_root: Path) -> list[Path]:
    r = rec_root / "recovery"
    return sorted(p for p in r.iterdir() if (p / "COMPLETE").exists()) if r.exists() else []


def _restore(run: str) -> None:
    """Bring an unfinished run from an earlier session back: run dir -> outputs, recovery -> scratch."""
    local = out_root() / "runs" / run
    for r in os.environ.get("PA3_PRIOR_ROOTS", "").split(os.pathsep):
        src = Path(r) / "runs" / run if r else None
        if src is None or not src.exists() or (src / "DONE").exists():
            continue
        if not local.exists():
            print(f"[resume] copying {run} logs/snapshots from {src}", flush=True)
            shutil.copytree(src, local, ignore=shutil.ignore_patterns("recovery"))
        if not _complete(_scratch_runs() / run) and _complete(src):
            dst = _scratch_runs() / run / "recovery" / _complete(src)[-1].name
            print(f"[resume] restoring recovery checkpoint {dst.name} for {run}", flush=True)
            shutil.copytree(_complete(src)[-1], dst)
        break
    if not _complete(_scratch_runs() / run) and _complete(local):  # paused earlier in a reused session
        shutil.copytree(_complete(local)[-1], _scratch_runs() / run / "recovery" / _complete(local)[-1].name)


def _persist_recovery(run: str) -> None:
    comp = _complete(_scratch_runs() / run)
    if not comp:
        return
    dst_root = out_root() / "runs" / run / "recovery"
    shutil.rmtree(dst_root, ignore_errors=True)
    shutil.copytree(comp[-1], dst_root / comp[-1].name)
    print(f"[pause] recovery {comp[-1].name} of {run} saved to outputs for the next session", flush=True)


def train(run: str, max_attempts: int = 3) -> None:
    if find(f"runs/{run}/DONE"):
        print(f"[train] {run}: done; skipping", flush=True)
        return
    _need(1200, f"training {run}")
    _restore(run)
    sets = [f"recovery_root={_scratch_runs()}",
            f"deadline_unix={DEADLINE.end - FINALIZE_RESERVE if DEADLINE else 'null'}"]
    for attempt in range(1, max_attempts + 1):
        print(f"[train] {run}: attempt {attempt} ({DEADLINE})", flush=True)
        rc = _torchrun(run, run, sets)
        run_dir = out_root() / "runs" / run
        if (run_dir / "DONE").exists():
            shutil.rmtree(_scratch_runs() / run, ignore_errors=True)
            shutil.rmtree(run_dir / "recovery", ignore_errors=True)
            print(f"[train] {run}: finished", flush=True)
            return
        if (run_dir / "PAUSED").exists():
            _persist_recovery(run)
            raise StagePaused(f"{run} paused before the session deadline")
        log = logs() / f"train_{run}.log"
        if is_oom(log) and run in ("full_ft", "full_ft_l2sp") and not decisions().get(f"{run}_zero1"):
            print(f"[train] {run}: OOM -> enabling ZeRO-1 (approved fallback) and resuming", flush=True)
            set_decision(f"{run}_zero1", True)
            shutil.rmtree(_scratch_runs() / run / "recovery", ignore_errors=True)  # optimizer format changes
        elif log_has(log, "consecutive skipped updates"):
            raise RuntimeError(f"{run}: persistent FP16 overflow; see {log}")
        print(f"[train] {run}: exit code {rc}; resuming from the latest recovery checkpoint", flush=True)
        _need(1200, f"retrying {run}")
    raise RuntimeError(f"{run} failed {max_attempts} times; see {logs() / f'train_{run}.log'}")


# ------------------------------------------------------------------ evaluation
def _eval_done(c: str) -> bool:
    if not find(f"eval/{c}/aggregate.json"):
        return False
    agg = json.loads(find(f"eval/{c}/aggregate.json").read_text())
    full = all(v.get("ok") for v in agg.get("split_check", {}).values()) and len(agg.get("split_check", {})) == 4
    return full and (not locate_questions() or bool(find(f"qualitative/per_condition/{c}.jsonl")))


def _eval_estimate() -> float:
    ts = []
    for p in (out_root() / "eval").glob("*/timing.json") if (out_root() / "eval").exists() else []:
        t = json.loads(p.read_text())
        tot = sum(t.get(k, 0) for k in ("build_seconds", "lm_eval_seconds", "qualitative_seconds", "merge_check_seconds"))
        if t.get("lm_eval_seconds"):
            ts.append(tot)
    return 1.25 * statistics.median(ts) if ts else 3000.0


def _eval_cmd(c: str, gen_batch: int, ll_batch: int, limit: int | None = None) -> list:
    cmd = py("-m", "pa3.eval.run", "--condition", c, "--gen_batch", str(gen_batch), "--ll_batch", str(ll_batch))
    if limit:
        cmd += ["--limit", str(limit)]
    q = locate_questions()
    if q:
        cmd += ["--questions", str(q)]
    return cmd


def evaluate(conds: list[str], extra_jobs: list | None = None, limit: int | None = None) -> None:
    """Evaluate conditions concurrently (one per GPU), with OOM back-off and one generic retry."""
    todo = [c for c in conds if limit or not _eval_done(c)]
    for c in conds:
        if c not in todo:
            print(f"[eval] {c}: done; skipping", flush=True)
    if not todo and not extra_jobs:
        return
    batches = {c: [64, 32] for c in todo}
    tries = {c: 0 for c in todo}
    extra_jobs = list(extra_jobs or [])
    while todo or extra_jobs:
        _need(_eval_estimate(), f"evaluating {todo}")
        jobs = [(c, _eval_cmd(c, *batches[c], limit), logs() / f"eval_{c}{'_smoke' if limit else ''}.log") for c in todo]
        jobs += extra_jobs
        extra_jobs = []
        cwd = os.getcwd()
        os.chdir(TASK)
        try:
            rcs = run_on_gpus(jobs, kill_at=DEADLINE.end - FINALIZE_RESERVE / 2 if DEADLINE else None)
        finally:
            os.chdir(cwd)
        if DEADLINE is not None and DEADLINE.remaining() < FINALIZE_RESERVE:
            raise StagePaused("session deadline reached during evaluation (finished parts are cached)")
        nxt = []
        for c in todo:
            if rcs.get(c) == 0 and (limit or _eval_done(c)):
                continue
            tries[c] += 1
            log = logs() / f"eval_{c}{'_smoke' if limit else ''}.log"
            if is_oom(log) and batches[c][0] > 8:
                batches[c] = [max(8, batches[c][0] // 2), max(4, batches[c][1] // 2)]
                print(f"[eval] {c}: OOM -> retry with batch sizes {batches[c]}", flush=True)
            elif tries[c] > 2:
                raise RuntimeError(f"evaluation of {c} failed repeatedly; see {log}")
            nxt.append(c)
        todo = nxt


def diagnostics_job() -> tuple[str, list, Path]:
    return ("diagnostics", py("-m", "pa3.diagnostics.run", "analyze"), logs() / "diagnostics.log")


def diagnostics_done() -> bool:
    return bool(find("diagnostics/weights.csv")) and bool(find("diagnostics/oplora_checks.json"))


# ------------------------------------------------------------------ qualitative + report
def collect_qualitative() -> None:
    from pa3.eval.conditions import ORDER
    from pa3.eval.qualitative import annotation_subset, load_questions, write_annotation_sheet, write_grouped_markdown

    q = locate_questions()
    if not q:
        print("[qualitative] questions yaml missing", flush=True)
        return
    qs = load_questions(q)
    rows = []
    for c in ORDER:
        p = find(f"qualitative/per_condition/{c}.jsonl")
        if p:
            rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    order_q = {qq["id"]: i for i, qq in enumerate(qs)}
    rows.sort(key=lambda r: (order_q.get(r["question_id"], 999), ORDER.index(r["condition"])))
    d = out_root() / "qualitative"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "responses.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_grouped_markdown(rows, qs, ORDER, d / "responses_grouped.md")
    sub = annotation_subset(rows, qs, ORDER)
    write_annotation_sheet(sub, d / "annotation_sheet.csv")
    print(f"[qualitative] {len(rows)} responses; annotation rows: {len(sub)}"
          + ("" if len(rows) == 168 and len(sub) == 42 else "  (incomplete — expected 168 / 42)"), flush=True)


def report() -> None:
    sh(py(TASK / "scripts" / "make_report_assets.py"), logs() / "report.log", cwd=TASK)


def status() -> dict:
    from pa3.eval.conditions import ORDER

    st = {
        "runs": {r: ("done" if find(f"runs/{r}/DONE") else
                     "paused" if find(f"runs/{r}/PAUSED") else "todo") for r in RUNS},
        "evals": {c: ("done" if _eval_done(c) else "todo") for c in ORDER},
        "diagnostics": "done" if diagnostics_done() else "todo",
        "qualitative_sheet": "done" if find("qualitative/annotation_sheet.csv") else "todo",
        "session_hours_used": round((time.time() - DEADLINE.start) / 3600, 2) if DEADLINE else None,
    }
    st["complete"] = all(v == "done" for v in st["runs"].values()) and all(v == "done" for v in st["evals"].values()) \
        and st["diagnostics"] == "done"
    (out_root() / "status.json").write_text(json.dumps(st, indent=1))
    return st


def _finish(paused: StagePaused | None) -> dict:
    try:
        collect_qualitative()
        report()
    except Exception as ex:  # noqa: BLE001  (reporting must never lose the session's outputs)
        print(f"[finish] report step failed: {ex}", flush=True)
    disk_report()
    st = status()
    print("\n================ STATUS ================\n" + json.dumps(st, indent=1), flush=True)
    if paused is not None:
        print(f"\n>>> Session budget reached: {paused}\n>>> Save Version, make the output a Dataset, attach it,"
              " and run the notebook again — it continues exactly where it stopped.", flush=True)
    elif st["complete"]:
        print("\n>>> ALL DONE. Everything for Task 1 is in pa3_outputs/.", flush=True)
    return st


# ------------------------------------------------------------------ stages
def _pipeline(steps) -> dict:
    paused = None
    try:
        for step in steps:
            step()
    except StagePaused as ex:
        paused = ex
    return _finish(paused)


def stage_auto() -> dict:
    return _pipeline([
        ensure_common, preflight,
        lambda: train("full_ft"),
        lambda: evaluate(["baseline", "fullft_u0625"]),
        lambda: train("full_ft_l2sp"),
        lambda: evaluate(["fullft_u1250", "l2sp_u1250"]),
        lambda: train("lora_r4"),
        lambda: train("lora_r16"),
        lambda: evaluate(["lora4_u1250", "lora16_u1250"]),
        lambda: train("oplora_r16"),
        lambda: evaluate(["oplora16_u1250"], extra_jobs=[] if diagnostics_done() else [diagnostics_job()]),
        _catch_up,
    ])


def _catch_up() -> None:
    from pa3.eval.conditions import ORDER

    missing = [c for c in ORDER if not _eval_done(c)]
    if missing:
        evaluate(missing)
    if not diagnostics_done():
        _need(1800, "diagnostics")
        sh(py("-m", "pa3.diagnostics.run", "analyze"), logs() / "diagnostics.log", cwd=TASK,
           timeout=DEADLINE.remaining() - FINALIZE_RESERVE if DEADLINE else None)


def stage_s1() -> dict:
    return _pipeline([ensure_common, preflight, lambda: train("full_ft"),
                      lambda: evaluate(["baseline", "fullft_u0625"]), lambda: evaluate(["fullft_u1250"])])


def stage_s2() -> dict:
    return _pipeline([ensure_common, preflight, lambda: train("full_ft_l2sp"), lambda: train("lora_r4"),
                      lambda: evaluate(["l2sp_u1250", "lora4_u1250"])])


def stage_s3() -> dict:
    return _pipeline([ensure_common, lambda: train("lora_r16"), lambda: train("oplora_r16"),
                      lambda: evaluate(["lora16_u1250", "oplora16_u1250"]), _catch_up])


def stage_s0(cpu_tests: bool = True, smoke_updates: int = 20) -> dict:
    """Extended smoke test (optional; AUTO already runs the essential preflight)."""
    import torch

    sm = out_root() / "smoke"
    sm.mkdir(parents=True, exist_ok=True)
    summary: dict = {"env": {"torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(n_gpus())]}}
    if cpu_tests:
        summary["cpu_tests_rc"] = sh(py("-m", "pytest", "-q", "-p", "no:cacheprovider", TASK / "tests"),
                                     sm / "cpu_tests.log", env={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
                                     cwd=TASK)
    ensure_common()
    preflight()
    g1, g2 = sm / "grad_w1.pt", sm / "grad_w2.pt"
    common = ["lora_dropout=0.0", "grad_dump_update=0", "max_updates=1"]
    for path, nproc in ((g1, 1), (g2, max(1, n_gpus()))):
        cmd = py("-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", "-m", "pa3.train.loop",
                 "--config", CFG / "base.yaml", CFG / "lora_r16.yaml", "--set", f"run_name=gradcheck_w{nproc}",
                 *common, f"grad_dump_path={path}")
        sh(cmd, logs() / f"gradcheck_w{nproc}.log", cwd=TASK)
    if g1.exists() and g2.exists():
        sh(py("-m", "pa3.smoke", "compare", g1, g2), sm / "gradcmp.log", cwd=TASK)
        summary["grad_equivalence_gpu"] = json.loads((sm / "grad_equivalence_gpu.json").read_text())
    smoke = {}
    for run in RUNS:
        name = f"{run}_smoke"
        rc = _torchrun(run, name, [f"max_updates={smoke_updates}", "extra={smoke_longest: true, smoke_export: %s}"
                                   % ("true" if run == "oplora_r16" else "false")])
        info = {"rc": rc}
        rp, lp = out_root() / "runs" / name / "smoke_resources.json", out_root() / "runs" / name / "smoke_log.jsonl"
        if rp.exists():
            res = json.loads(rp.read_text())
            rows = [json.loads(x) for x in lp.read_text().splitlines()] if lp.exists() else []
            steady = [r["sec"] for r in rows[2:]] or [r["sec"] for r in rows]
            s_per = sum(steady) / max(1, len(steady))
            info.update({"peak_alloc_gib_per_rank": [round(m.get("peak_alloc_gib", 0), 2) for m in res["peak_memory_per_rank"]],
                         "sec_per_update_steady": round(s_per, 2), "projected_train_hours": round(s_per * 1250 / 3600, 2),
                         "tokens_per_second": round(res["tokens_per_second"]), "skipped_updates": res["skipped_updates"]})
        smoke[run] = info
        print(f"[S0] {run}: {json.dumps(info)}", flush=True)
    summary["smoke_runs"] = smoke
    if find(f"runs/oplora_r16_smoke/weights_u{smoke_updates:04d}/adapter.safetensors"):
        sh(py("-m", "pa3.smoke", "merge", "oplora_r16_smoke", str(smoke_updates)), sm / "merge.log", cwd=TASK)
    evaluate(["baseline"], limit=8)
    for d in list((out_root() / "runs").glob("*_smoke")) + list((out_root() / "runs").glob("gradcheck_w*")):
        shutil.rmtree(d / ("weights_u%04d" % smoke_updates), ignore_errors=True)
    (sm / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return summary
