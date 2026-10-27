"""Kaggle stage orchestration (S0 smoke, S1/S2/S3).  Every step is idempotent: finished work
(in this session's outputs or in earlier sessions' outputs attached under /kaggle/input) is skipped.

    S0  smoke: env report, CPU test suite, data prep, probe set, OPLoRA bases, FP16 sanity,
        GPU DDP-vs-single gradient equivalence (+NCCL fallback), 20-update runs of all five
        methods (memory/throughput/projections; L2-SP ZeRO-1 fallback), merge check, z cache,
        lm-eval + qualitative smoke on the baseline.
    S1  Full FT (2 GPUs) -> eval baseline | fullft_u0625 in parallel -> eval fullft_u1250
    S2  L2-SP -> LoRA-4 -> eval l2sp_u1250 | lora4_u1250
    S3  LoRA-16 -> OPLoRA-16 -> eval lora16 | oplora16 -> diagnostics -> qualitative sheet -> report
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "configs"


# ------------------------------------------------------------------ environment
def out_root() -> Path:
    return Path(os.environ.setdefault("PA3_OUTPUT_ROOT", "/kaggle/working/pa3_outputs"))


def discover_prior_roots(base: str = "/kaggle/input") -> list[str]:
    roots = []
    b = Path(base)
    if b.exists():
        for p in sorted(b.glob("**/pa3_outputs")):
            if p.is_dir() and len(p.relative_to(b).parts) <= 6:
                roots.append(str(p))
    return roots


def setup_env(output_root: str = "/kaggle/working/pa3_outputs") -> dict:
    os.environ["PA3_OUTPUT_ROOT"] = output_root
    Path(output_root).mkdir(parents=True, exist_ok=True)
    priors = [r for r in discover_prior_roots() if Path(r).resolve() != Path(output_root).resolve()]
    os.environ["PA3_PRIOR_ROOTS"] = os.pathsep.join(priors)
    os.environ["PYTHONPATH"] = str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    for k, v in decisions().get("env", {}).items():
        os.environ[k] = v
    q = locate_questions()
    info = {"output_root": output_root, "prior_roots": priors, "questions_yaml": str(q) if q else None}
    print(json.dumps(info, indent=1))
    return info


def locate_questions() -> Path | None:
    dst = out_root() / "first_plot_questions.yaml"
    if dst.exists():
        return dst
    cands = list(Path("/kaggle/input").glob("**/first_plot_questions.yaml")) if Path("/kaggle/input").exists() else []
    cands += [REPO / "first_plot_questions.yaml", Path.cwd() / "first_plot_questions.yaml"]
    for c in cands:
        if c.exists():
            shutil.copy(c, dst)
            return dst
    for r in os.environ.get("PA3_PRIOR_ROOTS", "").split(os.pathsep):
        if r and (Path(r) / "first_plot_questions.yaml").exists():
            shutil.copy(Path(r) / "first_plot_questions.yaml", dst)
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


# ------------------------------------------------------------------ process helpers
def sh(cmd: list[str], log: Path | None = None, env: dict | None = None, timeout: float | None = None,
       echo: bool = True) -> int:
    log = log or (out_root() / "logs" / f"{int(time.time())}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(map(str, cmd))}\n  (log: {log})", flush=True)
    e = {**os.environ, **(env or {})}
    t0 = time.time()
    with open(log, "a") as fh:
        p = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env=e, cwd=str(REPO), bufsize=1)
        try:
            for line in p.stdout:
                fh.write(line)
                if echo and not _noise(line):
                    print(line, end="", flush=True)
                if timeout and time.time() - t0 > timeout:
                    p.kill()
                    print(f"!! timeout after {timeout}s", flush=True)
                    return -9
            return p.wait()
        finally:
            if p.poll() is None:
                p.kill()


def _noise(line: str) -> bool:
    return any(s in line for s in ("Loading weights", "it/s]", "s/it]", "Fetching", "Downloading", "Map:",
                                   "Generating ", "Requesting API"))


def n_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def disk_report(limit_gib: float = 19.5) -> float:
    root = Path("/kaggle/working") if Path("/kaggle/working").exists() else out_root()
    used = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 2**30
    print(f"[disk] {root}: {used:.2f} GiB used (Kaggle output limit ~20 GiB)")
    if used > limit_gib:
        print("!! WARNING: close to the Kaggle /kaggle/working limit")
    return used


# ------------------------------------------------------------------ training
def _l2sp_overrides(run: str) -> list[str]:
    d = decisions()
    if run.startswith("full_ft_l2sp"):
        return [f"zero1={str(d.get('l2sp_zero1', False)).lower()}",
                f"l2sp_ref_device={d.get('l2sp_ref_device', 'cuda')}"]
    return []


def _restore_prior_run(run: str) -> None:
    """If an earlier session died mid-run and its outputs are attached, copy the run dir locally."""
    local = out_root() / "runs" / run
    if local.exists():
        return
    for r in os.environ.get("PA3_PRIOR_ROOTS", "").split(os.pathsep):
        src = Path(r) / "runs" / run if r else None
        if src and src.exists() and not (src / "DONE").exists() and (src / "recovery").exists():
            print(f"[resume] copying incomplete run {run} from {src}")
            shutil.copytree(src, local)
            return


def train(run: str, *sets: str, nproc: int | None = None, run_name: str | None = None,
          timeout: float | None = None) -> int:
    name = run_name or run
    if find(f"runs/{name}/DONE") and not any(s.startswith("max_updates") for s in sets):
        print(f"[train] {name}: DONE (found {find(f'runs/{name}/DONE')}); skipping")
        return 0
    _restore_prior_run(name)
    nproc = nproc or max(1, n_gpus())
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
           "-m", "pa3.train.loop", "--config", CFG / "base.yaml", CFG / f"{run}.yaml",
           "--set", f"run_name={name}", *_l2sp_overrides(run), *sets]
    rc = sh(cmd, out_root() / "logs" / f"train_{name}.log", timeout=timeout)
    print(f"[train] {name} exit code {rc}")
    return rc


def _oom(log: Path) -> bool:
    t = log.read_text(errors="ignore") if log.exists() else ""
    return "OutOfMemoryError" in t or "CUDA out of memory" in t


# ------------------------------------------------------------------ evaluation
def _eval_done(c: str) -> bool:
    return bool(find(f"eval/{c}/aggregate.json")) and bool(find(f"qualitative/per_condition/{c}.jsonl")) \
        if locate_questions() else bool(find(f"eval/{c}/aggregate.json"))


def evaluate(conds: list[str], limit: int | None = None, extra: list[str] | None = None) -> dict[str, int]:
    """Run conditions concurrently, one per GPU (queue), each in its own process."""
    todo = [c for c in conds if limit or not _eval_done(c)]
    for c in conds:
        if c not in todo:
            print(f"[eval] {c}: done; skipping")
    slots = list(range(max(1, n_gpus())))
    running: dict[int, tuple[str, subprocess.Popen, object]] = {}
    rcs: dict[str, int] = {}
    q = locate_questions()
    while todo or running:
        while todo and len(running) < len(slots):
            gpu = next(s for s in slots if s not in running)
            c = todo.pop(0)
            log = out_root() / "logs" / f"eval_{c}{'_smoke' if limit else ''}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "pa3.eval.run", "--condition", c]
            if limit:
                cmd += ["--limit", str(limit)]
            if q:
                cmd += ["--questions", str(q)]
            cmd += extra or []
            fh = open(log, "a")
            print(f"[eval] {c} -> GPU{gpu} (log {log})", flush=True)
            p = subprocess.Popen(list(map(str, cmd)), stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO),
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            running[gpu] = (c, p, fh)
        time.sleep(15)
        for gpu in list(running):
            c, p, fh = running[gpu]
            if p.poll() is not None:
                fh.close()
                rcs[c] = p.returncode
                log = out_root() / "logs" / f"eval_{c}{'_smoke' if limit else ''}.log"
                tail = [l for l in log.read_text(errors="ignore").splitlines() if l.startswith("[")][-6:]
                print(f"[eval] {c} finished rc={p.returncode}\n  " + "\n  ".join(tail), flush=True)
                del running[gpu]
    return rcs


# ------------------------------------------------------------------ qualitative + report
def collect_qualitative() -> None:
    from pa3.eval.conditions import ORDER
    from pa3.eval.qualitative import annotation_subset, load_questions, write_annotation_sheet, write_grouped_markdown

    q = locate_questions()
    if not q:
        print("[qualitative] questions yaml missing")
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
    have = sorted({r["condition"] for r in rows})
    print(f"[qualitative] {len(rows)} responses from {len(have)} conditions; annotation rows: {len(sub)}"
          + ("" if len(rows) == 168 and len(sub) == 42 else "  (incomplete — expected 168 / 42)"))


def report() -> None:
    sh([sys.executable, REPO / "scripts" / "make_report_assets.py"], out_root() / "logs" / "report.log")


def ensure_common() -> None:
    """Data, probe set, OPLoRA bases and the starting-model activation cache."""
    if not find("data/report.json"):
        rc = sh([sys.executable, "-m", "pa3.data.prepare", "--out_dir", out_root() / "data"],
                out_root() / "logs" / "prepare.log")
        assert rc == 0, "data preparation failed"
    if not find("data/probe_ids.json"):
        sh([sys.executable, "-m", "pa3.diagnostics.probe_set"], out_root() / "logs" / "probe.log")
    if not find("bases/oplora_bases_k128.safetensors"):
        rc = sh([sys.executable, REPO / "scripts" / "compute_bases.py"], out_root() / "logs" / "bases.log")
        assert rc == 0, "basis computation failed"
    if not find("diagnostics/z_cache.pt"):
        sh([sys.executable, "-m", "pa3.diagnostics.run", "cache"], out_root() / "logs" / "zcache.log")


# ------------------------------------------------------------------ stages
def stage_s0(cpu_tests: bool = True, smoke_updates: int = 20) -> dict:
    import torch

    sm = out_root() / "smoke"
    sm.mkdir(parents=True, exist_ok=True)
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "n_gpus": n_gpus(),
           "gpus": [torch.cuda.get_device_name(i) for i in range(n_gpus())],
           "capability": [list(torch.cuda.get_device_capability(i)) for i in range(n_gpus())]}
    for mod in ("transformers", "peft", "lm_eval", "datasets", "accelerate"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as ex:  # noqa: BLE001
            env[mod] = f"ERR {ex}"
    (sm / "env.json").write_text(json.dumps(env, indent=1))
    print(json.dumps(env, indent=1))
    summary: dict = {"env": env}

    if cpu_tests:
        rc = sh([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(REPO / "tests")],
                sm / "cpu_tests.log", env={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"})
        summary["cpu_tests_rc"] = rc

    ensure_common()
    summary["data_report"] = json.loads(find("data/report.json").read_text())["counts"]

    sh([sys.executable, "-m", "pa3.smoke", "fp16"], sm / "fp16.log")
    summary["fp16_sanity"] = json.loads((sm / "fp16_sanity.json").read_text()) if (sm / "fp16_sanity.json").exists() else None

    # ---- DDP vs single-GPU accumulation-16 gradient equivalence on GPU (LoRA-16, dropout off)
    g1, g2 = sm / "grad_w1.pt", sm / "grad_w2.pt"
    common = ["lora_dropout=0.0", "grad_dump_update=0", "max_updates=1"]
    train("lora_r16", *common, f"grad_dump_path={g1}", nproc=1, run_name="gradcheck_w1")
    rc = train("lora_r16", *common, f"grad_dump_path={g2}", nproc=2, run_name="gradcheck_w2", timeout=1200)
    if rc != 0 and n_gpus() > 1:
        print("[S0] 2-GPU run failed; retrying with NCCL_P2P_DISABLE=1")
        os.environ["NCCL_P2P_DISABLE"] = "1"
        rc = train("lora_r16", *common, f"grad_dump_path={g2}", nproc=2, run_name="gradcheck_w2", timeout=1200)
        if rc == 0:
            set_decision("env", {**decisions().get("env", {}), "NCCL_P2P_DISABLE": "1"})
    summary["ddp_rc"] = rc
    if g1.exists() and g2.exists():
        sh([sys.executable, "-m", "pa3.smoke", "compare", g1, g2], sm / "gradcmp.log")
        summary["grad_equivalence_gpu"] = json.loads((sm / "grad_equivalence_gpu.json").read_text())
        g1.unlink()
        g2.unlink()

    # ---- 20-update smoke runs (first update = 32 longest examples -> worst-case memory)
    smoke = {}
    for run in ("full_ft", "full_ft_l2sp", "lora_r4", "lora_r16", "oplora_r16"):
        name = f"{run}_smoke"
        sets = [f"max_updates={smoke_updates}", "extra={smoke_longest: true, smoke_export: %s}" %
                ("true" if run == "oplora_r16" else "false")]
        log = out_root() / "logs" / f"train_{name}.log"
        rc = train(run, *sets, run_name=name)
        if run == "full_ft_l2sp" and rc != 0 and _oom(log):
            print("[S0] L2-SP OOM with plain DDP -> ZeRO-1 fallback (approved)")
            set_decision("l2sp_zero1", True)
            rc = train(run, *sets, run_name=name)
            if rc != 0 and _oom(log):
                print("[S0] still OOM -> ZeRO-1 + theta0 in pinned host memory")
                set_decision("l2sp_ref_device", "cpu")
                rc = train(run, *sets, run_name=name)
        res_p = out_root() / "runs" / name / "smoke_resources.json"
        log_p = out_root() / "runs" / name / "smoke_log.jsonl"
        info = {"rc": rc}
        if res_p.exists():
            res = json.loads(res_p.read_text())
            rows = [json.loads(x) for x in log_p.read_text().splitlines()] if log_p.exists() else []
            steady = [r["sec"] for r in rows[2:]] or [r["sec"] for r in rows]
            s_per = sum(steady) / max(1, len(steady))
            info.update({
                "peak_alloc_gib_per_rank": [round(m.get("peak_alloc_gib", 0), 2) for m in res["peak_memory_per_rank"]],
                "peak_reserved_gib_per_rank": [round(m.get("peak_reserved_gib", 0), 2) for m in res["peak_memory_per_rank"]],
                "sec_per_update_steady": round(s_per, 2),
                "projected_train_hours": round(s_per * 1250 / 3600, 2),
                "tokens_per_second": round(res["tokens_per_second"]),
                "skipped_updates": res["skipped_updates"],
                "trainable_params": res["trainable_params"],
                "zero1": res.get("zero1"), "l2sp": res.get("l2sp"),
                "first_losses": [round(r["sft_loss"], 4) for r in rows[:5]],
            })
        smoke[run] = info
        print(f"[S0] {run}: {json.dumps(info)}")
    summary["smoke_runs"] = smoke

    if find("runs/oplora_r16_smoke/weights_u%04d/adapter.safetensors" % smoke_updates):
        sh([sys.executable, "-m", "pa3.smoke", "merge", "oplora_r16_smoke", str(smoke_updates)], sm / "merge.log")
        p = sm / "merge_check_smoke.json"
        summary["merge_check_smoke"] = json.loads(p.read_text()) if p.exists() else None

    rcs = evaluate(["baseline"], limit=8)
    summary["eval_smoke_rc"] = rcs
    p = find("eval_smoke/baseline/aggregate.json")
    if p:
        a = json.loads(p.read_text())
        summary["eval_smoke"] = {"primary": a["primary"], "gsm8k_audit": a.get("gsm8k_audit")}
    summary["questions_yaml_found"] = bool(locate_questions())
    summary["disk_gib"] = disk_report()
    # clean smoke run dirs (keep their logs/resources only)
    for run in smoke:
        for d in (out_root() / "runs" / f"{run}_smoke").glob("weights_u*"):
            shutil.rmtree(d, ignore_errors=True)
    for d in ("gradcheck_w1", "gradcheck_w2"):
        shutil.rmtree(out_root() / "runs" / d, ignore_errors=True)
    (sm / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("\n================ S0 SUMMARY ================")
    print(json.dumps({k: v for k, v in summary.items() if k != "env"}, indent=1, default=str))
    return summary


def stage_s1() -> None:
    ensure_common()
    rc = train("full_ft")
    assert rc == 0, "Full FT failed — rerun this stage to resume from the last recovery checkpoint"
    evaluate(["baseline", "fullft_u0625", "fullft_u1250"])
    report()
    disk_report()


def stage_s2() -> None:
    ensure_common()
    rc = train("full_ft_l2sp")
    if rc != 0 and _oom(out_root() / "logs" / "train_full_ft_l2sp.log") and not decisions().get("l2sp_zero1"):
        print("[S2] L2-SP OOM -> ZeRO-1 fallback")
        set_decision("l2sp_zero1", True)
        rc = train("full_ft_l2sp")
    assert rc == 0, "L2-SP failed"
    assert train("lora_r4") == 0, "LoRA-4 failed"
    evaluate(["l2sp_u1250", "lora4_u1250"])
    report()
    disk_report()


def stage_s3() -> None:
    ensure_common()
    assert train("lora_r16") == 0, "LoRA-16 failed"
    assert train("oplora_r16") == 0, "OPLoRA failed"
    evaluate(["lora16_u1250", "oplora16_u1250"])
    # any condition missing from earlier sessions is (re-)evaluated here as well
    from pa3.eval.conditions import ORDER

    missing = [c for c in ORDER if not _eval_done(c)]
    if missing:
        print(f"[S3] evaluating conditions missing from earlier sessions: {missing}")
        evaluate(missing)
    sh([sys.executable, "-m", "pa3.diagnostics.run", "analyze"], out_root() / "logs" / "diagnostics.log")
    collect_qualitative()
    report()
    disk_report()
