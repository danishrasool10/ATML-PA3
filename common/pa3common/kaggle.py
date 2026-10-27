"""Task-agnostic Kaggle helpers: subprocess streaming, GPU job queue, session deadline, disk/scratch,
prior-session discovery and retries.  Used by task orchestrators (e.g. task_1/src/pa3/orchestrate.py)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

NOISE = ("Loading weights", "it/s]", "s/it]", "Fetching", "Downloading", "Map:", "Generating ",
         "Requesting API", "Filter:", "Resolving data files")


# ------------------------------------------------------------------ session deadline
class Deadline:
    """Wall-clock budget for one Kaggle session (hard limit 12 h; outputs are saved only if the
    notebook finishes, so every step must stop before the deadline)."""

    def __init__(self, budget_hours: float, start: float | None = None):
        self.start = start or float(os.environ.get("PA3_SESSION_START", time.time()))
        self.end = self.start + budget_hours * 3600
        os.environ["PA3_SESSION_START"] = str(self.start)
        os.environ["PA3_DEADLINE_UNIX"] = str(self.end)

    def remaining(self) -> float:
        return self.end - time.time()

    def fits(self, seconds: float, margin: float = 600) -> bool:
        return self.remaining() > seconds + margin

    def __str__(self) -> str:
        return f"{self.remaining() / 3600:.2f} h left of session budget"


class StagePaused(RuntimeError):
    """Raised when the session budget is exhausted; rerun the notebook to continue."""


# ------------------------------------------------------------------ processes
def sh(cmd: list, log: Path, env: dict | None = None, timeout: float | None = None, echo: bool = True,
       cwd: str | Path | None = None) -> int:
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(map(str, cmd))}\n  (log: {log})", flush=True)
    t0 = time.time()
    with open(log, "a") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(map(str, cmd))}\n")
        p = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env={**os.environ, **(env or {})}, cwd=str(cwd) if cwd else None, bufsize=1)
        try:
            for line in p.stdout:
                fh.write(line)
                fh.flush()
                if echo and not any(s in line for s in NOISE):
                    print(line, end="", flush=True)
                if timeout and time.time() - t0 > timeout:
                    p.kill()
                    print(f"!! timeout after {timeout:.0f}s", flush=True)
                    return -9
            return p.wait()
        finally:
            if p.poll() is None:
                p.kill()


def n_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def run_on_gpus(jobs: list[tuple[str, list, Path]], env_extra: dict | None = None,
                poll: float = 10.0, kill_at: float | None = None) -> dict[str, int]:
    """Run independent single-GPU jobs concurrently, one per visible GPU (queue). Returns rc per job.
    Jobs still running at unix time `kill_at` are killed (rc -9) so the session can end cleanly."""
    slots = list(range(max(1, n_gpus())))
    todo = list(jobs)
    running: dict[int, tuple[str, subprocess.Popen, object, Path]] = {}
    rcs: dict[str, int] = {}
    while todo or running:
        while todo and len(running) < len(slots):
            gpu = next(s for s in slots if s not in running)
            name, cmd, log = todo.pop(0)
            log = Path(log)
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log, "a")
            fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} GPU{gpu} $ {' '.join(map(str, cmd))}\n")
            fh.flush()
            print(f"[gpu-queue] {name} -> GPU{gpu} (log {log})", flush=True)
            p = subprocess.Popen(list(map(str, cmd)), stdout=fh, stderr=subprocess.STDOUT,
                                 env={**os.environ, **(env_extra or {}), "CUDA_VISIBLE_DEVICES": str(gpu)})
            running[gpu] = (name, p, fh, log)
        time.sleep(poll)
        if kill_at is not None and time.time() > kill_at:
            for gpu, (name, p, fh, log) in list(running.items()):
                print(f"[gpu-queue] {name}: session deadline reached -> killing", flush=True)
                p.kill()
            todo = []
        for gpu in list(running):
            name, p, fh, log = running[gpu]
            if p.poll() is not None:
                fh.close()
                rcs[name] = p.returncode
                tail = [l for l in log.read_text(errors="ignore").splitlines() if l.startswith("[")][-5:]
                print(f"[gpu-queue] {name} finished rc={p.returncode}\n  " + "\n  ".join(tail), flush=True)
                del running[gpu]
    return rcs


def log_has(log: Path, *needles: str) -> bool:
    t = Path(log).read_text(errors="ignore") if Path(log).exists() else ""
    return any(n in t for n in needles)


def is_oom(log: Path) -> bool:
    return log_has(log, "OutOfMemoryError", "CUDA out of memory")


def retry(fn: Callable, tries: int = 4, wait: float = 20.0, what: str = ""):
    for i in range(tries):
        try:
            return fn()
        except Exception as ex:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"[retry] {what} failed ({type(ex).__name__}: {str(ex)[:200]}); retrying in {wait:.0f}s",
                  flush=True)
            time.sleep(wait * (i + 1))


def pip_install(pkgs: list[str], tries: int = 3) -> None:
    def go():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", *pkgs],
                       check=True)
    retry(go, tries, 15, "pip install")


# ------------------------------------------------------------------ disk / paths
def scratch_root(name: str = "pa3_scratch") -> Path:
    """Large, non-persisted disk for rolling recovery checkpoints (keeps /kaggle/working < 20 GB)."""
    env = os.environ.get("PA3_SCRATCH")
    cands = [Path(env)] if env else [Path("/kaggle/temp") / name, Path("/tmp") / name]
    for c in cands:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".w"
            probe.write_text("ok")
            probe.unlink()
            os.environ["PA3_SCRATCH"] = str(c)
            return c
        except OSError:
            continue
    raise RuntimeError("no writable scratch directory")


def free_gib(path: str | Path) -> float:
    return shutil.disk_usage(str(path)).free / 2**30


def tree_gib(root: str | Path) -> float:
    root = Path(root)
    if not root.exists():
        return 0.0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 2**30


def disk_report(working: str | Path = "/kaggle/working", limit_gib: float = 19.0) -> float:
    w = Path(working) if Path(working).exists() else Path.cwd()
    used = tree_gib(w)
    msg = f"[disk] {w}: {used:.2f} GiB used (persisted-output limit ~20 GiB)"
    if os.environ.get("PA3_SCRATCH"):
        s = os.environ["PA3_SCRATCH"]
        msg += f" | scratch {s}: {tree_gib(s):.2f} GiB used, {free_gib(s):.0f} GiB free"
    print(msg, flush=True)
    if used > limit_gib:
        print("!! WARNING: /kaggle/working is close to the 20 GiB limit", flush=True)
    return used


def discover_prior_roots(dirname: str, base: str = "/kaggle/input", max_depth: int = 6) -> list[str]:
    """Earlier sessions' outputs attached as inputs: every directory called `dirname` under /kaggle/input."""
    b = Path(base)
    if not b.exists():
        return []
    return [str(p) for p in sorted(b.glob(f"**/{dirname}")) if p.is_dir() and len(p.relative_to(b).parts) <= max_depth]


def copytree_merge(src: Path, dst: Path, skip: tuple[str, ...] = ()) -> None:
    for f in Path(src).rglob("*"):
        rel = f.relative_to(src)
        if any(part in skip for part in rel.parts):
            continue
        t = Path(dst) / rel
        if f.is_dir():
            t.mkdir(parents=True, exist_ok=True)
        elif not t.exists():
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, t)
