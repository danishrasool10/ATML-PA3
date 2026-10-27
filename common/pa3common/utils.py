"""Shared helpers (task-agnostic): seeding, distributed helpers, JSON-lines logging, file helpers, memory stats."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


# ----------------------------------------------------------------------------- seeding
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_int_seed(*parts: Any) -> int:
    """Deterministic 63-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:15], 16)


def get_rng_state() -> dict:
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["cuda"])


# ----------------------------------------------------------------------------- distributed
def dist_info() -> tuple[int, int, int]:
    """(rank, local_rank, world_size) from torchrun env vars (defaults to single process)."""
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("LOCAL_RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
    )


def init_distributed() -> tuple[int, int, int, torch.device]:
    rank, local_rank, world = dist_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world > 1 and not dist.is_initialized():
        # generous timeout: rank 0 may write multi-GB checkpoints while the other rank waits at a barrier
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=int(os.environ.get(
            "PA3_DIST_TIMEOUT_MIN", "60"))))
    return rank, local_rank, world, device


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_dist():
        dist.barrier()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def broadcast_int(value: int, device: torch.device, src: int = 0) -> int:
    """Make every rank adopt rank `src`'s decision (e.g. pause / checkpoint now)."""
    if not is_dist():
        return int(value)
    t = torch.tensor([int(value)], device=device, dtype=torch.long)
    dist.broadcast(t, src=src)
    return int(t.item())


def all_gather_object(obj: Any) -> list[Any]:
    if not is_dist():
        return [obj]
    out: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def cleanup_distributed() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


# ----------------------------------------------------------------------------- logging / io
class JsonlLogger:
    def __init__(self, path: str | Path, enabled: bool = True, truncate_after: int | None = None):
        self.path = Path(path)
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if truncate_after is not None and self.path.exists():
                # on resume: drop log rows written after the checkpoint we resume from
                rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
                rows = [r for r in rows if r.get("update", 0) <= truncate_after]
                self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def log(self, row: dict) -> None:
        if self.enabled:
            with open(self.path, "a") as fh:
                fh.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, torch.Tensor):
        return o.item() if o.numel() == 1 else o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def dir_size_bytes(path: str | Path) -> int:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def rmtree(path: str | Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- artifacts
def find_artifact(rel: str | Path, roots: list[str | Path] | None = None) -> Path | None:
    """Look for `rel` under the writable outputs root first, then under prior-session outputs.

    Prior Kaggle sessions are attached read-only under /kaggle/input/<dataset>/...; the env var
    PA3_PRIOR_ROOTS (os.pathsep-separated) lists those roots (the notebook sets it).
    """
    rel = Path(rel)
    if rel.is_absolute():
        return rel if rel.exists() else None
    cands: list[Path] = [Path(r) for r in (roots or [])]
    cands.append(Path(os.environ.get("PA3_OUTPUT_ROOT", "outputs")))
    cands += [Path(p) for p in os.environ.get("PA3_PRIOR_ROOTS", "").split(os.pathsep) if p]
    for root in cands:
        p = root / rel
        if p.exists():
            return p
    return None


def output_root() -> Path:
    return Path(os.environ.get("PA3_OUTPUT_ROOT", "outputs"))


# ----------------------------------------------------------------------------- memory / time
def gpu_mem_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    return {
        "peak_alloc_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "total_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
    }


@contextmanager
def timer(store: dict, key: str, sync_device: torch.device | None = None):
    if sync_device is not None and sync_device.type == "cuda":
        torch.cuda.synchronize(sync_device)
    t0 = time.perf_counter()
    yield
    if sync_device is not None and sync_device.type == "cuda":
        torch.cuda.synchronize(sync_device)
    store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    seen: set[int] = set()
    n = 0
    for p in model.parameters():  # .parameters() already de-duplicates tied weights
        if id(p) in seen or (trainable_only and not p.requires_grad):
            continue
        seen.add(id(p))
        n += p.numel()
    return n
