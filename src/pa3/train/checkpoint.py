"""Recovery checkpoints (model, optimizer, scheduler, scaler, per-rank RNG, data position).

Layout:  <run_dir>/recovery/u0312/
            weights.safetensors  (trainable params, FP32)  OR  weights_ref.json -> snapshot
            optimizer.pt          (consolidated for ZeRO-1)
            state.pt              (scheduler, scaler, update, counters, timing)
            rng_rank{r}.pt
            COMPLETE              (written last; a checkpoint without it is ignored)
Rolling policy: after a new checkpoint is COMPLETE and reloads cleanly, older recovery dirs
are deleted (their sizes are recorded first).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from ..models.load import load_safetensors, save_safetensors, trainable_state_dict
from ..utils.common import barrier, dir_size_bytes, get_rng_state, read_json, rmtree, set_rng_state, write_json


def _ref_path(d: Path) -> Path:
    return (d / read_json(d / "weights_ref.json")["path"]).resolve()


def ckpt_dir(run_dir: Path, update: int) -> Path:
    return run_dir / "recovery" / f"u{update:04d}"


def list_complete(run_dir: Path) -> list[Path]:
    root = run_dir / "recovery"
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "COMPLETE").exists())


def _optimizer_state(opt: torch.optim.Optimizer, rank: int) -> dict | None:
    from torch.distributed.optim import ZeroRedundancyOptimizer

    if isinstance(opt, ZeroRedundancyOptimizer):
        opt.consolidate_state_dict(to=0)
        return opt.state_dict() if rank == 0 else None
    return opt.state_dict() if rank == 0 else None


def save_recovery(run_dir: Path, update: int, model: nn.Module, opt, sched, scaler, state: dict,
                  rank: int, weights_ref: Path | None = None) -> Path:
    d = ckpt_dir(run_dir, update)
    if rank == 0:
        rmtree(d)
        d.mkdir(parents=True)
    barrier()
    osd = _optimizer_state(opt, rank)  # collective for ZeRO
    torch.save(get_rng_state(), d / f"rng_rank{rank}.pt")
    if rank == 0:
        if weights_ref is not None:
            # relative path so the checkpoint survives being copied/mounted elsewhere (Kaggle inputs)
            write_json(d / "weights_ref.json", {"path": os.path.relpath(weights_ref.resolve(), d.resolve())})
        else:
            save_safetensors(trainable_state_dict(model), d / "weights.safetensors")
        torch.save(osd, d / "optimizer.pt")
        torch.save({"sched": sched.state_dict(), "scaler": scaler.state_dict(), **state}, d / "state.pt")
    barrier()
    if rank == 0:
        (d / "COMPLETE").write_text("ok")
    barrier()
    return d


def verify_recovery(d: Path) -> bool:
    try:
        st = torch.load(d / "state.pt", weights_only=False)
        ok = "update" in st and (d / "optimizer.pt").stat().st_size > 0
        if (d / "weights_ref.json").exists():
            ok = ok and _ref_path(d).exists()
        else:
            ok = ok and (d / "weights.safetensors").stat().st_size > 0
        return bool(ok)
    except Exception:
        return False


def recovery_size(d: Path) -> int:
    n = dir_size_bytes(d)
    if (d / "weights_ref.json").exists():
        n += _ref_path(d).stat().st_size
    return n


def rotate(run_dir: Path, keep_last: int, rank: int) -> list[dict]:
    """Delete all but the newest `keep_last` complete recovery checkpoints (rank 0)."""
    removed = []
    if rank == 0:
        comp = list_complete(run_dir)
        if comp and verify_recovery(comp[-1]):
            for d in comp[:-keep_last] if keep_last > 0 else comp:
                removed.append({"dir": d.name, "bytes": recovery_size(d)})
                rmtree(d)
    barrier()
    return removed


def load_recovery(d: Path, model: nn.Module, opt, sched, scaler, rank: int, device) -> dict:
    if (d / "weights_ref.json").exists():
        wpath = _ref_path(d)
    else:
        wpath = d / "weights.safetensors"
    sd = load_safetensors(wpath)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in sd.items():
            named[n].copy_(t.to(named[n].device, named[n].dtype))
    osd = torch.load(d / "optimizer.pt", map_location="cpu", weights_only=False)
    opt.load_state_dict(osd)
    st = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
    sched.load_state_dict(st.pop("sched"))
    scaler.load_state_dict(st.pop("scaler"))
    set_rng_state(torch.load(d / f"rng_rank{rank}.pt", weights_only=False))
    return st
