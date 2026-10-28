"""Recovery checkpoints (model, optimizer, scheduler, scaler, per-rank RNG, data position).

Layout:  <root>/recovery/u0312/      (root = cfg.recovery_dir; on Kaggle a scratch disk)
            weights.safetensors  (trainable params, FP32)
            optimizer.pt          (plain AdamW, rank 0)  OR  optimizer_rank{r}.pt (ZeRO-1: each
                                  rank's own shard; nothing is gathered/pickled into host RAM)
            state.pt              (scheduler, scaler, update, counters, timing)
            rng_rank{r}.pt
            COMPLETE              (written last; a checkpoint without it is ignored)
Rolling policy: after a new checkpoint is COMPLETE and reloads cleanly, older recovery dirs
are deleted (their sizes are recorded first).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..models.load import load_safetensors, save_safetensors, trainable_state_dict
from ..utils.common import barrier, dir_size_bytes, get_rng_state, rmtree, set_rng_state


def ckpt_dir(run_dir: Path, update: int) -> Path:
    return run_dir / "recovery" / f"u{update:04d}"


def list_complete(run_dir: Path) -> list[Path]:
    root = run_dir / "recovery"
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "COMPLETE").exists())


def _is_zero(opt) -> bool:
    from torch.distributed.optim import ZeroRedundancyOptimizer

    return isinstance(opt, ZeroRedundancyOptimizer)


def save_recovery(run_dir: Path, update: int, model: nn.Module, opt, sched, scaler, state: dict,
                  rank: int) -> Path:
    d = ckpt_dir(run_dir, update)
    if rank == 0:
        rmtree(d)
        d.mkdir(parents=True)
    barrier()
    torch.save(get_rng_state(), d / f"rng_rank{rank}.pt")
    if _is_zero(opt):  # local shard only (same world size + deterministic partition on resume)
        torch.save(opt.optim.state_dict(), d / f"optimizer_rank{rank}.pt")
    if rank == 0:
        save_safetensors(trainable_state_dict(model), d / "weights.safetensors")
        if not _is_zero(opt):
            torch.save(opt.state_dict(), d / "optimizer.pt")
        torch.save({"sched": sched.state_dict(), "scaler": scaler.state_dict(), **state}, d / "state.pt")
    barrier()
    if rank == 0:
        (d / "COMPLETE").write_text("ok")
    barrier()
    return d


def verify_recovery(d: Path) -> bool:
    try:
        st = torch.load(d / "state.pt", weights_only=False)
        opt_files = [d / "optimizer.pt"] if (d / "optimizer.pt").exists() else sorted(d.glob("optimizer_rank*.pt"))
        ok = "update" in st and bool(opt_files) and all(f.stat().st_size > 0 for f in opt_files)
        return bool(ok and (d / "weights.safetensors").stat().st_size > 0)
    except Exception:
        return False


def recovery_size(d: Path) -> int:
    return dir_size_bytes(d)


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
    sd = load_safetensors(d / "weights.safetensors")
    named = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in sd.items():
            named[n].copy_(t.to(named[n].device, named[n].dtype))
    if _is_zero(opt):
        f = d / f"optimizer_rank{rank}.pt"
        if not f.exists():
            raise RuntimeError(f"{d} has no ZeRO-1 shard for rank {rank} (saved with a different optimizer mode)")
        opt.optim.load_state_dict(torch.load(f, map_location="cpu", weights_only=False))
    else:
        if not (d / "optimizer.pt").exists():
            raise RuntimeError(f"{d} was saved with ZeRO-1; resume with zero1=true")
        opt.load_state_dict(torch.load(d / "optimizer.pt", map_location="cpu", weights_only=False))
    st = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
    sched.load_state_dict(st.pop("sched"))
    # LambdaLR.load_state_dict does not touch the optimizer; plain AdamW gets its lr back from its own state
    # dict, but ZeRO-1's outer param groups do not -> set the current lr explicitly for both cases
    for g, lr in zip(opt.param_groups, sched.get_last_lr()):
        g["lr"] = lr
    scaler.load_state_dict(st.pop("scaler"))
    set_rng_state(torch.load(d / f"rng_rank{rank}.pt", weights_only=False))
    return st
