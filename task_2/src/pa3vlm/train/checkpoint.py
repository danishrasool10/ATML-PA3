"""Resumable checkpoints (connector + LoRA factors + AdamW + GradScaler + RNG + progress) and stage weights.

Only trainable state is stored (the frozen backbone is reloaded from the hub), so a checkpoint is
~80 MB.  The newest `keep` resumable checkpoints are kept; every save/deletion is logged with its size.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from pa3common.lora import lora_state
from pa3common.utils import JsonlLogger, get_rng_state


def save_resumable(ck_dir: Path, update: int, stage: int, vlm, opt, scaler, extra: dict, keep: int,
                   log: JsonlLogger) -> Path:
    ck_dir.mkdir(parents=True, exist_ok=True)
    path = ck_dir / f"resume_{update:05d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"update": update, "stage": stage, "connector": {k: v.detach().cpu() for k, v in
                                                                 vlm.connector.state_dict().items()},
                "lora": lora_state(vlm.lm), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                "rng": get_rng_state(), "extra": extra, "time": time.time()}, tmp)
    os.replace(tmp, path)
    log.log({"update": update, "event": "save", "path": path.name, "bytes": path.stat().st_size})
    olds = sorted(ck_dir.glob("resume_*.pt"))[:-keep] if keep > 0 else []
    for o in olds:
        log.log({"update": update, "event": "delete", "path": o.name, "bytes": o.stat().st_size})
        o.unlink()
    return path


def latest_resumable(ck_dir: Path) -> Path | None:
    c = sorted(ck_dir.glob("resume_*.pt")) if ck_dir.exists() else []
    return c[-1] if c else None


def save_weights(path: Path, vlm, meta: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"connector": {k: v.detach().cpu() for k, v in vlm.connector.state_dict().items()},
                "lora": lora_state(vlm.lm), "meta": meta}, tmp)
    os.replace(tmp, path)
    return path.stat().st_size
