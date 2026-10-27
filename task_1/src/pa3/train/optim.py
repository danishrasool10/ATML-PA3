"""AdamW with decay/no-decay groups and the standard HF cosine-with-warmup schedule."""
from __future__ import annotations

import math

import torch
from torch import nn


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay every trainable tensor with >= 2 dims that is not a bias; exclude the rest."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or n.endswith(".bias")) else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(model: nn.Module, lr: float, betas, eps: float, weight_decay: float,
                    zero1: bool = False) -> torch.optim.Optimizer:
    groups = param_groups(model, weight_decay)
    kw = dict(lr=lr, betas=tuple(betas), eps=eps)
    if zero1:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        opt = ZeroRedundancyOptimizer(groups[0]["params"], optimizer_class=torch.optim.AdamW,
                                      weight_decay=groups[0]["weight_decay"], **kw)
        for g in groups[1:]:
            opt.add_param_group(g)
        return opt
    return torch.optim.AdamW(groups, foreach=True, **kw)


def hf_cosine_lambda(warmup: int, total: int, num_cycles: float = 0.5):
    """Identical to transformers.get_cosine_schedule_with_warmup (decays to 0)."""

    def f(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return f


def build_scheduler(opt: torch.optim.Optimizer, warmup: int, total: int):
    return torch.optim.lr_scheduler.LambdaLR(opt, hf_cosine_lambda(warmup, total))
