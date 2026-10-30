"""Shared LoRA module (plain LoRA, the Task 1 LoRALinear without the OPLoRA projection).

    y = W0 x + (alpha / r) * B A dropout(x)

Initialisation matches PEFT's default (`init_lora_weights=True`): A ~ U(-1/sqrt(d_in), 1/sqrt(d_in))
(= kaiming_uniform(a=sqrt 5)), B = 0, so injecting adapters leaves the model function unchanged.
A is drawn from a CPU generator seeded by (seed, "lora_A", module name): identical factors regardless
of module construction order or device.  Trainable factors are FP32; the frozen base may be FP16.
Task 1 keeps its own copy (task_1/src/pa3/models/lora.py); this one is for Task 2 onwards.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .utils import stable_int_seed


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, init_seed: int):
        super().__init__()
        assert base.bias is None, "Qwen3 attention projections have no bias"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dout, din = base.weight.shape
        self.r, self.alpha, self.scale = r, float(alpha), float(alpha) / r
        g = torch.Generator().manual_seed(init_seed)
        bound = 1.0 / math.sqrt(din)
        A = (torch.rand(r, din, generator=g, dtype=torch.float32) * 2 - 1) * bound
        dev = base.weight.device
        self.lora_A = nn.Parameter(A.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(dout, r, dtype=torch.float32, device=dev))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        h = self.dropout(x)
        if not torch.is_autocast_enabled(x.device.type):
            h = h.to(self.lora_A.dtype)
        out = F.linear(F.linear(h, self.lora_A), self.lora_B)
        return y + (out * self.scale).to(y.dtype)

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}"


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, leaf = name.rsplit(".", 1)
    setattr(root.get_submodule(parent_name), leaf, new)


def inject_lora(model: nn.Module, r: int, alpha: float, dropout: float, seed: int,
                targets: tuple[str, ...] | list[str]) -> dict[str, LoRALinear]:
    """Wrap every nn.Linear whose leaf name is in `targets` (base weights frozen)."""
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in set(targets)]
    assert names, "no target modules found"
    out: dict[str, LoRALinear] = {}
    for n in names:
        mod = LoRALinear(model.get_submodule(n), r, alpha, dropout, stable_int_seed(seed, "lora_A", n))
        _set_submodule(model, n, mod)
        out[n] = mod
    return out


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {f"{n}.{k}": getattr(m, k).detach().float().cpu().clone()
            for n, m in lora_modules(model).items() for k in ("lora_A", "lora_B")}


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    mods = lora_modules(model)
    assert len(state) == 2 * len(mods), (len(state), len(mods))
    for n, m in mods.items():
        m.lora_A.data.copy_(state[f"{n}.lora_A"].to(m.lora_A.device))
        m.lora_B.data.copy_(state[f"{n}.lora_B"].to(m.lora_B.device))
