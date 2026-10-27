"""One LoRA module for both LoRA and OPLoRA.

    LoRA   : y = W0 x + (alpha/r) * B  A  dropout(x)
    OPLoRA : y = W0 x + (alpha/r) * B~ A~ dropout(x),
             A~ = A - (A V_k) V_k^T ,   B~ = B - U_k (U_k^T B)

U_k, V_k are frozen FP32 buffers (leading singular bases of the starting weight).  The
effective factors are recomputed from the raw trainable factors in every forward (cost
O(r k (d_in + d_out))), in FP32 with autocast disabled; no dense (I - V V^T) is ever built.

Initialisation matches PEFT's default (`init_lora_weights=True`): A ~ kaiming_uniform(a=sqrt 5)
= U(-1/sqrt(d_in), 1/sqrt(d_in)), B = 0.  A is drawn from a CPU generator seeded by
(seed, "lora_A", module name), so LoRA-16 and OPLoRA-16 start from *identical* raw factors
independent of module construction order, device, or world size.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.common import stable_int_seed


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, init_seed: int,
                 U: torch.Tensor | None = None, V: torch.Tensor | None = None):
        super().__init__()
        assert base.bias is None, "Qwen3 projections have no bias"
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
        self.projected = U is not None
        if self.projected:
            assert V is not None and U.shape[0] == dout and V.shape[0] == din
            self.register_buffer("U", U.to(dev, torch.float32).contiguous(), persistent=False)
            self.register_buffer("V", V.to(dev, torch.float32).contiguous(), persistent=False)

    # -- factors ------------------------------------------------------------------------
    def effective_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        A, B = self.lora_A, self.lora_B
        if not self.projected:
            return A, B
        with torch.autocast(device_type=A.device.type, enabled=False):
            A_eff = A - (A @ self.V) @ self.V.T
            B_eff = B - self.U @ (self.U.T @ B)
        return A_eff, B_eff

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        """FP32 dense correction (alpha/r) B~ A~ of shape (d_out, d_in)."""
        A, B = self.effective_factors()
        with torch.autocast(device_type=A.device.type, enabled=False):
            return self.scale * (B.float() @ A.float())

    # -- forward ------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        A, B = self.effective_factors()
        h = self.dropout(x)
        if not torch.is_autocast_enabled(x.device.type):
            h = h.to(A.dtype)
        out = F.linear(F.linear(h, A), B)
        return y + (out * self.scale).to(y.dtype)

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, projected={self.projected}"


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, leaf = name.rsplit(".", 1)
    setattr(root.get_submodule(parent_name), leaf, new)


def inject_adapters(model: nn.Module, r: int, alpha: float, dropout: float, seed: int,
                    targets: tuple[str, ...], bases: dict[str, dict[str, torch.Tensor]] | None = None
                    ) -> dict[str, LoRALinear]:
    """Freeze the whole model and wrap every targeted projection with LoRALinear."""
    from .load import iter_target_linears

    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, m in iter_target_linears(model, targets) if isinstance(m, nn.Linear)]
    assert names, "no target modules found"
    out: dict[str, LoRALinear] = {}
    for n in names:
        base = model.get_submodule(n)
        U = V = None
        if bases is not None:
            U, V = bases[n]["U"], bases[n]["V"]
        mod = LoRALinear(base, r, alpha, dropout, stable_int_seed(seed, "lora_A", n), U, V)
        _set_submodule(model, n, mod)
        out[n] = mod
    return out


def adapter_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


@torch.no_grad()
def export_factors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Raw and effective (projected for OPLoRA) factors, FP32, CPU.

    Keys: '<name>.A', '<name>.B' (raw trainable)  and  '<name>.A_eff', '<name>.B_eff'
    (what inference/merging/diagnostics must use).  For plain LoRA A_eff == A, B_eff == B.
    """
    t: dict[str, torch.Tensor] = {}
    for n, m in adapter_modules(model).items():
        A_eff, B_eff = m.effective_factors()
        t[f"{n}.A"] = m.lora_A.detach().float().cpu().contiguous().clone()
        t[f"{n}.B"] = m.lora_B.detach().float().cpu().contiguous().clone()
        t[f"{n}.A_eff"] = A_eff.detach().float().cpu().contiguous().clone()
        t[f"{n}.B_eff"] = B_eff.detach().float().cpu().contiguous().clone()
    return t


def factor_names(factors: dict[str, torch.Tensor]) -> list[str]:
    return sorted({k.rsplit(".", 1)[0] for k in factors if k.endswith(".A_eff")})


def delta_from_factors(factors: dict[str, torch.Tensor], name: str, scale: float) -> torch.Tensor:
    return scale * (factors[f"{name}.B_eff"].float() @ factors[f"{name}.A_eff"].float())


@torch.no_grad()
def merge_into(model: nn.Module, factors: dict[str, torch.Tensor], scale: float,
               reference: nn.Module | None = None) -> None:
    """W <- W0 + (alpha/r) B_eff A_eff, computed in FP32 then cast to the target dtype.

    `reference` (optional) supplies W0 in FP32 (exact bf16 upcast); otherwise the target's
    own weights are upcast.
    """
    for n in factor_names(factors):
        tgt = model.get_submodule(n)
        W0 = (reference.get_submodule(n) if reference is not None else tgt).weight.detach().float().cpu()
        W = W0 + delta_from_factors(factors, n, scale)
        tgt.weight.copy_(W.to(tgt.weight.device, tgt.weight.dtype))


def load_factors_into_modules(model: nn.Module, factors: dict[str, torch.Tensor]) -> None:
    """Restore raw trainable factors (for resume or unmerged evaluation)."""
    for n, m in adapter_modules(model).items():
        m.lora_A.data.copy_(factors[f"{n}.A"].to(m.lora_A.device))
        m.lora_B.data.copy_(factors[f"{n}.B"].to(m.lora_B.device))


def save_adapter_meta(path: str | Path, **kw) -> None:
    from ..utils.common import write_json

    write_json(path, kw)
