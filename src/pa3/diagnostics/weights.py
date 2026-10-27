"""Weight-space and activation diagnostics on down_proj of layers 0, 13, 27 (FP32).

    rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2     (U_k: starting weight's leading left bases)
    eta   = ||dW||_F / ||W0||_F
    R(z)  = ||dW z||_2 / (||W0 z||_2 + 1e-8)       on cached starting-model inputs z

dW = W_trained - W0 for Full FT variants (FP32 snapshots vs exact FP32 upcast of the bf16
checkpoint); dW = (alpha/r) B_eff A_eff for adapters (projected factors for OPLoRA).
The same bases file (k = 128) is used for every method.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..models.load import load_safetensors
from ..models.lora import delta_from_factors
from ..models.oplora import constraint_checks
from ..eval.conditions import Condition, adapter_meta, weights_path

EPS = 1e-8


def down_name(layer: int) -> str:
    return f"model.layers.{layer}.mlp.down_proj"


def deltas_for_condition(cond: Condition, W0: dict[str, torch.Tensor], names: list[str]) -> dict[str, torch.Tensor]:
    p = weights_path(cond)
    if cond.kind == "full":
        sd = load_safetensors(p)
        return {n: sd[f"{n}.weight"].float() - W0[n] for n in names}
    meta = adapter_meta(cond)
    fac = load_safetensors(p)
    return {n: delta_from_factors(fac, n, float(meta["scale"])) for n in names}


@torch.no_grad()
def weight_metrics(dW: torch.Tensor, W0: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> dict:
    d = dW.double()
    nd2 = float(d.square().sum())
    out = {
        "dW_fro": math.sqrt(nd2),
        "W0_fro": float(W0.double().norm()),
        "eta": math.sqrt(nd2) / float(W0.double().norm()),
        "rho_k": float((U.double().T @ d).square().sum()) / nd2 if nd2 > 0 else float("nan"),
        "rho_k_input_side": float((d @ V.double()).square().sum()) / nd2 if nd2 > 0 else float("nan"),
    }
    # reference: a random matrix of the same shape would have rho_k ~ k / d_out
    out["rho_k_random_baseline"] = U.shape[1] / U.shape[0]
    return out


@torch.no_grad()
def activation_ratios(dW: torch.Tensor, W0: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Z: (n, d_in) -> R: (n,)"""
    Zd = Z.double()
    num = (Zd @ dW.double().T).norm(dim=1)
    den = (Zd @ W0.double().T).norm(dim=1) + EPS
    return (num / den).float()


@torch.no_grad()
def full_displacement(cond: Condition, base: nn.Module) -> float:
    """||theta - theta0||_2 over all unique parameters (Full FT snapshots)."""
    sd = load_safetensors(weights_path(cond))
    acc = 0.0
    for n, p in base.named_parameters():
        acc += float((sd[n].double() - p.detach().double()).square().sum())
    return math.sqrt(acc)


@torch.no_grad()
def adapter_displacement(cond: Condition) -> float:
    meta = adapter_meta(cond)
    fac = load_safetensors(weights_path(cond))
    names = sorted({k.rsplit(".", 1)[0] for k in fac if k.endswith(".A_eff")})
    return math.sqrt(sum(float(delta_from_factors(fac, n, float(meta["scale"])).double().square().sum())
                         for n in names))


def oplora_checks_all(cond: Condition, base: nn.Module, bases: dict) -> dict:
    meta = adapter_meta(cond)
    fac = load_safetensors(weights_path(cond))
    rows = {}
    for n in sorted({k.rsplit(".", 1)[0] for k in fac if k.endswith(".A_eff")}):
        dW = delta_from_factors(fac, n, float(meta["scale"]))
        W0 = base.get_submodule(n).weight.detach().float()
        b = bases[n]
        rows[n] = constraint_checks(W0, dW, b["U"], b["V"], b["S"], full_svd=True)
    keys = ["left_leak", "right_leak", "prop2_right_resid", "prop2_left_resid", "topk_sv_max_rel_change"]
    summary = {k: max(r[k] for r in rows.values()) for k in keys}
    summary["n_modules"] = len(rows)
    summary["all_leading_k_preserved"] = all(r["leading_k_preserved"] for r in rows.values())
    return {"summary_max_over_modules": summary, "per_module": rows}
