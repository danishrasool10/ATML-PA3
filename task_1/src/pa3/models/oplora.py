"""OPLoRA singular bases (computed once, FP32) and numerical constraint checks.

For every targeted W0 (d_out x d_in) we store U_k (d_out x k), V_k (d_in x k) and S_k (k,)
from the thin SVD W0 = U S V^T of the FP32 (exact bf16 upcast) starting weight.
The same bases are reused by the weight-space diagnostics (rho_k) for *all* methods.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn

from .load import iter_target_linears, load_safetensors, save_safetensors


@torch.no_grad()
def compute_bases(model: nn.Module, k: int, targets: tuple[str, ...], device: torch.device
                  ) -> tuple[dict[str, torch.Tensor], dict]:
    tensors: dict[str, torch.Tensor] = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    n_mod = 0
    for name, mod in iter_target_linears(model, targets):
        W = mod.weight.detach().to(device=device, dtype=torch.float32)
        assert k <= min(W.shape), f"k={k} exceeds rank bound of {name} {tuple(W.shape)}"
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        tensors[f"{name}.U"] = U[:, :k].contiguous().cpu()
        tensors[f"{name}.V"] = Vh[:k].T.contiguous().cpu()
        tensors[f"{name}.S"] = S[:k].contiguous().cpu()
        n_mod += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    secs = time.perf_counter() - t0
    basis_bytes = sum(v.numel() * 4 for kk, v in tensors.items() if not kk.endswith(".S"))
    meta = {"k": k, "n_modules": n_mod, "svd_seconds": secs, "basis_bytes_fp32": basis_bytes,
            "device": str(device)}
    return tensors, meta


def save_bases(tensors: dict[str, torch.Tensor], meta: dict, path: str | Path) -> int:
    return save_safetensors(tensors, path, metadata=meta)


def load_bases(path: str | Path, device: str | torch.device = "cpu") -> dict[str, dict[str, torch.Tensor]]:
    flat = load_safetensors(path, device)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for key, v in flat.items():
        name, part = key.rsplit(".", 1)
        out.setdefault(name, {})[part] = v
    return out


def read_bases_meta(path: str | Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})


@torch.no_grad()
def constraint_checks(W0: torch.Tensor, dW: torch.Tensor, U: torch.Tensor, V: torch.Tensor,
                      S: torch.Tensor, full_svd: bool = True) -> dict:
    """Numerical verification of OPLoRA's constraints (independent of benchmark results).

    left_leak  = ||U_k^T dW||_F / ||dW||_F        (should be ~0)
    right_leak = ||dW V_k||_F  / ||dW||_F         (should be ~0)
    prop2_*    : the k leading singular triples of W0 remain singular triples of W0 + dW
                 ||(W0+dW) V_k - U_k S_k||_F / ||S_k||, ||U_k^T (W0+dW) - S_k V_k^T||_F / ||S_k||
    topk_sv_*  : whether they are still the *leading* k (requires ||dW||_2 on the complement
                 to stay below sigma_k); reported, not assumed.
    """
    W0, dW, U, V, S = (t.double() for t in (W0, dW, U, V, S))
    nd = dW.norm()
    W1 = W0 + dW
    out = {
        "dW_fro": float(nd),
        "left_leak": float((U.T @ dW).norm() / nd) if nd > 0 else 0.0,
        "right_leak": float((dW @ V).norm() / nd) if nd > 0 else 0.0,
        "prop2_right_resid": float((W1 @ V - U * S).norm() / S.norm()),
        "prop2_left_resid": float((U.T @ W1 - S[:, None] * V.T).norm() / S.norm()),
    }
    if full_svd:
        sv = torch.linalg.svdvals(W1)
        k = S.numel()
        out["topk_sv_max_rel_change"] = float(((sv[:k] - S).abs() / S).max())
        out["sigma_k"] = float(S[-1])
        out["sigma_k_plus_1_after"] = float(sv[k]) if sv.numel() > k else float("nan")
        # FP32 bases/SVD give ~1e-6 relative noise; a genuinely displaced leading triple changes by >> 1e-4
        out["leading_k_preserved"] = bool(out["topk_sv_max_rel_change"] < 1e-4)
    return out
