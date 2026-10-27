"""L2-SP: Omega(theta, theta0) = 1/2 sum_j (theta_j - theta0_j)^2 over all unique parameters.

* theta0 is the starting checkpoint, fixed.  The hub weights are bf16, so storing theta0 in
  bf16 is *exact* (verified at construction); otherwise FP32 is used.
* Tied parameters are counted once (model.named_parameters() de-duplicates).
* Omega is not averaged over parameters.
* The penalty gradient lambda*(theta - theta0) is added directly to .grad once per optimizer
  update, after GradScaler.unscale_ and after DDP has synchronised the SFT gradient, so it is
  neither loss-scaled nor divided by the world size.  Clipping then acts on the total gradient.
* Omega/penalty are evaluated at the parameters the gradient is computed at (pre-update).
"""
from __future__ import annotations

import time

import torch
from torch import nn

CHUNK = 1 << 24  # 16M elements (64 MB FP32 temporaries)


class L2SPReference:
    def __init__(self, model: nn.Module, device: torch.device, ref_device: str = "cuda"):
        self.device = device
        self.on_host = ref_device == "cpu" or device.type != "cuda"
        self.names: list[str] = []
        self.refs: list[torch.Tensor] = []
        exact_bf16 = True
        for n, p in model.named_parameters():
            r = p.detach().to(torch.bfloat16)
            if not torch.equal(r.float(), p.detach().float()):
                exact_bf16 = False
                break
        self.dtype = torch.bfloat16 if exact_bf16 else torch.float32
        t0 = time.perf_counter()
        for n, p in model.named_parameters():
            r = p.detach().to(self.dtype).clone()
            r = r.pin_memory() if (self.on_host and device.type == "cuda") else r.to("cpu" if self.on_host else device)
            self.names.append(n)
            self.refs.append(r)
        self.setup_seconds = time.perf_counter() - t0
        self.n_params = sum(r.numel() for r in self.refs)
        self.storage_bytes = sum(r.numel() * r.element_size() for r in self.refs)
        self.transfer_bytes = 0
        self.transfer_seconds = 0.0  # host->device streaming time (only when on_host)
        self.penalty_seconds = 0.0  # total time spent in omega/grad passes

    def _params(self, model: nn.Module):
        named = dict(model.named_parameters())
        return [named[n] for n in self.names]

    @torch.no_grad()
    def omega_and_grad(self, model: nn.Module, lam: float, add_grad: bool = True) -> float:
        """Return Omega (python float) and, if add_grad, do p.grad += lam * (p - p0)."""
        sync = self.device.type == "cuda"
        if sync:
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        acc = torch.zeros((), dtype=torch.float64, device=self.device)
        tx = 0.0
        for p, r in zip(self._params(model), self.refs):
            pf = p.detach().view(-1)
            rf = r.view(-1)
            g = p.grad.view(-1) if (add_grad and p.grad is not None) else None
            for s in range(0, pf.numel(), CHUNK):
                e = min(s + CHUNK, pf.numel())
                if self.on_host and self.device.type == "cuda":
                    t1 = time.perf_counter()
                    rc = rf[s:e].to(self.device, non_blocking=True)
                    self.transfer_bytes += rc.numel() * rc.element_size()
                    tx += time.perf_counter() - t1
                else:
                    rc = rf[s:e]
                d = pf[s:e].float() - rc.float()
                acc += d.square().sum(dtype=torch.float64)
                if g is not None:
                    g[s:e].add_(d.to(g.dtype), alpha=lam)
        omega = 0.5 * float(acc)  # .item() synchronises
        self.transfer_seconds += tx
        self.penalty_seconds += time.perf_counter() - t0
        return omega

    def stats(self) -> dict:
        return {
            "ref_dtype": str(self.dtype).replace("torch.", ""),
            "ref_on_host": self.on_host,
            "ref_storage_bytes": self.storage_bytes,
            "ref_n_params": self.n_params,
            "ref_setup_seconds": self.setup_seconds,
            "ref_transfer_bytes": self.transfer_bytes,
            "ref_transfer_seconds": self.transfer_seconds,
            "penalty_seconds_total": self.penalty_seconds,
        }
