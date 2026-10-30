"""Connector g_phi: Linear 768 -> 1024, GELU, Linear 1024 -> 1024 (trainable throughout), followed by a fixed,
non-trainable calibration multiplier s (buffer `scale`, set from training data before Stage 1)."""
from __future__ import annotations

import torch
from torch import nn

from pa3common.utils import stable_int_seed


class Connector(nn.Module):
    def __init__(self, d_in: int = 768, d_hidden: int = 1024, d_out: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.register_buffer("scale", torch.ones((), dtype=torch.float32))

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.raw(x) * self.scale


def make_connector(seed: int, d_out: int = 1024) -> Connector:
    """PyTorch default Linear init drawn from a dedicated seeded generator (independent of global RNG use)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(stable_int_seed(seed, "connector"))
        return Connector(d_out=d_out)
