"""Masked next-token cross-entropy, summed over supervised tokens, with a chunked lm_head.

Only positions whose *next* token is supervised (response tokens + <|im_end|>) are passed
through lm_head, in chunks of `chunk` rows, each wrapped in activation checkpointing.  This
avoids materialising (B, T, 151936) logits (~2.5 GB in FP32 at B=2, T=2048).
The module is what gets wrapped by DDP, so DDP sees every parameter used in forward.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..data.dataset import IGNORE


def _ce_sum(h: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    logits = F.linear(h, w)  # FP16 under autocast
    return F.cross_entropy(logits.float(), t, reduction="sum")


class CausalLMTokenLoss(nn.Module):
    def __init__(self, lm: nn.Module, chunk: int = 1024):
        super().__init__()
        self.lm = lm
        self.chunk = chunk

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.lm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        h = out.last_hidden_state[:, :-1]
        tgt = labels[:, 1:]
        mask = tgt != IGNORE
        hs = h[mask]
        tg = tgt[mask]
        w = self.lm.lm_head.weight
        total = hs.new_zeros((), dtype=torch.float32)
        for s in range(0, hs.shape[0], self.chunk):
            hc, tc = hs[s:s + self.chunk], tg[s:s + self.chunk]
            if torch.is_grad_enabled():
                total = total + checkpoint(_ce_sum, hc, w, tc, use_reentrant=False)
            else:
                total = total + _ce_sum(hc, w, tc)
        return total, mask.sum()
