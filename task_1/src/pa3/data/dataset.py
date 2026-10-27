"""Pre-tokenised store, deterministic DDP-aware batch plan, and right-padding collate.

Batch plan (identical example sets to single-GPU microbatch 2 x accumulation 16):
    update k uses global examples [32k, 32k+32), split into 16 consecutive pairs
    (2i, 2i+1).  With world size W each rank gets 16/W consecutive pairs:
    rank r -> pairs [r*16/W, (r+1)*16/W).
The loss of every microbatch is normalised by N_k = number of supervised tokens in all 32
examples of update k (known from the manifest, so no communication is needed), and multiplied
by W to cancel DDP's gradient averaging.  The synced gradient therefore equals the
single-GPU accumulation-16 gradient up to floating-point summation order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

IGNORE = -100


class TokenStore:
    def __init__(self, data_dir: str | Path):
        d = Path(data_dir)
        self.tokens = np.load(d / "tokens.npy", mmap_mode="r")
        self.offsets = np.load(d / "offsets.npy")
        self.prompt_len = np.load(d / "prompt_len.npy")
        self.resp_len = np.load(d / "resp_len.npy")
        self.src_index = np.load(d / "src_index.npy")

    def __len__(self) -> int:
        return len(self.prompt_len)

    def example(self, i: int) -> tuple[np.ndarray, int]:
        s, e = self.offsets[i], self.offsets[i + 1]
        return np.asarray(self.tokens[s:e], dtype=np.int64), int(self.prompt_len[i])


@dataclass
class UpdatePlan:
    update: int  # 0-based optimizer-update index
    n_tokens: int  # supervised tokens in the whole effective batch (all ranks)
    microbatches: list[list[int]]  # this rank's microbatches (global example indices)


def plan_update(store: TokenStore, k: int, rank: int, world: int, micro_bs: int = 2,
                grad_accum: int = 16) -> UpdatePlan:
    eff = micro_bs * grad_accum
    if grad_accum % world:
        raise ValueError(f"grad_accum={grad_accum} not divisible by world={world}")
    base = k * eff
    idx = list(range(base, base + eff))
    if idx[-1] >= len(store):
        raise IndexError(f"update {k} needs example {idx[-1]} but store has {len(store)}")
    micro = [idx[j: j + micro_bs] for j in range(0, eff, micro_bs)]
    per = grad_accum // world
    mine = micro[rank * per:(rank + 1) * per]
    n_tok = int(store.resp_len[base: base + eff].sum())
    return UpdatePlan(update=k, n_tokens=n_tok, microbatches=mine)


def collate(store: TokenStore, indices: list[int], pad_id: int) -> dict[str, torch.Tensor]:
    seqs = [store.example(i) for i in indices]
    T = max(len(s) for s, _ in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    lab = torch.full((len(seqs), T), IGNORE, dtype=torch.long)
    for b, (s, p) in enumerate(seqs):
        n = len(s)
        t = torch.from_numpy(s)
        ids[b, :n] = t
        att[b, :n] = 1
        lab[b, p:n] = t[p:]  # response tokens + <|im_end|>; prompt and padding stay IGNORE
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


def plan_longest(store: TokenStore, rank: int, world: int, micro_bs: int = 2, grad_accum: int = 16) -> UpdatePlan:
    """Smoke-test plan: the effective batch made of the longest examples (worst-case memory)."""
    eff = micro_bs * grad_accum
    lens = np.diff(store.offsets)
    idx = [int(i) for i in np.argsort(-lens, kind="stable")[:eff]]
    micro = [idx[j: j + micro_bs] for j in range(0, eff, micro_bs)]
    per = grad_accum // world
    return UpdatePlan(update=-1, n_tokens=int(store.resp_len[idx].sum()),
                      microbatches=micro[rank * per:(rank + 1) * per])
