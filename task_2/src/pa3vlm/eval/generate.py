"""Greedy decoding with a KV cache and explicit position ids (left-padded batches, inputs_embeds prefill).

Written by hand rather than via HF `generate` so that positions for left-padded embedding inputs are
under our control (cumsum(mask)-1, continued by +1 per step); checked against HF generate in the tests.
"""
from __future__ import annotations

import time

import torch

from ..data.collate import Batch
from ..models.vlm import VLM


@torch.no_grad()
def greedy(vlm: VLM, b: Batch, max_new: int, eos_ids: list[int], pad_id: int) -> tuple[list[list[int]], list[bool]]:
    """Returns new token ids per row (termination token excluded) and whether the row stopped by itself."""
    body, head = vlm.body, vlm.lm.lm_head
    out = body(inputs_embeds=vlm.embed(b), attention_mask=b.attn, position_ids=b.pos, use_cache=True)
    past = out.past_key_values
    nxt = head(out.last_hidden_state[:, -1]).float().argmax(-1)
    B = nxt.shape[0]
    eos = torch.tensor(eos_ids, device=nxt.device)
    done = torch.zeros(B, dtype=torch.bool, device=nxt.device)
    stopped = torch.zeros(B, dtype=torch.bool, device=nxt.device)
    toks: list[torch.Tensor] = []
    attn, pos = b.attn, b.pos[:, -1:]
    for step in range(max_new):
        is_eos = torch.isin(nxt, eos)
        stopped |= is_eos & ~done
        toks.append(torch.where(done | is_eos, torch.full_like(nxt, -1), nxt))
        done |= is_eos
        if bool(done.all()) or step == max_new - 1:
            break
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=attn.device)], dim=1)
        pos = pos + 1
        out = body(input_ids=nxt.unsqueeze(1), attention_mask=attn, position_ids=pos, past_key_values=past,
                   use_cache=True)
        past = out.past_key_values
        nxt = head(out.last_hidden_state[:, -1]).float().argmax(-1)
    T = torch.stack(toks, 1).tolist() if toks else [[] for _ in range(B)]
    seqs = []
    for row in T:
        r = []
        for t in row:
            if t == -1:
                break
            r.append(t)
        seqs.append(r)
    return seqs, stopped.tolist()


def timed_greedy(vlm: VLM, b: Batch, max_new: int, eos_ids: list[int], pad_id: int, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    seqs, stopped = greedy(vlm, b, max_new, eos_ids, pad_id)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return seqs, stopped, time.perf_counter() - t0
