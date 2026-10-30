"""Batch construction: token ids with visual slots, attention mask, positions, labels.

Row layout: prefix | n_vis visual slots | suffix | target   (the visual slots are dropped for no-image rows).
- attention_mask: 1 on real tokens (text and visual), 0 on padding; the model applies causal masking.
- position_ids  : cumsum(attention_mask) - 1  -> contiguous 1-D RoPE positions over real tokens, for either
                  padding side; rebuilt automatically when the visual slots are removed.
- labels        : token id at target positions (answer/caption content + <|im_end|>), -100 elsewhere
                  (prefix, visual slots, suffix/prompt, padding).  Labels are *unshifted*; the loss shifts.
- term          : True at the termination-token label positions (excluded from answer NLL).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .prompts import Encoded


@dataclass
class Batch:
    ids: torch.Tensor
    vis: torch.Tensor
    attn: torch.Tensor
    pos: torch.Tensor
    labels: torch.Tensor
    term: torch.Tensor
    feats: torch.Tensor | None     # [n_rows_with_image, n_vis, 768], rows in batch order
    images: list

    def to(self, device: torch.device) -> "Batch":
        mv = {k: getattr(self, k).to(device, non_blocking=True) for k in ("ids", "vis", "attn", "pos", "labels", "term")}
        f = self.feats.to(device, non_blocking=True) if self.feats is not None else None
        return Batch(**mv, feats=f, images=self.images)


def collate(encs: list[Encoded], images: list[str | None], get_feats, n_vis: int, pad_id: int,
            side: str = "right", with_target: bool = True) -> Batch:
    assert len(encs) == len(images) and side in ("left", "right")
    rows = []
    for e, img in zip(encs, images):
        nv = n_vis if img is not None else 0
        tgt = e.target if with_target else []
        ids = list(e.prefix) + [pad_id] * nv + list(e.suffix) + list(tgt)
        vis = [False] * len(e.prefix) + [True] * nv + [False] * (len(e.suffix) + len(tgt))
        lab = [-100] * (len(ids) - len(tgt)) + list(tgt)
        term = [False] * (len(ids) - len(tgt)) + [False] * (len(tgt) - e.n_term) + [True] * min(e.n_term, len(tgt))
        rows.append((ids, vis, lab, term))
    L = max(len(r[0]) for r in rows)
    B = len(rows)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    vis = torch.zeros((B, L), dtype=torch.bool)
    attn = torch.zeros((B, L), dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    term = torch.zeros((B, L), dtype=torch.bool)
    for b, (i, v, l, t) in enumerate(rows):
        n = len(i)
        sl = slice(0, n) if side == "right" else slice(L - n, L)
        ids[b, sl] = torch.tensor(i)
        vis[b, sl] = torch.tensor(v)
        attn[b, sl] = 1
        lab[b, sl] = torch.tensor(l)
        term[b, sl] = torch.tensor(t)
    pos = (attn.cumsum(-1) - 1).clamp_min(0)
    keys = [k for k in images if k is not None]
    feats = get_feats(keys) if keys else None
    if feats is not None:
        assert feats.shape[:2] == (len(keys), n_vis), (feats.shape, n_vis)
    return Batch(ids, vis, attn, pos, lab, term, feats, list(images))
