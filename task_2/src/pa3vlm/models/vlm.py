"""SigLIP-features -> connector -> Qwen3 wrapper.

Visual embeddings are spliced into the frozen input embeddings with an out-of-place masked_scatter, so
autograd reaches the connector although every Qwen parameter has requires_grad=False (the LM forward is
NOT under no_grad).  The LM head is applied only at supervised positions (shifted labels != -100).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..data.collate import Batch
from .connector import Connector


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    return tok


def load_lm(model_id: str, dtype: torch.dtype, attn_impl: str = "sdpa", grad_ckpt: bool = True) -> nn.Module:
    from transformers import AutoModelForCausalLM

    lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn_impl)
    lm.config.use_cache = False
    for p in lm.parameters():
        p.requires_grad_(False)
    if grad_ckpt:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return lm


class VLM(nn.Module):
    def __init__(self, lm: nn.Module, connector: Connector):
        super().__init__()
        self.lm = lm
        self.connector = connector

    @property
    def body(self) -> nn.Module:
        return self.lm.model

    def embed(self, b: Batch) -> torch.Tensor:
        emb = self.lm.get_input_embeddings()(b.ids)
        if b.feats is None:
            assert not bool(b.vis.any())
            return emb
        v = self.connector(b.feats.to(self.connector.fc1.weight.device))
        assert v.shape[0] * v.shape[1] == int(b.vis.sum()), "visual slot count mismatch"
        return emb.masked_scatter(b.vis.unsqueeze(-1).expand_as(emb), v.to(emb.dtype))

    def hidden(self, b: Batch) -> torch.Tensor:
        out = self.body(inputs_embeds=self.embed(b), attention_mask=b.attn, position_ids=b.pos, use_cache=False)
        return out.last_hidden_state

    def supervised_nll(self, b: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(token NLL fp32, row index, is-termination) for every supervised token (next-token prediction)."""
        h = self.hidden(b)
        lab = b.labels[:, 1:]
        sel = lab != -100
        logits = self.lm.lm_head(h[:, :-1][sel]).float()
        nll = F.cross_entropy(logits, lab[sel], reduction="none")
        rows = sel.nonzero(as_tuple=True)[0]
        return nll, rows, b.term[:, 1:][sel]


def per_example_mean(nll: torch.Tensor, rows: torch.Tensor, n_rows: int, keep: torch.Tensor | None = None
                     ) -> torch.Tensor:
    """Mean token NLL per example (rows without kept tokens -> nan)."""
    if keep is not None:
        nll, rows = nll[keep], rows[keep]
    s = torch.zeros(n_rows, dtype=nll.dtype, device=nll.device).index_add_(0, rows, nll)
    c = torch.zeros(n_rows, dtype=nll.dtype, device=nll.device).index_add_(0, rows, torch.ones_like(nll))
    return s / c
