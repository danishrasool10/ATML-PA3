"""The seven evaluation conditions and how to materialise each as an FP16 model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ..models.load import load_full_weights, load_model, load_safetensors
from ..models.lora import merge_into
from ..utils.common import find_artifact, read_json


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str  # base | full | adapter
    run: str | None = None
    update: int | None = None

    @property
    def weights_rel(self) -> str | None:
        if self.kind == "base":
            return None
        fname = "model.safetensors" if self.kind == "full" else "adapter.safetensors"
        return f"runs/{self.run}/weights_u{self.update:04d}/{fname}"


CONDITIONS: dict[str, Condition] = {c.name: c for c in [
    Condition("baseline", "base"),
    Condition("fullft_u0625", "full", "full_ft", 625),
    Condition("fullft_u1250", "full", "full_ft", 1250),
    Condition("l2sp_u1250", "full", "full_ft_l2sp", 1250),
    Condition("lora4_u1250", "adapter", "lora_r4", 1250),
    Condition("lora16_u1250", "adapter", "lora_r16", 1250),
    Condition("oplora16_u1250", "adapter", "oplora_r16", 1250),
]}
ORDER = list(CONDITIONS)
# (runs that produce each condition — used by the notebook's stage planner)
RUN_OF = {c.name: c.run for c in CONDITIONS.values()}


def weights_path(cond: Condition) -> Path | None:
    if cond.kind == "base":
        return None
    p = find_artifact(cond.weights_rel)
    if p is None:
        raise FileNotFoundError(f"{cond.name}: missing {cond.weights_rel}")
    return p


def adapter_meta(cond: Condition) -> dict:
    return read_json(weights_path(cond).parent / "adapter_meta.json")


def build_fp32(cond: Condition, model_id: str, attn_impl: str = "sdpa") -> nn.Module:
    """FP32 model for this condition on CPU (merged for adapters; exact upcast for base)."""
    model = load_model(model_id, torch.float32, attn_impl)
    if cond.kind == "full":
        load_full_weights(model, weights_path(cond))
    elif cond.kind == "adapter":
        meta = adapter_meta(cond)
        merge_into(model, load_safetensors(weights_path(cond)), float(meta["scale"]))
    return model


def build_eval_model(cond: Condition, model_id: str, device: torch.device, attn_impl: str = "sdpa") -> nn.Module:
    """Merge in FP32 on CPU, cast to FP16 (all evaluations run in FP16, dropout off)."""
    m = build_fp32(cond, model_id, attn_impl).to(torch.float16).to(device).eval()
    m.config.use_cache = True
    g = m.generation_config
    g.do_sample, g.temperature, g.top_p, g.top_k = False, None, None, None
    return m
