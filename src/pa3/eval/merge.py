"""Merged-vs-unmerged logit check for adapter conditions.

Unmerged = training-time computation: fresh backbone + LoRALinear(raw A, B) (+ OPLoRA
projection recomputed from the stored bases).  Merged = W0 + (alpha/r) B_eff A_eff.
Checked twice: in FP32 (should agree to ~1e-5, verifies the algebra and the exported
projected factors) and in FP16 inference precision (the configuration actually evaluated).
"""
from __future__ import annotations

import torch

from ..data.formatting import render_prompt
from ..models.load import load_model, load_safetensors
from ..models.lora import adapter_modules, inject_adapters, load_factors_into_modules, merge_into
from ..models.oplora import load_bases
from ..utils.common import find_artifact
from .conditions import Condition, adapter_meta, weights_path

CHECK_PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?",
    "Write a short haiku about autumn leaves.",
    "Hello! I can fulfill your one wish. What is the one thing you want?",
    "Explain why the sky is blue in two sentences.",
]


@torch.no_grad()
def _logits(model, batch, autocast: bool):
    ctx = torch.autocast("cuda", dtype=torch.float16) if autocast else torch.autocast("cpu", enabled=False)
    with ctx:
        return model(**batch).logits.float()


@torch.no_grad()
def merge_check(cond: Condition, model_id: str, tok, device: torch.device, bases_rel: str,
                targets: tuple[str, ...], seed: int = 6304) -> dict:
    meta = adapter_meta(cond)
    factors = load_safetensors(weights_path(cond))
    bases = None
    if meta["method"] == "oplora":
        bases = load_bases(find_artifact(bases_rel))
    texts = [render_prompt(tok, p) for p in CHECK_PROMPTS]
    tok.padding_side = "left"
    batch = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    batch = {k: v.to(device) for k, v in batch.items()}
    out = {}
    use_amp = device.type == "cuda"
    precisions = [("fp32", torch.float32, False)] + ([("fp16", torch.float16, True)] if use_amp else [])
    for tag, dt, amp in precisions:
        unm = load_model(model_id, dt)
        inject_adapters(unm, int(meta["r"]), float(meta["alpha"]), 0.0, seed, targets, bases)
        load_factors_into_modules(unm, factors)
        unm.to(device).eval()
        # exported effective factors must equal the projection recomputed from raw factors
        fac_err = 0.0
        for n, m in adapter_modules(unm).items():
            A_eff, B_eff = m.effective_factors()
            fac_err = max(fac_err, float((A_eff.cpu() - factors[f"{n}.A_eff"]).abs().max()),
                          float((B_eff.cpu() - factors[f"{n}.B_eff"]).abs().max()))
        lu = _logits(unm, batch, amp)
        del unm
        mer = load_model(model_id, torch.float32)
        merge_into(mer, factors, float(meta["scale"]))
        mer = mer.to(dt).to(device).eval()
        lm = _logits(mer, batch, amp)
        del mer
        mask = batch["attention_mask"].bool()
        diff = (lu - lm).abs()[mask]
        out[tag] = {
            "max_abs_logit_diff": float(diff.max()),
            "mean_abs_logit_diff": float(diff.mean()),
            "top1_agreement": float((lu.argmax(-1) == lm.argmax(-1))[mask].float().mean()),
            "exported_vs_recomputed_factor_max_err": fac_err,
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out
