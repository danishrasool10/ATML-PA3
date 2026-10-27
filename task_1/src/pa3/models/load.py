"""Model / tokenizer loading and weight (de)serialisation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
from torch import nn


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    return tok


def load_model(model_id: str, dtype: torch.dtype, attn_impl: str = "sdpa") -> nn.Module:
    """Load Qwen3 in `dtype`.  The hub checkpoint is bf16, so FP32 loading is exact."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn_impl)
    model.config.use_cache = False
    return model


def iter_target_linears(model: nn.Module, targets: tuple[str, ...]) -> Iterator[tuple[str, nn.Module]]:
    """Yield (qualified_name, module) for every targeted projection in every decoder layer.

    Names are canonical HF names, e.g. 'model.layers.0.mlp.down_proj', so they are stable
    across LoRA/OPLoRA/Full-FT runs.  LoRALinear wrappers are yielded themselves (not .base).
    """
    from .lora import LoRALinear

    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in targets and isinstance(mod, (nn.Linear, LoRALinear)) and ".layers." in f".{name}":
            yield name, mod


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Unique trainable parameters (tied weights appear once), detached, contiguous, CPU."""
    return {n: p.detach().to("cpu").contiguous().clone() for n, p in model.named_parameters() if p.requires_grad}


def save_safetensors(tensors: dict[str, torch.Tensor], path: str | Path, metadata: dict | None = None) -> int:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata={k: str(v) for k, v in (metadata or {}).items()})
    tmp.replace(path)
    return path.stat().st_size


def load_safetensors(path: str | Path, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path), device=str(device))


def load_full_weights(model: nn.Module, path: str | Path) -> None:
    """Load a Full-FT FP32 snapshot (unique-parameter state dict) into `model` (any dtype)."""
    sd = load_safetensors(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]  # tied to embed_tokens
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    if getattr(model.config, "tie_word_embeddings", False):
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
