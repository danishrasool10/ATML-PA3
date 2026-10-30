"""Assemble tokenizer, prompt format, data bundle, feature source and the VLM from a run config."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from pa3common.utils import output_root

from ..data.bundle import Bundle
from ..data.prompts import ChatFormat
from ..features.store import FeatureStore, RandomFeatures
from .connector import make_connector
from .vlm import VLM, load_lm, load_tokenizer


def device_dtype(cfg: dict) -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda", 0), getattr(torch, cfg["dtype"])
    return torch.device("cpu"), torch.float32


def feature_source(cfg: dict):
    if cfg.get("features") == "random":
        return RandomFeatures(cfg["pool"])
    scratch = Path(os.environ.get("PA3_SCRATCH", "/tmp/pa3_scratch"))
    return FeatureStore(scratch / cfg["features_dir"], cfg["pool"])


def build(cfg: dict, device: torch.device, dtype: torch.dtype, grad_ckpt: bool | None = None):
    tok = load_tokenizer(cfg["model_id"])
    fmt = ChatFormat(tok, cfg["caps"])
    bundle = Bundle(output_root() / cfg["data_dir"], fmt)
    store = feature_source(cfg)
    lm = load_lm(cfg["model_id"], dtype, cfg["attn_impl"], cfg["grad_checkpointing"] if grad_ckpt is None else grad_ckpt)
    connector = make_connector(cfg["seed"], d_out=lm.config.hidden_size)
    vlm = VLM(lm, connector).to(device)
    vlm.connector.float()
    return tok, fmt, bundle, store, vlm
