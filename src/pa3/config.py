"""Run configuration: a flat dataclass loaded from base.yaml + a method yaml + CLI overrides."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class RunConfig:
    # ---- identity -------------------------------------------------------------------------
    run_name: str = "full_ft"
    method: str = "full"  # one of: full | l2sp | lora | oplora
    model_id: str = "Qwen/Qwen3-0.6B"
    seed: int = 6304

    # ---- data -----------------------------------------------------------------------------
    n_examples: int = 40_000
    max_len: int = 2048
    data_dir: str = "data"  # (relative to $PA3_OUTPUT_ROOT) manifest + pre-tokenized arrays (see data/prepare.py)

    # ---- batching (expressed in single-GPU terms; per-GPU accumulation = grad_accum // world) -
    micro_bs: int = 2
    grad_accum: int = 16  # global accumulation steps of `micro_bs` -> effective batch 32
    num_updates: int = 1250

    # ---- optimisation ---------------------------------------------------------------------
    lr: float = 2e-5
    warmup_updates: int = 38
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    clip_norm: float = 1.0
    zero1: bool = False  # ZeroRedundancyOptimizer (shard AdamW states across ranks)

    # ---- precision / memory ---------------------------------------------------------------
    amp: bool = True  # FP16 autocast + dynamic GradScaler (auto-disabled on CPU)
    init_scale: float = 65536.0
    grad_checkpointing: bool = True
    ce_chunk: int = 1024  # response positions per lm_head/CE chunk
    attn_impl: str = "sdpa"

    # ---- L2-SP ----------------------------------------------------------------------------
    l2sp_lambda: float = 0.0
    l2sp_ref_device: str = "cuda"  # "cuda" | "cpu" (pinned host memory, streamed per param)

    # ---- LoRA / OPLoRA --------------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = TARGET_MODULES
    oplora_k: int = 128
    bases_path: str = "bases/oplora_bases_k128.safetensors"

    # ---- checkpointing --------------------------------------------------------------------
    output_root: str = "runs"  # relative to $PA3_OUTPUT_ROOT
    recovery_updates: tuple[int, ...] = (312, 625, 937, 1250)
    weight_snapshot_updates: tuple[int, ...] = (1250,)  # Full FT adds 625
    keep_last_recovery: int = 1  # rolling recovery checkpoints kept on disk
    drop_final_recovery: bool = True  # delete optimizer state once the run finished + exported

    # ---- debugging / smoke ----------------------------------------------------------------
    max_updates: int | None = None  # stop early (smoke tests); schedule still uses num_updates
    grad_dump_update: int | None = None  # dump synced/unscaled grads at this update and exit
    grad_dump_path: str | None = None
    crash_after_update: int | None = None  # simulate a killed session right after this update's ckpt
    log_every: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- derived --------------------------------------------------------------------------
    @property
    def effective_batch(self) -> int:
        return self.micro_bs * self.grad_accum

    @property
    def is_adapter(self) -> bool:
        return self.method in ("lora", "oplora")

    @property
    def run_dir(self) -> Path:
        root = Path(self.output_root)
        if not root.is_absolute():
            root = Path(os.environ.get("PA3_OUTPUT_ROOT", "outputs")) / root
        return root / self.run_name

    @property
    def lora_scale(self) -> float:
        return self.lora_alpha / self.lora_r

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["effective_batch"] = self.effective_batch
        return d

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _coerce(cfg: RunConfig, key: str, value: Any) -> Any:
    f = {f.name: f for f in dataclasses.fields(RunConfig)}[key]
    cur = getattr(cfg, key)
    if isinstance(value, str) and not isinstance(cur, str):
        value = yaml.safe_load(value)  # "1e-3" -> float, "[1,2]" -> list, "true" -> bool, "null"
    if isinstance(cur, tuple) and isinstance(value, list):
        value = tuple(value)
    if f.type in ("float",) and isinstance(value, int):
        value = float(value)
    return value


def load_config(*yaml_paths: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    cfg = RunConfig()
    names = {f.name for f in dataclasses.fields(RunConfig)}
    merged: dict[str, Any] = {}
    for p in yaml_paths:
        with open(p) as fh:
            merged.update(yaml.safe_load(fh) or {})
    merged.update(overrides or {})
    for k, v in merged.items():
        if k not in names:
            raise KeyError(f"Unknown config key: {k}")
        setattr(cfg, k, _coerce(cfg, k, v))
    if cfg.method not in ("full", "l2sp", "lora", "oplora"):
        raise ValueError(cfg.method)
    if cfg.method != "l2sp" and cfg.l2sp_lambda != 0.0:
        raise ValueError("Only the l2sp method may use a non-zero lambda")
    return cfg


def parse_cli(argv: list[str] | None = None) -> RunConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True, help="base.yaml then method.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    a = ap.parse_args(argv)
    ov = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        ov[k] = v
    return load_config(*a.config, overrides=ov)
