"""Configuration and output paths.

The config is `configs/task3.yaml`; `PA3_CONFIG` points elsewhere and `PA3_CONFIG_OVERRIDES` (JSON) patches
keys (used by the tests).  Outputs live under `PA3_OUTPUT_ROOT`; large caches under `PA3_SCRATCH`.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml

TASK = Path(__file__).resolve().parents[2]
DATASETS = ("math", "mathvista")
D_INDEX = {"math": 0, "mathvista": 1}
NO_ANSWER = -1          # group id of missing extractions (votes, always graded incorrect)


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config() -> dict:
    path = Path(os.environ.get("PA3_CONFIG", TASK / "configs" / "task3.yaml"))
    cfg = yaml.safe_load(path.read_text())
    ov = os.environ.get("PA3_CONFIG_OVERRIDES")
    if ov:
        cfg = _merge(cfg, json.loads(ov))
    return cfg


def out_root() -> Path:
    p = Path(os.environ.get("PA3_OUTPUT_ROOT", "pa3_task3_outputs"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def scratch() -> Path:
    p = Path(os.environ.get("PA3_SCRATCH", str(out_root() / "_scratch")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir(ds: str) -> Path:
    p = scratch() / "cache" / ds
    p.mkdir(parents=True, exist_ok=True)
    return p


def odir(*parts: str) -> Path:
    p = out_root().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
