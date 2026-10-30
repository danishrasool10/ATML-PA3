"""YAML config loading: configs/base.yaml + a run config + `key=value` overrides (YAML-parsed values)."""
from __future__ import annotations

from pathlib import Path

import yaml

TASK = Path(__file__).resolve().parents[2]
CONFIGS = TASK / "configs"


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None, sets: list[str] | None = None, base: str | Path | None = None) -> dict:
    cfg = yaml.safe_load(Path(base or CONFIGS / "base.yaml").read_text())
    if path:
        p = Path(path)
        if not p.exists():
            p = CONFIGS / f"{path}.yaml"
        cfg = _merge(cfg, yaml.safe_load(p.read_text()) or {})
    for s in sets or []:
        k, v = s.split("=", 1)
        node = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        node[keys[-1]] = yaml.safe_load(v)
    return cfg


def load_data_config(path: str | Path | None = None, sets: list[str] | None = None) -> dict:
    return load_config(None, sets, base=path or CONFIGS / "data.yaml")
