"""Find the supplied ATML-PA3-Data folders (LLM folder = no attention file, VLM folder = has one)."""
from __future__ import annotations

from pathlib import Path


def find_model_dirs(preferred: str | Path, fallback: str | Path = "/kaggle/input") -> dict[str, Path]:
    roots = [Path(preferred)] + ([Path(fallback)] if Path(fallback) != Path(preferred) else [])
    for root in roots:
        if not root.exists():
            continue
        gens = sorted(root.rglob("generations.parquet"))
        out: dict[str, Path] = {}
        for g in gens:
            d = g.parent
            kind = "mathvista" if (d / "attention_metrics.parquet").exists() else "math"
            if not (d / "probability_metrics.parquet").exists():
                continue
            if kind in out and out[kind] != d:
                raise ValueError(f"two {kind} folders found: {out[kind]} and {d}")
            out[kind] = d
        if {"math", "mathvista"} <= set(out):
            return out
    raise FileNotFoundError(f"ATML-PA3-Data not found under {[str(r) for r in roots]}: need one folder with "
                            "generations.parquet + probability_metrics.parquet and one that also has "
                            "attention_metrics.parquet. Attach the dataset (Add Input) and set data_root.")


def find_readme(model_dirs: dict[str, Path]) -> Path | None:
    seen = set()
    for d in model_dirs.values():
        for base in (d, d.parent, d.parent.parent):
            if base in seen:
                continue
            seen.add(base)
            for p in sorted(base.glob("*")):
                if p.is_file() and p.name.lower().startswith("readme"):
                    return p
    return None
