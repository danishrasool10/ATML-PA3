"""Find the supplied ATML-PA3-Data folders (README.MD layout):

    ATML-PA3-Data/README.MD
    ATML-PA3-Data/qwen25-math/{generations,probability_metrics}.parquet
    ATML-PA3-Data/qwen25vl-mathvista/{generations,probability_metrics,attention_metrics}.parquet

Searched under data_root, then under all of /kaggle/input (wherever Kaggle mounts the attached dataset), following
symlinks.  The MathVista folder is the one with attention_metrics.parquet.  A listing of what was searched is
written to data/input_listing.txt so a failure can be diagnosed from the results zip.
"""
from __future__ import annotations

import os
from pathlib import Path

FILES = {"gen": "generations.parquet", "prob": "probability_metrics.parquet", "attn": "attention_metrics.parquet"}
FOLDERS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}


def _walk(root: Path, max_depth: int = 8):
    seen = set()
    base = len(Path(root).parts)
    for dp, dns, fns in os.walk(root, followlinks=True):
        rp = os.path.realpath(dp)
        if rp in seen or len(Path(dp).parts) - base > max_depth:
            dns[:] = []
            continue
        seen.add(rp)
        dns.sort()
        yield Path(dp), dns, sorted(fns)


def listing(roots: list[Path], max_lines: int = 400) -> str:
    out = []
    for root in map(Path, roots):
        if not root.exists():
            out.append(f"{root}  (does not exist)")
            continue
        for dp, _, fns in _walk(root, max_depth=6):
            depth = len(dp.parts) - len(root.parts)
            out.append("  " * depth + f"{dp.name or dp}/")
            for f in fns[:30]:
                try:
                    out.append("  " * (depth + 1) + f"{f}  ({(dp / f).stat().st_size / 2**20:.1f} MiB)")
                except OSError:
                    out.append("  " * (depth + 1) + f)
            if len(out) > max_lines:
                return "\n".join(out + ["... (truncated)"])
    return "\n".join(out)


def find_folders(data_root: str | Path, fallback: str | Path = "/kaggle/input",
                 listing_path: Path | None = None) -> dict[str, Path]:
    roots = [Path(data_root)] + ([Path(fallback)] if Path(fallback) != Path(data_root) else [])
    if listing_path is not None:
        listing_path.parent.mkdir(parents=True, exist_ok=True)
        listing_path.write_text(listing(roots))
    for root in roots:
        if not root.exists():
            continue
        found: dict[str, list[Path]] = {"math": [], "mathvista": []}
        for dp, _, fns in _walk(root):
            if FILES["gen"] in fns and FILES["prob"] in fns:
                found["mathvista" if FILES["attn"] in fns else "math"].append(dp)
        if found["math"] and found["mathvista"]:
            out = {}
            for ds, cands in found.items():
                named = [c for c in cands if c.name == FOLDERS[ds]]
                pick = named or cands
                if len(pick) > 1:
                    raise ValueError(f"several {ds} folders found: {[str(c) for c in pick]}; set DATA_ROOT")
                out[ds] = pick[0]
            return out
    raise FileNotFoundError(
        "ATML-PA3-Data not found: need qwen25-math/ (generations + probability_metrics parquet) and "
        "qwen25vl-mathvista/ (+ attention_metrics parquet). Is the dataset attached (Add Input)? Searched:\n"
        + listing(roots, max_lines=60))


def find_readme(folders: dict[str, Path]) -> Path | None:
    for d in folders.values():
        for base in (d.parent, d):
            for p in sorted(base.glob("*")):
                if p.is_file() and p.name.lower() == "readme.md":
                    return p
    return None
