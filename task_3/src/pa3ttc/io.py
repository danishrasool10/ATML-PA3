"""Small torch-free I/O helpers (the offline stages never import torch)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _clean(o: Any) -> Any:
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=2, default=_default))
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(_clean(row), default=_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:   # a line cut off by a killed process
                pass
    return rows


def pin_cpus(var: str = "PA3_CPUS") -> list[int] | None:
    """Restrict this process to the CPU cores listed in env `var` (e.g. "2,3")."""
    spec = os.environ.get(var)
    if not spec or not hasattr(os, "sched_setaffinity"):
        return None
    cores = {int(c) for c in spec.split(",") if c.strip()}
    try:
        os.sched_setaffinity(0, cores)
    except OSError:
        return None
    return sorted(os.sched_getaffinity(0))


def df_to_md(df, floatfmt: str = ".4f", index: bool = False) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=True):
        idx, values = row[0], row[1:]
        vals = [str(idx)] if index else []
        for v in values:
            if isinstance(v, (float, np.floating)):
                vals.append("" if not np.isfinite(v) else format(v, floatfmt))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"
