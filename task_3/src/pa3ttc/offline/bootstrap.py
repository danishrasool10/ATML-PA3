"""Paired question-level bootstrap (resample questions, never rollouts) on per-question values that were already
averaged over the 10 candidate orders."""
from __future__ import annotations

import numpy as np


def boot_idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def paired(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    bm = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(bm, 2.5)), "hi": float(np.percentile(bm, 97.5)),
            "a": float(np.mean(a)), "b": float(np.mean(b)), "n": int(len(d))}


def mean_ci(a: np.ndarray, idx: np.ndarray) -> dict:
    a = np.asarray(a, float)
    bm = a[idx].mean(axis=1)
    return {"mean": float(a.mean()), "lo": float(np.percentile(bm, 2.5)), "hi": float(np.percentile(bm, 97.5))}
