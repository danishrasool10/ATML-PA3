"""Pair bootstrap (complete complementary pairs are the resampling unit).

Within a suite (and question-type subset) every metric and every condition uses the *same* resampled pair
indices, so condition differences (e.g. correct - swap) are paired.  Percentile 95% intervals.
"""
from __future__ import annotations

import numpy as np


def resample_indices(n_pairs: int, n_boot: int, seed_parts: list[int]) -> np.ndarray:
    return np.random.default_rng(seed_parts).integers(0, n_pairs, size=(n_boot, n_pairs))


def ci(values: np.ndarray, idx: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    point = float(values.mean())
    boots = values[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def diff_ci(a: np.ndarray, b: np.ndarray, idx: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """CI of mean(a) - mean(b) with a, b aligned on the same pairs."""
    assert a.shape == b.shape
    return ci(a - b, idx, alpha)
