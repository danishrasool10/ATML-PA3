"""Per-stage learning-rate schedule: linear warmup over w = ceil(warmup_frac * T) updates starting at peak/w
(update 1), then cosine decay reaching exactly final_frac * peak at the stage's last update T."""
from __future__ import annotations

import math


def warmup_updates(T: int, frac: float) -> int:
    return max(1, math.ceil(frac * T - 1e-9))


def lr_at(t: int, T: int, peak: float, warmup_frac: float = 0.05, final_frac: float = 0.1) -> float:
    """t = 1..T (1-based update index within the stage)."""
    assert 1 <= t <= T
    w = warmup_updates(T, warmup_frac)
    if t <= w:
        return peak * t / w
    prog = (t - w) / max(1, T - w)
    return peak * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * prog)))
