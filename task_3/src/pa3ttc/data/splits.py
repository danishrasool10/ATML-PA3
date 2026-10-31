"""Development / held-out splits, the live-generation subset, and candidate orders.

Splits (per dataset, fresh generator): sort question IDs as strings, permute with default_rng(6304); first 200
= dev, remaining 800 = held-out eval; the live subset is the first 50 MATH eval IDs in permutation order.
Orders: per question (index q in the sorted list) default_rng([6304, d, q]) draws 10 permutations of the 32
rollout positions (d = 0 MATH, 1 MathVista).  Stateless, so any subset reproduces the same orders.
"""
from __future__ import annotations

import numpy as np


def make_split(qids: list[str], seed: int, n_dev: int) -> dict:
    s = np.array(sorted(str(q) for q in qids))
    perm = np.random.default_rng(seed).permutation(len(s))
    p = s[perm]
    return {"seed": seed, "rule": "sorted(str ids) -> default_rng(seed).permutation; dev=first n_dev",
            "sorted": s.tolist(), "permutation": perm.tolist(), "permuted": p.tolist(),
            "dev": p[:n_dev].tolist(), "eval": p[n_dev:].tolist()}


def make_orders(n_q: int, d_index: int, seed: int, n_orders: int, n_roll: int) -> np.ndarray:
    out = np.empty((n_q, n_orders, n_roll), dtype=np.int16)
    for q in range(n_q):
        rng = np.random.default_rng([seed, d_index, q])
        for k in range(n_orders):
            out[q, k] = rng.permutation(n_roll)
    return out
