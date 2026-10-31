"""Selection rules over one candidate prefix (arrays in candidate order: index 0 = first revealed).

Ties: highest/lowest score -> earliest candidate; answer votes -> the answer whose earliest supporting
candidate comes first.  RWC: average ranks among the N candidates (rank 1 = best, ties averaged),
w_i = (N - r_i)/(N - 1), W(a) = sum of w over a's candidates.  At N = 1 every rule returns the only candidate.
NaN scores (e.g. T = 0) count as -inf.  NO_ANSWER (-1) is an ordinary answer group in every vote.
"""
from __future__ import annotations

import numpy as np

TIE_EPS = 1e-9


def clean(scores) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    return np.where(np.isnan(s), -np.inf, s)


def argmax_earliest(scores) -> int:
    return int(np.argmax(clean(scores)))


def argmin_earliest(vals) -> int:
    v = np.asarray(vals, dtype=np.float64)
    return int(np.argmin(np.where(np.isnan(v), np.inf, v)))


def vote(groups, weights=None) -> tuple[int, int, dict]:
    """-> (winning group, index of its earliest supporter, totals in first-appearance order)."""
    tot: dict[int, float] = {}
    first: dict[int, int] = {}
    for i, g in enumerate(groups):
        g = int(g)
        if g not in tot:
            tot[g] = 0.0
            first[g] = i
        tot[g] += 1.0 if weights is None else float(weights[i])
    best = None
    for g in tot:                       # first-appearance order => ties go to the earliest supporter
        if best is None or tot[g] > tot[best] + TIE_EPS:
            best = g
    return best, first[best], tot


def avg_rank_desc(scores) -> np.ndarray:
    x = -clean(scores)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def rwc_weights(scores) -> np.ndarray:
    n = len(scores)
    if n == 1:
        return np.ones(1)
    return (n - avg_rank_desc(scores)) / (n - 1)


def rwc(groups, scores) -> tuple[int, int, dict]:
    if len(groups) == 1:
        return int(groups[0]), 0, {int(groups[0]): 1.0}
    return vote(groups, rwc_weights(scores))


def vote_margin(tot: dict) -> float:
    """(largest - second largest) / total weight; runner-up 0 when only one answer group exists."""
    v = sorted(tot.values(), reverse=True)
    s = sum(v)
    if s <= 0:
        return 0.0
    return (v[0] - (v[1] if len(v) > 1 else 0.0)) / s


def select(method: str, groups, scores=None, lengths=None) -> int:
    """Index (in candidate order) of the rollout a method returns.  For vote rules this is the earliest
    supporter of the winning answer (only its answer matters)."""
    n = len(groups)
    if n == 1 or method == "first":
        return 0
    if method == "sc":
        return vote(groups)[1]
    if method == "shortest":
        return argmin_earliest(lengths)
    if method.endswith("_rwc"):
        return rwc(groups, scores)[1]
    return argmax_earliest(scores)
