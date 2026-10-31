"""Step 5: adaptive sampling simulated on the supplied pools (stopping between completed rollouts).

Rollouts are revealed in the saved orders; stopping is checked after n in {4, 8, 16}, otherwise all 32 are used.
Both controllers return the current self-consistency answer.
  agreement:     unique, non-NO_ANSWER leading answer with count / n >= tau
  quality-aware: agreement AND SC winner == RWC winner AND RWC margin >= gamma, where RWC uses self-certainty
                 (MATH) or VisConf recalibrated on the n revealed rollouts (MathVista); margin = (W1 - W2) / sum W
Features are precomputed per (question, order, n); a policy is then a vectorised first-true search.
"""
from __future__ import annotations

import numpy as np

from ..config import NO_ANSWER
from ..select.rules import rwc, vote, vote_margin

CHECKS_ALL = (4, 8, 16, 32)


def features(group, correct, T, qscore_fn, orders, checks=CHECKS_ALL) -> dict[str, np.ndarray]:
    """qscore_fn(q, k, o, n) -> quality scores of the first n candidates (candidate order)."""
    Q, K, R = orders.shape
    nC = len(checks)
    f = {k: np.zeros((Q, K, nC), t) for k, t in [("lead_frac", float), ("lead_unique", bool), ("lead_na", bool),
                                                   ("agree_rwc", bool), ("margin", float), ("sc_correct", bool),
                                                   ("cost", np.int64), ("sc_group", np.int32)]}
    for q in range(Q):
        for k in range(K):
            o = orders[q, k]
            g_o, c_o, T_o = group[q, o], correct[q, o], T[q, o]
            for ci, n in enumerate(checks):
                g = g_o[:n]
                win, idx, tot = vote(g)
                counts = sorted(tot.values(), reverse=True)
                rg, _, rtot = rwc(g, qscore_fn(q, k, o, n))
                f["lead_frac"][q, k, ci] = counts[0] / n
                f["lead_unique"][q, k, ci] = len(counts) == 1 or counts[0] > counts[1]
                f["lead_na"][q, k, ci] = win == NO_ANSWER
                f["agree_rwc"][q, k, ci] = rg == win
                f["margin"][q, k, ci] = vote_margin(rtot)
                f["sc_correct"][q, k, ci] = c_o[idx]
                f["sc_group"][q, k, ci] = win
                f["cost"][q, k, ci] = int(T_o[:n].sum())
    return f


def stop_index(f: dict, tau: float | None, gamma: float | None, family: str, n_checks: int = 3) -> np.ndarray:
    """Index into CHECKS_ALL where each (q, k) stops (n_checks = last index = 32 if nothing triggers).
    tau=None -> the always-32 fallback policy."""
    Q, K, _ = f["lead_frac"].shape
    if tau is None:
        return np.full((Q, K), n_checks, dtype=np.int64)
    cond = f["lead_unique"] & ~f["lead_na"] & (f["lead_frac"] >= tau - 1e-12)
    if family == "quality":
        cond = cond & f["agree_rwc"] & (f["margin"] >= gamma - 1e-12)
    cond = cond[:, :, :n_checks]
    any_ = cond.any(axis=2)
    first = cond.argmax(axis=2)
    return np.where(any_, first, n_checks)


def evaluate(f: dict, stop: np.ndarray, qmask: np.ndarray | None = None) -> dict:
    """Mean over orders within a question, then over questions."""
    take = lambda a: np.take_along_axis(a, stop[:, :, None], axis=2)[:, :, 0]  # noqa: E731
    acc_q = take(f["sc_correct"]).mean(1)
    cost_q = take(f["cost"]).mean(1)
    n_q = np.array(CHECKS_ALL)[stop].mean(1)
    if qmask is not None:
        acc_q, cost_q, n_q = acc_q[qmask], cost_q[qmask], n_q[qmask]
    return {"acc": float(acc_q.mean()), "cost": float(cost_q.mean()), "rollouts": float(n_q.mean()),
            "acc_q": acc_q, "cost_q": cost_q, "n_q": n_q}
