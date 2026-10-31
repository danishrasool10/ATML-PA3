"""Order-independent rollout signals: T_i, L_i = mean log p(sampled token), K_i = mean KL(U || p),
V_i = mean visual engagement v_t = A_img / (A_img + A_prompt + A_generated) over tokens where it is defined.
T_i = 0 gives NaN scores, which selection treats as -inf (approved decision 10)."""
from __future__ import annotations

import numpy as np


def seg_mean(vals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    lens = np.diff(offsets)
    x = vals.astype(np.float64)
    ok = np.isfinite(x)
    P = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    N = np.concatenate([[0], np.cumsum(ok)])
    s = P[offsets[1:]] - P[offsets[:-1]]
    n = N[offsets[1:]] - N[offsets[:-1]]
    out = np.full(len(lens), np.nan)
    m = n > 0
    out[m] = s[m] / n[m]
    return out


def rollout_scores(pool) -> dict[str, np.ndarray]:
    Q, R = pool.Q, pool.R
    out = {"T": pool.T.astype(np.float64),
           "L": seg_mean(pool.logp, pool.offsets).reshape(Q, R),
           "K": seg_mean(pool.kl, pool.offsets).reshape(Q, R)}
    if pool.v is not None:
        out["V"] = seg_mean(pool.v, pool.offsets).reshape(Q, R)
    for k in ("L", "K", "V"):
        if k in out:
            out[k][pool.T == 0] = np.nan
    return out
