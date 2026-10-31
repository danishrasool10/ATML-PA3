"""VisConf with candidate-set calibration, computed exactly but fast.

For a candidate set G: a = Q.05, b = Q.95 of the pooled token KL values of G (linear interpolation), d =
max(b - a, 1e-8), c_t = clip((k_t - a) / d, 0, 1), C_i = mean_t c_t, Z_i = V_i * C_i, and the tokenwise
ablation Z_tok_i = mean_t c_t v_t (over tokens with defined v_t).

Each rollout's KL values are stored sorted, with its own prefix sums of k, v, k*v and of the valid-v count.
c_t is 0 for k <= a, 1 for k >= a + d and (k - a)/d in between, so with i0 = #(k <= a) and i1 = #(k < a + d):
  sum_t c_t   = (S_k[i1] - S_k[i0] - a (i1 - i0)) / d + (T - i1)
  sum_t c_t v = (S_kv[i1] - S_kv[i0] - a (S_v[i1] - S_v[i0])) / d + (S_v[T] - S_v[i1])
Only the tokens of G enter the percentiles, so unseen rollouts can never change a score (tested).
"""
from __future__ import annotations

import numpy as np


class VisConfIndex:
    def __init__(self, offsets: np.ndarray, kl: np.ndarray, v: np.ndarray):
        self.off = offsets.astype(np.int64)
        n = len(self.off) - 1
        seg = np.repeat(np.arange(n), np.diff(self.off))
        k = kl.astype(np.float64)
        order = np.lexsort((k, seg))
        self.sk = k[order]
        vv = v.astype(np.float64)[order]
        ok = np.isfinite(vv)
        v0 = np.where(ok, vv, 0.0)
        k0 = np.where(np.isfinite(self.sk), self.sk, 0.0)
        # per-rollout prefix sums (each segment starts from its own 0, so a rollout's sums never depend on the
        # values of any other rollout — not even through floating-point rounding); segment r lives at
        # [off[r] + r, off[r + 1] + r]
        m = len(self.sk) + n
        self.Pk, self.Pv, self.Pkv = np.zeros(m), np.zeros(m), np.zeros(m)
        self.Pn = np.zeros(m, dtype=np.int64)
        for r in range(n):
            s, e = self.off[r], self.off[r + 1]
            b = s + r
            if e > s:
                self.Pk[b + 1:b + 1 + e - s] = np.cumsum(k0[s:e])
                self.Pv[b + 1:b + 1 + e - s] = np.cumsum(v0[s:e])
                self.Pkv[b + 1:b + 1 + e - s] = np.cumsum(k0[s:e] * v0[s:e])
                self.Pn[b + 1:b + 1 + e - s] = np.cumsum(ok[s:e])

    def calibration(self, rows) -> tuple[float, float]:
        parts = [self.sk[self.off[r]:self.off[r + 1]] for r in rows]
        pooled = np.concatenate(parts) if parts else np.zeros(0)
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size == 0:
            return np.nan, np.nan
        a, b = np.percentile(pooled, [5, 95])
        return float(a), float(max(b - a, 1e-8))

    def scores(self, rows, V: np.ndarray | None = None):
        """rows: global rollout indices of the candidate set (any order).  V: their V_i (same order).
        -> (C, Z, Z_tok, a, d) with one entry per row."""
        rows = np.asarray(rows)
        a, d = self.calibration(rows)
        C = np.full(len(rows), np.nan)
        Zt = np.full(len(rows), np.nan)
        if np.isfinite(a):
            u = a + d
            for j, r in enumerate(rows):
                s, e = int(self.off[r]), int(self.off[r + 1])
                T = e - s
                if T == 0:
                    continue
                b = s + int(r)
                seg = self.sk[s:e]
                i0 = int(np.searchsorted(seg, a, "right"))
                i1 = int(np.searchsorted(seg, u, "left"))
                C[j] = ((self.Pk[b + i1] - self.Pk[b + i0] - a * (i1 - i0)) / d + (T - i1)) / T
                nv = self.Pn[b + T]
                if nv > 0:
                    num = ((self.Pkv[b + i1] - self.Pkv[b + i0] - a * (self.Pv[b + i1] - self.Pv[b + i0])) / d
                           + (self.Pv[b + T] - self.Pv[b + i1]))
                    Zt[j] = num / nv
        Z = C * V if V is not None else np.full(len(rows), np.nan)
        return C, Z, Zt, a, d


def naive_scores(segments_k: list[np.ndarray], segments_v: list[np.ndarray]):
    """Direct implementation of the formulas (tests only)."""
    pooled = np.concatenate([s for s in segments_k]) if segments_k else np.zeros(0)
    a, b = np.percentile(pooled, [5, 95])
    d = max(b - a, 1e-8)
    C, Z, Zt = [], [], []
    for k, v in zip(segments_k, segments_v):
        if len(k) == 0:
            C.append(np.nan), Z.append(np.nan), Zt.append(np.nan)
            continue
        c = np.clip((k - a) / d, 0, 1)
        ok = np.isfinite(v)
        Ci = c.mean()
        Vi = v[ok].mean() if ok.any() else np.nan
        C.append(Ci)
        Z.append(Ci * Vi)
        Zt.append((c[ok] * v[ok]).mean() if ok.any() else np.nan)
    return np.array(C), np.array(Z), np.array(Zt), a, d


def token_c(k: np.ndarray, a: float, d: float) -> np.ndarray:
    return np.clip((k.astype(np.float64) - a) / d, 0.0, 1.0)
