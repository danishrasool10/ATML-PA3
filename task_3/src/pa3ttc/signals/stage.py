"""Signals stage: per-rollout L, K, V, T for both datasets; prefix-calibrated VisConf / tokenwise fusion for
MathVista at every budget N >= 2 and every saved order (scores stored in candidate order)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import odir
from ..data.load import Pool
from ..io import write_json
from .rollout import rollout_scores
from .visconf import VisConfIndex


def run(cfg: dict) -> dict:
    info = {}
    for ds in ("math", "mathvista"):
        pool = Pool(ds)
        sc = rollout_scores(pool)
        d = odir("signals", ds)
        np.savez(d / "rollout_scores.npz", **sc)
        df = pd.DataFrame({"qid": np.repeat(pool.qids, pool.R), "rollout_id": pool.rollout_ids.ravel(),
                           **{k: v.ravel() for k, v in sc.items()}})
        df.to_parquet(d / "rollout_scores.parquet", index=False)
        info[ds] = {k: {"mean": float(np.nanmean(v)), "nan": int(np.isnan(v).sum())} for k, v in sc.items()}
        if ds != "mathvista":
            continue
        t0 = time.time()
        orders = np.load(odir("splits") / f"{ds}_orders.npy")
        idx = VisConfIndex(pool.offsets, pool.kl, pool.v)
        Q, K, R = orders.shape
        V = sc["V"]
        out = {}
        for N in [n for n in cfg["budgets"] if n >= 2]:
            Z = np.full((Q, K, N), np.nan, np.float64)
            C, Zt = Z.copy(), Z.copy()
            A, D = np.full((Q, K), np.nan), np.full((Q, K), np.nan)
            for q in range(Q):
                if N == R:   # same candidate set for every order: compute once, then permute
                    pos = np.arange(R)
                    c, z, zt, a, dd = idx.scores(q * R + pos, V[q, pos])
                    for k in range(K):
                        o = orders[q, k]
                        C[q, k], Z[q, k], Zt[q, k], A[q, k], D[q, k] = c[o], z[o], zt[o], a, dd
                    continue
                for k in range(K):
                    pos = orders[q, k, :N]
                    C[q, k], Z[q, k], Zt[q, k], A[q, k], D[q, k] = idx.scores(q * R + pos, V[q, pos])
            out.update({f"C_{N}": C, f"Z_{N}": Z, f"Ztok_{N}": Zt, f"a_{N}": A, f"d_{N}": D})
        np.savez(d / "visconf.npz", **out)
        info[ds]["visconf_seconds"] = round(time.time() - t0, 1)
    write_json(odir("signals") / "summary.json", info)
    print(f"[signals] {info}", flush=True)
    return info
