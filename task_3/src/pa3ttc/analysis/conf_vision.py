"""Step 3b (MathVista): confidence-vision relationship and the tokenwise vs trajectory-level fusion check.

Token level, per rollout: Pearson(c_t, v_t) with c_t calibrated on the full 32-rollout pool of its question, and
Pearson(raw KL_t, v_t); summarised per question (mean over rollouts) and across held-out questions.
Rollout level: Spearman(C_i, V_i) across the 32 rollouts of a question.
Identity check: Z_tok - Z = Cov_t(c_t, v_t) (population covariance) for rollouts where v_t is always defined.
Illustrative trajectories: fixed rule — held-out questions bucketed by (VisConf+RWC vs self-consistency at N=32,
order 0); 2 per bucket drawn with default_rng(6304) from the sorted question IDs; per question the highest- and
lowest-Z answered rollouts; rolling means (window 16) of c_t and v_t over generation position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import odir
from ..io import write_json
from ..offline.runner import load_inputs
from ..signals.visconf import token_c
from .signal_assoc import spearman


def _pearson(x, y) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _summ(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)), "frac_positive": float((x > 0).mean())}


def run(cfg: dict) -> dict:
    ds = "mathvista"
    pool, group, correct, sc, vc, orders, split = load_inputs(ds)
    R = pool.R
    qpos = {q: i for i, q in enumerate(pool.qids)}
    ev = [qpos[q] for q in split["eval"]]
    zpos = np.full((pool.Q, R), np.nan)
    ztpos = zpos.copy()
    for q in range(pool.Q):
        zpos[q, orders[q, 0]] = vc["Z_32"][q, 0]
        ztpos[q, orders[q, 0]] = vc["Ztok_32"][q, 0]
    rows, ident = [], []
    for q in ev:
        a, d = vc["a_32"][q, 0], vc["d_32"][q, 0]
        for r in range(R):
            s = pool.seg(q, r)
            k, v = pool.kl[s].astype(np.float64), pool.v[s].astype(np.float64)
            ok = np.isfinite(v)
            c = token_c(k, a, d)
            rho_cv = _pearson(c[ok], v[ok])
            rho_kv = _pearson(k[ok], v[ok])
            cov = float(np.mean(c * v) - c.mean() * v.mean()) if ok.all() and len(c) else np.nan
            if ok.all() and len(c) and np.isfinite(ztpos[q, r]):
                ident.append(abs((ztpos[q, r] - zpos[q, r]) - cov))
            rows.append({"qid": pool.qids[q], "rpos": r, "T": len(k), "correct": bool(correct[q, r]),
                         "rho_c_v": rho_cv, "rho_kl_v": rho_kv, "cov_c_v": cov, "C": np.nan, "V": sc["V"][q, r],
                         "Z": zpos[q, r], "Ztok": ztpos[q, r]})
    df = pd.DataFrame(rows)
    df["C"] = df["Z"] / df["V"]
    out = odir("analysis", ds)
    df.to_parquet(out / "conf_vision_per_rollout.parquet", index=False)
    perq = df.groupby("qid")[["rho_c_v", "rho_kl_v", "cov_c_v"]].mean()
    roll_level = np.array([spearman(sc["V"][q], df[df["qid"] == pool.qids[q]]["C"].to_numpy()) for q in ev])
    summary = {
        "token_level_rho_c_v_per_question": _summ(perq["rho_c_v"].to_numpy()),
        "token_level_rho_kl_v_per_question": _summ(perq["rho_kl_v"].to_numpy()),
        "token_level_rho_c_v_correct_rollouts": _summ(df[df["correct"]]["rho_c_v"].to_numpy()),
        "token_level_rho_c_v_incorrect_rollouts": _summ(df[~df["correct"]]["rho_c_v"].to_numpy()),
        "cov_c_v_per_question": _summ(perq["cov_c_v"].to_numpy()),
        "rollout_level_spearman_C_V": _summ(roll_level),
        "identity_check": {"n_rollouts": len(ident), "max_abs_error": float(max(ident)) if ident else None},
        "mean_Ztok_minus_Z": float(np.nanmean(df["Ztok"] - df["Z"])),
    }
    write_json(out / "conf_vision_summary.json", summary)
    summary["trajectories"] = trajectories(pool, correct, sc, vc, orders, split, cfg, zpos)
    return summary


def trajectories(pool, correct, sc, vc, orders, split, cfg, zpos) -> dict:
    po = pd.read_parquet(odir("offline", "mathvista") / "per_order.parquet",
                         filters=[("N", "==", 32), ("order", "==", 0)])
    po = po[po["method"].isin(["sc", "visconf_rwc"]) & (po["split"] == "eval")]
    w = po.pivot_table(index="qid", columns="method", values=["correct", "group"], observed=True)
    buckets = {
        "agree_both_correct": w[(w[("group", "sc")] == w[("group", "visconf_rwc")]) & (w[("correct", "sc")] == 1)],
        "disagree_visconf_correct": w[(w[("correct", "visconf_rwc")] == 1) & (w[("correct", "sc")] == 0)],
        "disagree_sc_correct": w[(w[("correct", "visconf_rwc")] == 0) & (w[("correct", "sc")] == 1)],
        "agree_both_wrong": w[(w[("group", "sc")] == w[("group", "visconf_rwc")]) & (w[("correct", "sc")] == 0)],
    }
    rng = np.random.default_rng(cfg["seed"])
    qpos = {q: i for i, q in enumerate(pool.qids)}
    picked: dict = {}
    series = []
    for name, b in buckets.items():
        ids = sorted(b.index.tolist())
        sel = rng.choice(ids, size=min(2, len(ids)), replace=False).tolist() if ids else []
        picked[name] = {"n_in_bucket": len(ids), "selected": sel}
        for qid in sel:
            q = qpos[qid]
            z = zpos[q].copy()
            z[pool.T[q] == 0] = np.nan
            if not np.isfinite(z).any():
                continue
            a, d = vc["a_32"][q, 0], vc["d_32"][q, 0]
            for tag, r in (("highest_Z", int(np.nanargmax(z))), ("lowest_Z", int(np.nanargmin(z)))):
                s = pool.seg(q, r)
                c = token_c(pool.kl[s], a, d)
                v = pool.v[s].astype(np.float64)
                win = 16
                cr = pd.Series(c).rolling(win, min_periods=1).mean().to_numpy()
                vr = pd.Series(v).rolling(win, min_periods=1).mean().to_numpy()
                step = max(1, len(c) // 200)
                series.append({"bucket": name, "qid": qid, "which": tag, "rpos": r,
                               "rollout_id": pool.rollout_ids[q, r], "correct": bool(correct[q, r]),
                               "Z": float(z[r]), "T": len(c), "c_roll": cr[::step].tolist(),
                               "v_roll": vr[::step].tolist(), "step": step})
    write_json(odir("analysis", "mathvista") / "trajectories.json", {"rule": __doc__.split("Illustrative")[1].strip(),
                                                                      "picked": picked, "series": series})
    return picked
