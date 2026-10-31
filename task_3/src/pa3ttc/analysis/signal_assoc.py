"""Step 3a: within-question associations at N = 32 between rollout length, scores and correctness.

For each held-out question: Spearman(signal, correct), Spearman(signal, length); correctness in the signal's top
quartile (8 best rollouts) minus its bottom quartile (8 worst; ties broken by rollout position, NaN = -inf).
Averages are reported over mixed-correctness questions (main, approved decision 9) and over all questions.
`neg_length` (shorter = better) is the length-only control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..config import odir
from ..io import df_to_md
from ..offline.runner import load_inputs


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.where(np.isnan(x), -np.inf, x)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def quartile_diff(score: np.ndarray, correct: np.ndarray) -> float:
    s = np.where(np.isnan(score), -np.inf, score)
    R = len(s)
    q = R // 4
    order = np.lexsort((np.arange(R), -s))        # best first; ties -> lower rollout position first
    return float(correct[order[:q]].mean() - correct[order[-q:]].mean())


def signals_at_32(ds, sc, vc, orders) -> dict[str, np.ndarray]:
    sig = {"neg_length": -sc["T"], "logp_L": sc["L"], "selfcert_K": sc["K"]}
    if ds == "mathvista":
        sig["vision_V"] = sc["V"]
        Q = orders.shape[0]
        for key, name in (("Z", "visconf_Z"), ("Ztok", "tokfuse_Ztok")):
            out = np.full_like(sc["V"], np.nan)
            for q in range(Q):
                out[q, orders[q, 0]] = vc[f"{key}_32"][q, 0]
            sig[name] = out
    return sig


def run_dataset(ds: str, cfg: dict) -> dict:
    pool, group, correct, sc, vc, orders, split = load_inputs(ds)
    qpos = {q: i for i, q in enumerate(pool.qids)}
    ev = np.array([qpos[q] for q in split["eval"]])
    corr = correct.astype(float)
    T = sc["T"]
    sig = signals_at_32(ds, sc, vc, orders)
    mixed = np.array([0 < corr[q].mean() < 1 for q in ev])
    rows = []
    for name, s in sig.items():
        sp_c = np.array([spearman(s[q], corr[q]) for q in ev])
        sp_t = np.array([spearman(s[q], T[q]) for q in ev])
        qd = np.array([quartile_diff(s[q], corr[q]) for q in ev])
        rows.append({"signal": name, "spearman_vs_correct_mixed": np.nanmean(sp_c[mixed]),
                     "n_mixed": int(mixed.sum()), "spearman_vs_length": np.nanmean(sp_t),
                     "top_minus_bottom_quartile_mixed": qd[mixed].mean(),
                     "top_minus_bottom_quartile_all": qd.mean(), "n_all": len(ev)})
    assoc = pd.DataFrame(rows)
    # length of correct vs incorrect rollouts within mixed questions
    lc, li = [], []
    for q in ev[mixed]:
        lc.append(T[q][correct[q]].mean())
        li.append(T[q][~correct[q]].mean())
    extra = {"mean_len_correct_mixed": float(np.mean(lc)) if lc else np.nan,
             "mean_len_incorrect_mixed": float(np.mean(li)) if li else np.nan,
             "frac_questions_mixed": float(mixed.mean()),
             "frac_all_correct": float(np.mean([corr[q].mean() == 1 for q in ev])),
             "frac_all_wrong": float(np.mean([corr[q].mean() == 0 for q in ev]))}
    d = odir("analysis", ds)
    assoc.to_csv(d / "associations.csv", index=False)
    (d / "associations.md").write_text(f"## {ds}: within-question associations at N=32 (held-out)\n\n"
                                       + df_to_md(assoc) + "\n" + "\n".join(f"- {k}: {v:.4f}" for k, v in extra.items()) + "\n")
    return {"associations": rows, **extra}
