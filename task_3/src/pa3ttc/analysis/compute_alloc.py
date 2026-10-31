"""Step 6: compute allocation — does the n-th sample add a new answer, the first correct answer, or a repeat?
Held-out questions, averaged over the 10 orders then over questions.  Also the coverage-vs-selection gap per
method and budget (oracle coverage minus achieved accuracy)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import NO_ANSWER, odir
from ..io import df_to_md
from ..offline.runner import LABEL, load_inputs


def marginal(ds: str) -> pd.DataFrame:
    pool, group, correct, sc, vc, orders, split = load_inputs(ds)
    qpos = {q: i for i, q in enumerate(pool.qids)}
    ev = [qpos[q] for q in split["eval"]]
    Q, K, R = orders.shape
    new = np.zeros((len(ev), K, R))
    new_ans = np.zeros_like(new)
    first_corr = np.zeros_like(new)
    for i, q in enumerate(ev):
        for k in range(K):
            seen, have_c = set(), False
            for n, r in enumerate(orders[q, k]):
                g = int(group[q, r])
                if g not in seen:
                    new[i, k, n] = 1
                    new_ans[i, k, n] = g != NO_ANSWER
                    seen.add(g)
                if correct[q, r] and not have_c:
                    first_corr[i, k, n] = 1
                    have_c = True
    df = pd.DataFrame({"n": np.arange(1, R + 1), "p_new_group": new.mean(1).mean(0),
                       "p_new_answered_group": new_ans.mean(1).mean(0),
                       "p_first_correct_appears": first_corr.mean(1).mean(0)})
    df["p_repeat"] = 1 - df["p_new_group"]
    df["coverage_cum"] = df["p_first_correct_appears"].cumsum()
    return df


def gap_table(ds: str) -> pd.DataFrame:
    s = pd.read_csv(odir("offline", ds) / "summary_eval.csv")
    s["gap"] = s["oracle"] - s["acc"]
    t = s.pivot_table(index="method", columns="N", values="gap")
    t = t.loc[[m for m in LABEL if m in t.index]]
    t.index = [LABEL[m] for m in t.index]
    t.columns = [f"N={c}" for c in t.columns]
    return t


def run(cfg: dict) -> dict:
    out = {}
    for ds in ("math", "mathvista"):
        d = odir("compute", ds)
        m = marginal(ds)
        m.to_csv(d / "marginal_samples.csv", index=False)
        g = gap_table(ds)
        g.to_csv(d / "selection_gap.csv")
        (d / "compute.md").write_text(
            f"## {ds}: what the n-th sample adds (held-out)\n\n"
            + df_to_md(m[m["n"].isin([1, 2, 3, 4, 8, 16, 32])]) + "\n## oracle coverage - accuracy (selection gap)\n\n"
            + df_to_md(g, index=True))
        out[ds] = {"p_new_group_at_32": float(m["p_new_group"].iloc[-1])}
    return out
