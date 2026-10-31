"""Development tuning, freezing and held-out evaluation of the stopping controllers.

Per controller family the candidates are its grid plus the always-32 fallback.  Selected = lowest mean dev token
cost among settings whose dev accuracy >= fixed-32 SC dev accuracy - 1 pp (averaged over the ten orders); ties ->
higher accuracy, then smaller tau, then smaller gamma.  The choice is written to frozen.json BEFORE the held-out
questions are evaluated; the held-out evaluation reads that file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import odir
from ..io import df_to_md, read_json, write_json
from ..offline.bootstrap import boot_idx, paired
from ..offline.runner import load_inputs
from .controllers import CHECKS_ALL, evaluate, features, stop_index


def _qscore_fn(ds, sc, vc):
    if ds == "math":
        K = sc["K"]
        return lambda q, k, o, n: K[q, o[:n]]
    return lambda q, k, o, n: vc[f"Z_{n}"][q, k]


def grid(cfg) -> list[dict]:
    a = cfg["adaptive"]
    g = [{"family": "agreement", "tau": t, "gamma": None} for t in a["taus"]]
    g += [{"family": "agreement", "tau": None, "gamma": None}]
    g += [{"family": "quality", "tau": t, "gamma": gm} for t in a["taus"] for gm in a["gammas"]]
    g += [{"family": "quality", "tau": None, "gamma": None}]
    return g


def _key(r):
    inf = float("inf")
    return (round(r["cost"], 9), -round(r["acc"], 12), inf if r["tau"] is None else r["tau"],
            inf if r["gamma"] is None else r["gamma"])


def select_setting(rows: list[dict], target: float) -> dict:
    ok = [r for r in rows if r["acc"] >= target - 1e-12]
    return sorted(ok, key=_key)[0]


def run_dataset(ds: str, cfg: dict) -> dict:
    pool, group, correct, sc, vc, orders, split = load_inputs(ds)
    qpos = {q: i for i, q in enumerate(pool.qids)}
    dev = np.zeros(pool.Q, bool)
    dev[[qpos[q] for q in split["dev"]]] = True
    ev = np.zeros(pool.Q, bool)
    ev[[qpos[q] for q in split["eval"]]] = True
    f = features(group, correct, pool.T, _qscore_fn(ds, sc, vc), orders)
    d = odir("adaptive", ds)
    np.savez(d / "features.npz", **f)
    tol = cfg["adaptive"]["tolerance_pp"] / 100.0
    sc32_dev = evaluate(f, stop_index(f, None, None, "agreement"), dev)["acc"]
    rows = []
    for s in grid(cfg):
        r = evaluate(f, stop_index(f, s["tau"], s["gamma"], s["family"]), dev)
        rows.append({**s, "acc": r["acc"], "cost": r["cost"], "rollouts": r["rollouts"]})
    pd.DataFrame(rows).to_csv(d / "dev_grid.csv", index=False)
    frozen = {"sc32_dev_acc": sc32_dev, "target_acc": sc32_dev - tol, "rule": __doc__.strip()}
    for fam in ("agreement", "quality"):
        frozen[fam] = select_setting([r for r in rows if r["family"] == fam], sc32_dev - tol)
    write_json(d / "frozen.json", frozen)
    return heldout(ds, cfg, f, ev, pool)


def heldout(ds: str, cfg: dict, f: dict, ev: np.ndarray, pool) -> dict:
    d = odir("adaptive", ds)
    frozen = read_json(d / "frozen.json")            # evaluation uses only the frozen file
    idx = boot_idx(int(ev.sum()), cfg["n_boot"], cfg["seed"])
    pqo = pd.read_parquet(odir("offline", ds) / "per_question.parquet")
    pqo = pqo[(pqo["method"] == "sc") & (pqo["split"] == "eval")]
    qorder = [pool.qids[i] for i in np.where(ev)[0]]
    fixed = {}
    for N in cfg["budgets"]:
        x = pqo[pqo["N"] == N].set_index("qid").loc[qorder]
        fixed[N] = {"acc": float(x["acc"].mean()), "cost": float(x["cost"].mean()),
                    "acc_q": x["acc"].to_numpy(), "cost_q": x["cost"].to_numpy()}
    res = {"frozen": {k: frozen[k] for k in ("agreement", "quality")}, "fixed_sc": {
        N: {"acc": v["acc"], "cost": v["cost"]} for N, v in fixed.items()}}
    full_stop = np.full(f["lead_frac"].shape[:2], 3)
    sc32_q = evaluate(f, full_stop, ev)["acc_q"]
    sc32_correct = f["sc_correct"][:, :, 3]
    p_correct = None
    rows_tbl = [{"policy": f"fixed SC N={N}", "acc": v["acc"], "cost": v["cost"], "rollouts": N} for N, v in fixed.items()]
    cases = []
    for fam in ("agreement", "quality"):
        s = frozen[fam]
        stop = stop_index(f, s["tau"], s["gamma"], fam)
        r = evaluate(f, stop, ev)
        st = np.array(CHECKS_ALL)[stop][ev]                              # [Qe, K]
        dist = {int(n): float((st == n).mean()) for n in CHECKS_ALL}
        wrong = ~np.take_along_axis(f["sc_correct"], stop[:, :, None], 2)[:, :, 0]
        early = stop < 3
        ie = (early & wrong)[ev]
        rec = (early & wrong & sc32_correct)[ev]
        # nearest-cost fixed budget and linear interpolation of the fixed-N SC curve at the adaptive cost
        costs = np.array([fixed[N]["cost"] for N in cfg["budgets"]])
        accs = np.array([fixed[N]["acc"] for N in cfg["budgets"]])
        Nn = cfg["budgets"][int(np.argmin(np.abs(costs - r["cost"])))]
        interp = float(np.interp(r["cost"], costs, accs))
        res[fam] = {
            "setting": s, "acc": r["acc"], "cost": r["cost"], "rollouts": r["rollouts"], "stop_distribution": dist,
            "incorrect_early_stop_rate": float(ie.mean()), "recoverable_by_fixed32_rate": float(rec.mean()),
            "frac_early_stops": float(early[ev].mean()),
            "acc_of_early_stops": float((~wrong & early)[ev].sum() / max(1, early[ev].sum())),
            "nearest_cost_fixed_N": Nn, "fixed_sc_acc_interpolated_at_same_cost": interp,
            "vs_fixed32": paired(r["acc_q"], sc32_q, idx),
            "vs_nearest_cost_fixed": paired(r["acc_q"], fixed[Nn]["acc_q"], idx),
            "cost_saving_vs_fixed32": 1 - r["cost"] / fixed[32]["cost"],
        }
        rows_tbl.append({"policy": f"{fam} (tau={s['tau']}, gamma={s['gamma']})", "acc": r["acc"], "cost": r["cost"],
                         "rollouts": r["rollouts"]})
        # per-question retrospective table: stopping vs pool correct-fraction and agreement
        if p_correct is None:
            lab = np.load(odir("answers", ds) / "labels.npz")
            p_correct = lab["correct"].mean(1)
        evq = np.where(ev)[0]
        tab = pd.DataFrame({"qid": [pool.qids[q] for q in evq], "p_correct_pool": p_correct[evq],
                            "mean_stop_n": r["n_q"], "acc": r["acc_q"], "acc_sc32": sc32_q,
                            "lead_frac_at_4": f["lead_frac"][evq, :, 0].mean(1),
                            "incorrect_early_frac": ie.mean(1), "recoverable_frac": rec.mean(1)})
        tab.to_csv(d / f"heldout_per_question_{fam}.csv", index=False)
        bins = pd.cut(tab["p_correct_pool"], [-0.001, 0.0, 0.25, 0.5, 0.75, 0.999, 1.0],
                      labels=["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1"])
        tab.groupby(bins, observed=False)[["mean_stop_n", "acc", "acc_sc32", "incorrect_early_frac",
                                           "recoverable_frac"]].agg(["mean"]).assign(
            n=tab.groupby(bins, observed=False).size()).to_csv(d / f"by_pool_correct_fraction_{fam}.csv")
        # example cases: incorrect early stops that fixed-32 would recover (order 0 first)
        for qi_local, q in enumerate(evq):
            for k in range(f["lead_frac"].shape[1]):
                if early[q, k] and wrong[q, k] and sc32_correct[q, k]:
                    cases.append({"family": fam, "qid": pool.qids[q], "order": k,
                                  "stop_n": int(CHECKS_ALL[stop[q, k]]),
                                  "stopped_group": int(f["sc_group"][q, k, stop[q, k]]),
                                  "fixed32_group": int(f["sc_group"][q, k, 3]),
                                  "p_correct_pool": float(p_correct[q])})
    pd.DataFrame(cases).to_csv(d / "recoverable_early_stops.csv", index=False)
    tbl = pd.DataFrame(rows_tbl)
    (d / "heldout.md").write_text(f"## {ds}: adaptive stopping on held-out questions (frozen dev settings)\n\n"
                                  + df_to_md(tbl) + "\n")
    tbl.to_csv(d / "heldout_table.csv", index=False)
    write_json(d / "heldout.json", res)
    return res


def run(cfg: dict) -> dict:
    out = {}
    for ds in ("math", "mathvista"):
        out[ds] = run_dataset(ds, cfg)
        print(f"[adaptive] {ds}: frozen {out[ds]['frozen']}", flush=True)
    return out
