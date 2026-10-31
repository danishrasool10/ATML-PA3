"""Steps 1-2: every selection rule at every candidate budget, for every saved order.

per_order.parquet  one row per (question, order, N, method): selected rollout, answer group/answer, correctness,
                   generated-token cost (sum of T over consumed candidates; FirstAns consumes only its first),
                   oracle coverage and distinct answer groups among the first N candidates.
per_question.parquet  the same averaged over the 10 orders (the unit for all summaries and bootstraps).
summary_<split>.csv/md, decomposition, bootstrap.csv (eval split, paired over questions).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import NO_ANSWER, odir
from ..data.load import Pool
from ..io import df_to_md, read_json, write_json
from ..select.rules import select
from .bootstrap import boot_idx, paired

BASE = [("first", None), ("sc", None), ("logp", "L"), ("logp_rwc", "L"), ("selfcert", "K"), ("selfcert_rwc", "K"),
        ("shortest", None)]
VLM = [("vision", "V"), ("vision_rwc", "V"), ("visconf", "Z"), ("visconf_rwc", "Z"), ("tokfuse", "Ztok"),
       ("tokfuse_rwc", "Ztok")]
EXTRA = {"logp_rwc": "extra ablation (approved 11)", "shortest": "length-only control",
         "tokfuse": "Step-3 ablation", "tokfuse_rwc": "Step-3 ablation"}
LABEL = {"first": "FirstAns", "sc": "Self-consistency", "logp": "Highest mean log-prob", "logp_rwc": "Log-prob + RWC",
         "selfcert": "Highest self-certainty", "selfcert_rwc": "Self-certainty + RWC", "shortest": "Shortest rollout",
         "vision": "Vision-only (highest V)", "vision_rwc": "Vision-only + RWC", "visconf": "VisConf (highest Z)",
         "visconf_rwc": "VisConf + RWC", "tokfuse": "Tokenwise fusion (highest)", "tokfuse_rwc": "Tokenwise fusion + RWC"}


def methods_for(ds: str) -> list[tuple[str, str | None]]:
    return BASE + (VLM if ds == "mathvista" else [])


def load_inputs(ds: str):
    pool = Pool(ds)
    lab = np.load(odir("answers", ds) / "labels.npz")
    sc = dict(np.load(odir("signals", ds) / "rollout_scores.npz"))
    vc = dict(np.load(odir("signals", ds) / "visconf.npz")) if ds == "mathvista" else {}
    orders = np.load(odir("splits") / f"{ds}_orders.npy")
    split = read_json(odir("splits") / f"{ds}.json")
    return pool, lab["group"], lab["correct"], sc, vc, orders, split


def prefix_scores(key, sc, vc, q, k, o, N):
    if key in ("Z", "Ztok"):
        return vc[f"{key}_{N}"][q, k] if N >= 2 else np.zeros(1)
    return sc[key][q, o[:N]]


def run_dataset(ds: str, cfg: dict) -> pd.DataFrame:
    pool, group, correct, sc, vc, orders, split = load_inputs(ds)
    meths = methods_for(ds)
    Q, K, R = orders.shape
    B = list(cfg["budgets"])
    n = Q * K * len(B) * len(meths)
    col = {c: np.zeros(n, t) for c, t in [("qi", np.int32), ("order", np.int8), ("N", np.int8), ("m", np.int8),
                                          ("rpos", np.int16), ("group", np.int32), ("correct", bool),
                                          ("cost", np.int64), ("oracle", bool), ("n_groups", np.int16),
                                          ("n_groups_answered", np.int16)]}
    T = pool.T
    i = 0
    for q in range(Q):
        for k in range(K):
            o = orders[q, k]
            g_o, c_o, T_o = group[q, o], correct[q, o], T[q, o]
            for N in B:
                gN = g_o[:N]
                oracle = bool(c_o[:N].any())
                ug = set(gN.tolist())
                ng, nga = len(ug), len(ug - {NO_ANSWER})
                cost_all = int(T_o[:N].sum())
                for mi, (m, key) in enumerate(meths):
                    s = prefix_scores(key, sc, vc, q, k, o, N) if key else None
                    j = select(m, gN, s, T_o[:N])
                    col["qi"][i], col["order"][i], col["N"][i], col["m"][i] = q, k, N, mi
                    col["rpos"][i], col["group"][i], col["correct"][i] = o[j], gN[j], c_o[j]
                    col["cost"][i] = int(T_o[0]) if m == "first" else cost_all
                    col["oracle"][i], col["n_groups"][i], col["n_groups_answered"][i] = oracle, ng, nga
                    i += 1
    df = pd.DataFrame(col)
    df["method"] = pd.Categorical.from_codes(df.pop("m"), [m for m, _ in meths])
    qids = np.array(pool.qids)
    df.insert(0, "qid", qids[df["qi"].to_numpy()])
    sp = {q: "dev" for q in split["dev"]} | {q: "eval" for q in split["eval"]}
    df.insert(1, "split", df["qid"].map(sp))
    df["rollout_id"] = pool.rollout_ids[df["qi"].to_numpy(), df["rpos"].to_numpy()]
    g = pd.read_parquet(odir("answers", ds) / "groups.parquet")
    amap = dict(zip(zip(g["qid"], g["group"]), g["answer"]))
    df["answer"] = [amap.get((a, b)) for a, b in zip(df["qid"].tolist(), df["group"].tolist())]
    return df


def per_question(df: pd.DataFrame) -> pd.DataFrame:
    pq_ = (df.groupby(["qid", "split", "N", "method"], observed=True)
           .agg(acc=("correct", "mean"), oracle=("oracle", "mean"), cost=("cost", "mean"),
                n_groups=("n_groups", "mean"), n_groups_answered=("n_groups_answered", "mean"))
           .reset_index())
    pq_["missed"] = pq_["oracle"] - pq_["acc"]          # correct candidate available but not selected
    pq_["no_correct"] = 1.0 - pq_["oracle"]             # sampling failure
    return pq_


def summarize(pq_: pd.DataFrame, split: str) -> pd.DataFrame:
    s = (pq_[pq_["split"] == split].groupby(["N", "method"], observed=True)
         [["acc", "oracle", "cost", "n_groups", "n_groups_answered", "missed", "no_correct"]].mean().reset_index())
    s["label"] = s["method"].map(LABEL)
    s["note"] = s["method"].map(EXTRA).fillna("")
    return s


def comparisons(ds: str) -> list[tuple[str, str]]:
    c = [("sc", "first")]
    scores = ["logp", "selfcert"] + (["vision", "visconf", "tokfuse"] if ds == "mathvista" else [])
    for s in scores:
        c += [(s, "sc"), (s + "_rwc", "sc"), (s + "_rwc", s)]
    c += [("shortest", "sc")]
    if ds == "mathvista":
        c += [("visconf", "vision"), ("visconf_rwc", "vision_rwc"), ("visconf", "selfcert"),
              ("visconf_rwc", "selfcert_rwc"), ("tokfuse", "visconf"), ("tokfuse_rwc", "visconf_rwc")]
    return c


def bootstrap_table(pq_: pd.DataFrame, ds: str, cfg: dict, split: str = "eval") -> pd.DataFrame:
    e = pq_[pq_["split"] == split]
    piv = e.pivot_table(index="qid", columns=["N", "method"], values="acc", observed=True)
    idx = boot_idx(len(piv), cfg["n_boot"], cfg["seed"])
    rows = []
    for N in [n for n in cfg["budgets"] if n >= 2]:
        for a, b in comparisons(ds):
            if (N, a) in piv and (N, b) in piv:
                rows.append({"dataset": ds, "split": split, "N": N, "A": a, "B": b,
                             **paired(piv[(N, a)].to_numpy(), piv[(N, b)].to_numpy(), idx)})
    return pd.DataFrame(rows)


def compact_table(s: pd.DataFrame, value: str = "acc") -> pd.DataFrame:
    t = s.pivot_table(index="method", columns="N", values=value, observed=True)
    order = [m for m in LABEL if m in t.index]
    t = t.loc[order]
    t.index = [LABEL[m] + (" *" if m in EXTRA else "") for m in t.index]
    t.columns = [f"N={c}" for c in t.columns]
    return t


def run(cfg: dict) -> dict:
    info = {}
    for ds in ("math", "mathvista"):
        t0 = time.time()
        df = run_dataset(ds, cfg)
        d = odir("offline", ds)
        df.to_parquet(d / "per_order.parquet", index=False)
        pq_ = per_question(df)
        pq_.to_parquet(d / "per_question.parquet", index=False)
        for split in ("eval", "dev"):
            s = summarize(pq_, split)
            s.to_csv(d / f"summary_{split}.csv", index=False)
            (d / f"summary_{split}.md").write_text(
                f"## {ds} — {split} split: accuracy by method and budget (* = extra/ablation)\n\n"
                + df_to_md(compact_table(s), index=True)
                + "\n### oracle coverage / distinct answer groups / mean token cost (all-candidate methods)\n\n"
                + df_to_md(s[s["method"] == "sc"][["N", "oracle", "n_groups", "n_groups_answered", "cost",
                                                     "missed", "no_correct"]].reset_index(drop=True)))
        bt = bootstrap_table(pq_, ds, cfg)
        bt.to_csv(d / "bootstrap.csv", index=False)
        orders = np.load(odir("splits") / f"{ds}_orders.npy")
        pool = Pool(ds)
        pd.DataFrame([{"qid": pool.qids[q], "order": k, "candidate_rollout_ids": pool.rollout_ids[q, orders[q, k]].tolist()}
                      for q in range(orders.shape[0]) for k in range(orders.shape[1])]).to_parquet(
            odir("splits") / f"{ds}_candidate_orders.parquet", index=False)
        info[ds] = {"rows": len(df), "seconds": round(time.time() - t0, 1)}
        print(f"[offline] {ds}: {info[ds]}", flush=True)
    write_json(odir("offline") / "info.json", info)
    return info
