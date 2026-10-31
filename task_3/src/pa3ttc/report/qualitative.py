"""Representative cases with a stated, seeded selection rule (default_rng(6304) over sorted question IDs).

1. Selection disagreement (held-out, N = 32, order 0): self-consistency vs the quality-aware RWC rule
   (self-certainty + RWC for MATH, VisConf + RWC for MathVista); 2 questions where only the quality rule is
   right and 2 where only SC is right.
2. Revision (live): up to 3 wrong->correct and 3 correct->wrong transitions, lowest live index first.
3. Premature stopping (held-out): 2 incorrect early stops per dataset and controller that fixed-32 SC would have
   recovered (order-level cases).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import odir
from ..data.load import Pool
from ..io import read_jsonl, write_json

QRULE = {"math": "selfcert_rwc", "mathvista": "visconf_rwc"}


def _tail(s, n=700) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else "…" + s[-n:]


def _head(s, n=500) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + "…"


def _groups_line(g: pd.DataFrame, qid: str) -> str:
    x = g[g["qid"] == qid].sort_values("size", ascending=False)
    return ", ".join(f"`{a}`×{n}{' ✓' if c else ''}" for a, n, c in zip(x["answer"], x["size"], x["correct"]))


def run(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    md = [f"# Qualitative cases\n\nSelection rule:\n\n{__doc__.strip()}\n"]
    out: dict = {"selection": [], "revision": [], "stopping": []}
    for ds in ("math", "mathvista"):
        pool = Pool(ds)
        texts = pool.texts().set_index(["qid", "rollout_id"])["text"]
        g = pd.read_parquet(odir("answers", ds) / "groups.parquet")
        po = pd.read_parquet(odir("offline", ds) / "per_order.parquet", filters=[("N", "==", 32), ("order", "==", 0)])
        po = po[(po["split"] == "eval") & po["method"].isin(["sc", QRULE[ds]])]
        w = po.pivot_table(index="qid", columns="method", values="correct", observed=True)
        rows = po.set_index(["qid", "method"])
        qtext = pool.questions.set_index("qid")
        md.append(f"\n## {ds}: selection disagreement at N=32 (SC vs {QRULE[ds]})\n")
        for tag, mask in (("quality_right_sc_wrong", (w[QRULE[ds]] == 1) & (w["sc"] == 0)),
                          ("sc_right_quality_wrong", (w[QRULE[ds]] == 0) & (w["sc"] == 1))):
            ids = sorted(w.index[mask].tolist())
            pick = rng.choice(ids, size=min(2, len(ids)), replace=False).tolist() if ids else []
            md.append(f"\n### {tag} ({len(ids)} held-out questions in this bucket; showing {len(pick)})\n")
            for qid in pick:
                case = {"dataset": ds, "bucket": tag, "qid": qid, "gold": qtext.loc[qid, "gold"],
                        "groups": _groups_line(g, qid)}
                md.append(f"\n**{qid}** — gold `{case['gold']}`  \nProblem: {_head(qtext.loc[qid, 'problem'])}  \n"
                          f"Answer groups (order 0): {case['groups']}\n")
                for m in ("sc", QRULE[ds]):
                    r = rows.loc[(qid, m)]
                    case[m] = {"answer": r["answer"], "correct": bool(r["correct"]), "rollout_id": r["rollout_id"]}
                    txt = texts.get((qid, r["rollout_id"]), "")
                    md.append(f"- {m}: `{r['answer']}` ({'correct' if r['correct'] else 'wrong'}), rollout "
                              f"{r['rollout_id']} — tail:\n\n  > {_tail(txt, 400).replace(chr(10), ' ')}\n")
                out["selection"].append(case)
        for fam in ("agreement", "quality"):
            p = odir("adaptive", ds) / "recoverable_early_stops.csv"
            if not p.exists():
                continue
            rc = pd.read_csv(p, dtype={"qid": str})
            rc = rc[rc["family"] == fam]
            if rc.empty:
                continue
            pick = rc.iloc[sorted(rng.choice(len(rc), size=min(2, len(rc)), replace=False).tolist())]
            md.append(f"\n## {ds}: premature stops ({fam} controller) recovered by fixed-32 SC ({len(rc)} order-level cases)\n")
            for _, r in pick.iterrows():
                ga = g[(g["qid"] == r["qid"])].set_index("group")["answer"]
                case = {"dataset": ds, "family": fam, "qid": r["qid"], "order": int(r["order"]),
                        "stop_n": int(r["stop_n"]), "stopped_answer": ga.get(r["stopped_group"]),
                        "fixed32_answer": ga.get(r["fixed32_group"]), "p_correct_pool": r["p_correct_pool"],
                        "gold": qtext.loc[r["qid"], "gold"]}
                out["stopping"].append(case)
                md.append(f"- **{r['qid']}** order {case['order']}: stopped at n={case['stop_n']} with "
                          f"`{case['stopped_answer']}`; fixed-32 SC `{case['fixed32_answer']}` (gold `{case['gold']}`, "
                          f"pool correct fraction {case['p_correct_pool']:.2f}); 32-pool groups: {_groups_line(g, r['qid'])}\n")
    lp = odir("live") / "per_question.csv"
    if lp.exists():
        lq = pd.read_csv(lp, dtype={"qid": str})
        for c in ("initial_correct", "refine_correct"):
            lq[c] = lq[c].astype(str).str.lower().eq("true")
        calls = {(r["qid"], r["slot"]): r for r in read_jsonl(odir("live") / "responses.jsonl")}
        md.append("\n## Live: revision transitions\n")
        for tag, mask in (("wrong_to_correct", ~lq["initial_correct"] & lq["refine_correct"]),
                          ("correct_to_wrong", lq["initial_correct"] & ~lq["refine_correct"])):
            sel = lq[mask].sort_values("live_idx").head(3)
            md.append(f"\n### {tag} ({int(mask.sum())} questions; showing {len(sel)})\n")
            for _, r in sel.iterrows():
                c1, c4 = calls[(r["qid"], 1)], calls[(r["qid"], 4)]
                out["revision"].append({"qid": r["qid"], "tag": tag, "gold": r["gold"],
                                        "initial": r["initial_answer"], "revised": r["refine_answer"]})
                md.append(f"\n**{r['qid']}** gold `{r['gold']}`: initial `{r['initial_answer']}` → revised "
                          f"`{r['refine_answer']}` (initial truncated: {c1['truncated']}, revision truncated: "
                          f"{c4['truncated']})\n\n  > revision tail: {_tail(c4['response'], 500).replace(chr(10), ' ')}\n")
    d = odir("qualitative")
    (d / "cases.md").write_text("\n".join(md))
    write_json(d / "cases.json", out)
    return {k: len(v) for k, v in out.items()}
