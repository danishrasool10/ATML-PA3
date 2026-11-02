"""Scorer stage: answer groups + correctness labels for every stored rollout, frozen with a manifest.

Writes answers/<ds>/labels.npz (group [Q,R], correct [Q,R] group grade, indiv [Q,R]), rollout_groups.parquet,
groups.parquet and answers/scorer_manifest.json (math-verify version, source hashes, counts, extractor check).
Later stages only read these files ("fix the scorer before final evaluation").
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import NO_ANSWER, odir
from ..data.load import Pool
from ..io import write_json
from . import math_equiv, mathvista
from .extract import last_boxed

_TIMEOUT = 5


def _math_q(args):
    answers, gold = args
    return math_equiv.group_and_grade(answers, gold, _TIMEOUT)


def _init(timeout):
    global _TIMEOUT
    _TIMEOUT = timeout


def _src_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def mathvista_meta(pool: Pool, cfg: dict) -> tuple[list[dict], dict]:
    q = pool.questions
    hf, info = None, {"hf_repo": cfg.get("mathvista_metadata_hf")}
    try:
        if not info["hf_repo"]:
            raise RuntimeError("mathvista_metadata_hf not set")
        from pa3common.kaggle import retry  # noqa: PLC0415

        hf, files = retry(lambda: mathvista.hf_testmini(cfg["mathvista_metadata_hf"]), 3, 15, "MathVista metadata")
        info["hf_files"] = files
    except Exception as ex:  # noqa: BLE001
        info["hf_error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
        print(f"!! MathVista metadata download failed ({info['hf_error']}); deriving answer types", flush=True)
    metas, stats = mathvista.build_metadata(q["qid"].tolist(), q["problem"].tolist(), q["gold"].tolist(), hf)
    write_json(odir("answers", "mathvista") / "question_metadata.json",
               {qid: m for qid, m in zip(q["qid"].tolist(), metas)})
    return metas, {**info, **stats}


def score_dataset(ds: str, cfg: dict, workers: int) -> dict:
    pool = Pool(ds)
    Q, R = pool.Q, pool.R
    ext = pool.meta["extracted"].to_numpy().reshape(Q, R)
    golds = pool.questions["gold"].tolist()
    if ds == "math":
        jobs = [(list(ext[q]), golds[q]) for q in range(Q)]
        if workers > 1:
            with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg["mv_timeout_s"],)) as p:
                res = p.map(_math_q, jobs, chunksize=4)
        else:
            _init(cfg["mv_timeout_s"])
            res = [_math_q(j) for j in jobs]
        meta_src = None
    else:
        metas, meta_src = mathvista_meta(pool, cfg)
        res = [mathvista.group_and_grade(list(ext[q]), golds[q], metas[q]) for q in range(Q)]
    group = np.array([r["groups"] for r in res], dtype=np.int32)
    gc = [np.array(r["group_correct"], bool) for r in res]
    correct = np.zeros((Q, R), bool)
    for q in range(Q):
        m = group[q] != NO_ANSWER
        correct[q, m] = gc[q][group[q, m]]
    indiv = np.array([r["indiv_correct"] for r in res], dtype=bool)
    d = odir("answers", ds)
    np.savez(d / "labels.npz", group=group, correct=correct, indiv=indiv)
    rows = pd.DataFrame({"qi": np.repeat(np.arange(Q), R), "qid": np.repeat(pool.qids, R),
                         "rpos": np.tile(np.arange(R), Q), "rollout_id": pool.rollout_ids.ravel(),
                         "extracted": ext.ravel(), "group": group.ravel(), "correct": correct.ravel(),
                         "indiv_correct": indiv.ravel()})
    if ds == "mathvista":
        rows["kind"] = [k for r in res for k in r["rollout_kind"]]
    rows.to_parquet(d / "rollout_groups.parquet", index=False)
    g = [{"qid": pool.qids[q], "group": gi, "answer": a, "kind": k, "correct": bool(c),
          "size": int((group[q] == gi).sum())}
         for q, r in enumerate(res)
         for gi, (a, k, c) in enumerate(zip(r["group_answer"], r["group_kind"], r["group_correct"]))]
    g += [{"qid": pool.qids[q], "group": NO_ANSWER, "answer": "NO_ANSWER", "kind": "na", "correct": False,
           "size": int((group[q] == NO_ANSWER).sum())} for q in range(Q) if (group[q] == NO_ANSWER).any()]
    pd.DataFrame(g).to_parquet(d / "groups.parquet", index=False)
    n_groups = np.array([len(set(row.tolist())) for row in group])
    stats = {"questions": Q, "rollouts": int(Q * R), "no_answer_rollouts": int((group == NO_ANSWER).sum()),
             "rollout_accuracy": float(correct.mean()), "oracle_at_32": float(correct.any(1).mean()),
             "mean_distinct_groups_at_32": float(n_groups.mean()),
             "indiv_vs_group_grade_disagreements": int((indiv != correct).sum()),
             "questions_with_disagreement": int((indiv != correct).any(1).sum())}
    if ds == "math":
        kinds = [k for r in res for k in r["group_kind"]]
        stats["string_groups"] = int(sum(k == "str" for k in kinds))
        stats["gold_unparsed_questions"] = int(sum(not r["gold_parsed"] for r in res))
    else:
        rk = pd.Series([k for r in res for k in r["rollout_kind"]]).value_counts().to_dict()
        stats["rollout_kinds"] = {str(k): int(v) for k, v in rk.items()}
        stats["gold_unmatched_questions"] = int(sum(r["gold_key"] is None for r in res))
        stats["metadata"] = meta_src
    return stats


def extractor_check() -> dict:
    """MATH only: our README extractor (used for the live run) re-applied to the stored generated_text must
    reproduce the supplied extracted_answer exactly.  (MathVista used the export's own dataset_default extractor.)"""
    pool = Pool("math")
    t = pool.texts()
    ours = [last_boxed(x) for x in t["text"].tolist()]
    theirs = [None if (x is None or (isinstance(x, float) and np.isnan(x))) else x
              for x in pool.meta["extracted"].tolist()]
    same = np.array([a == b for a, b in zip(ours, theirs)])
    ex = [{"qid": q, "rollout_id": int(r), "ours": a, "supplied": b}
          for q, r, a, b, s in zip(t["qid"], t["rollout_id"], ours, theirs, same) if not s][:10]
    return {"exact_agreement": float(same.mean()), "n": int(len(same)), "mismatches": int((~same).sum()),
            "examples": ex}


def run(cfg: dict) -> dict:
    workers = max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1)
    man = {"math_verify": _version("math-verify"), "latex2sympy2_extended": _version("latex2sympy2_extended"),
           "scorer_source_sha256_16": _src_hash(), "mv_timeout_s": cfg["mv_timeout_s"],
           "rules": {"math": math_equiv.__doc__.strip(), "mathvista": mathvista.__doc__.strip()}}
    for ds in ("math", "mathvista"):
        man[ds] = score_dataset(ds, cfg, workers)
        if ds == "math":
            man[ds]["extractor_check"] = extractor_check()
        print(f"[scorer] {ds}: {man[ds]}", flush=True)
    write_json(odir("answers") / "scorer_manifest.json", man)
    return man
