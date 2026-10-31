"""Command-line entry for one pipeline stage:  python -m pa3ttc.stages <stage>

Stages: data, splits, scorer, signals, offline, analysis, adaptive, compute, live_eval, qualitative, report.
Each stage pins itself to the cores in PA3_CPUS, reads earlier stages' files and writes _stages/<stage>.json
when it finishes.
"""
from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np

from .config import D_INDEX, DATASETS, cache_dir, load_config, odir
from .io import pin_cpus, read_json, write_json


def stage_data(cfg):
    from .data.discover import find_model_dirs, find_readme
    from .data.load import load_dataset

    dirs = find_model_dirs(cfg["data_root"])
    rd = find_readme(dirs)
    if rd:
        shutil.copy(rd, odir("data") / "README_supplied.md")
        print(f"[data] README found: {rd}", flush=True)
    out = {"model_dirs": {k: str(v) for k, v in dirs.items()}, "readme": str(rd) if rd else None}
    for ds in DATASETS:
        rep = load_dataset(dirs[ds], ds, cfg)
        out[ds] = {"n_questions": rep["n_questions"], "alignment": rep["alignment"], "columns": rep["columns"]}
        print(f"[data] {ds}: {out[ds]}", flush=True)
    return out


def stage_splits(cfg):
    from .data.load import Pool
    from .data.splits import make_orders, make_split

    out = {}
    for ds in DATASETS:
        pool = Pool(ds)
        if pool.Q != cfg["n_questions_expected"]:
            print(f"!! {ds}: {pool.Q} questions (expected {cfg['n_questions_expected']})", flush=True)
        sp = make_split(pool.qids, cfg["seed"], cfg["n_dev"])
        write_json(odir("splits") / f"{ds}.json", sp)
        orders = make_orders(pool.Q, D_INDEX[ds], cfg["seed"], cfg["n_orders"], pool.R)
        np.save(odir("splits") / f"{ds}_orders.npy", orders)
        out[ds] = {"dev": len(sp["dev"]), "eval": len(sp["eval"]), "first_eval_ids": sp["eval"][:5]}
    pool = Pool("math")
    live_ids = read_json(odir("splits") / "math.json")["eval"][: cfg["n_live"]]
    q = pool.questions.set_index("qid")
    if "problem" not in q.columns or q["problem"].isna().all() or (q["problem"].astype(str) == "None").all():
        raise KeyError("no problem-text column found in the MATH generations; set columns.math.problem")
    qs = [{"live_idx": i, "qid": x, "problem": str(q.loc[x, "problem"]), "gold": q.loc[x, "gold"]}
          for i, x in enumerate(live_ids)]
    write_json(odir("live") / "questions.json", qs)
    write_json(odir("splits") / "live_ids.json", live_ids)
    out["live"] = {"n": len(qs), "first": live_ids[:3]}
    return out


def stage_scorer(cfg):
    from .answers.scorer import run
    return run(cfg)


def stage_signals(cfg):
    from .signals.stage import run
    return run(cfg)


def stage_offline(cfg):
    from .offline.runner import run
    return run(cfg)


def stage_analysis(cfg):
    from .analysis import conf_vision, signal_assoc

    out = {ds: signal_assoc.run_dataset(ds, cfg) for ds in DATASETS}
    out["conf_vision"] = conf_vision.run(cfg)
    return out


def stage_adaptive(cfg):
    from .adaptive.tune import run
    return run(cfg)


def stage_compute(cfg):
    from .analysis.compute_alloc import run
    return run(cfg)


def stage_live_eval(cfg):
    from .live.evaluate import run
    return run(cfg)


def stage_qualitative(cfg):
    from .report.qualitative import run
    return run(cfg)


def stage_report(cfg):
    from .report.figures import run
    return run(cfg)


STAGES = {n[6:]: f for n, f in dict(globals()).items() if n.startswith("stage_")}


def data_cached() -> bool:
    return all((cache_dir(ds) / "tokens.npz").exists() for ds in DATASETS)


def main(name: str) -> int:
    cores = pin_cpus()
    cfg = load_config()
    t0 = time.time()
    print(f"[stage {name}] cores={cores}", flush=True)
    if os.environ.get("PA3_TEST_FAIL_STAGE") == name:      # used only by tests/test_orchestrate.py
        raise RuntimeError(f"injected failure in stage {name}")
    res = STAGES[name](cfg)
    write_json(odir("_stages") / f"{name}.json", {"seconds": round(time.time() - t0, 1), "result": res})
    print(f"[stage {name}] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
