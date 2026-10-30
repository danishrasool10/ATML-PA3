"""Seeded data selection (seed 6304) -> manifests under $PA3_OUTPUT_ROOT/<data_dir>.

    python -m pa3vlm.data.splits [--data-config configs/data.yaml] [--no-images]

Order: evaluation sets first from val2014 (VQA test, VQA dev, grounding test, grounding dev, caption dev),
each image used at most once across all of them; then training sets from train2014 (captions, VQA pairs,
grounding pools built on caption + VQA-train images).  Disjointness is asserted and recorded.

Unified pair record:
  {"pair_id", "suite": "grounding"|"vqa", "kind", "question", "category"?, "examples": [
      {"ex_id", "image": "val2014:123", "image_id", "target", "canonical", "answers": [10]|None, "qid"?}, x2]}
Caption record: {"ex_id", "image", "image_id", "ann_id", "caption"}
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from pa3common.utils import output_root, write_json

from ..config import load_data_config
from .coco import index_captions, index_instances
from .download import ensure_annotation, ensure_images
from .grounding import eval_pairs, train_pools
from .vqa import candidate_pairs, load_vqa


def _load(name: str, raw: Path) -> dict:
    p = ensure_annotation(name, raw)
    print(f"[data] loading {p}", flush=True)
    return json.loads(p.read_text())


def _dedupe_pairs(pairs: list) -> list:
    seen, out = set(), []
    for p in pairs:
        k = frozenset(p)
        if len(k) == 2 and k not in seen:
            seen.add(k)
            out.append(list(p))
    return out


def _take_vqa(cands: list[dict], quotas: dict, rng: np.random.Generator, used: set[int] | None) -> list[dict]:
    """Per answer-type quota; if `used` is given every image is used at most once (eval sets)."""
    out = []
    for kind in sorted(quotas):
        pool = [c for c in cands if c["kind"] == kind]
        order = rng.permutation(len(pool))
        got = 0
        for j in order:
            c = pool[int(j)]
            ids = [e["image_id"] for e in c["examples"]]
            if used is not None and any(i in used for i in ids):
                continue
            if used is not None:
                used.update(ids)
            out.append(c)
            got += 1
            if got == quotas[kind]:
                break
        assert got == quotas[kind], f"only {got}/{quotas[kind]} VQA pairs of type {kind}"
    return out


def _finalize(pairs: list[dict], name: str, suite: str) -> list[dict]:
    for j, p in enumerate(pairs):
        p["pair_id"] = f"{name}-{j:04d}"
        p["suite"] = suite
        for s, e in enumerate(p["examples"]):
            e["ex_id"] = f"{p['pair_id']}/{s}"
    return pairs


def select_all(raw: Path, cfg: dict) -> tuple[dict, dict]:
    seed = cfg["seed"]
    rep: dict = {"seed": seed, "config": cfg}
    # ---------------------------------------------------------------- val2014: evaluation sets
    caps_val = index_captions(_load("captions_val2014.json", raw))
    _, stats_val = index_instances(_load("instances_val2014.json", raw))
    qv, av = load_vqa(_load("v2_OpenEnded_mscoco_val2014_questions.json", raw),
                      _load("v2_mscoco_val2014_annotations.json", raw))
    pv = _dedupe_pairs(_load("v2_mscoco_val2014_complementary_pairs.json", raw))
    cv = candidate_pairs(pv, qv, av, "val2014", cfg["vqa_min_agree"])
    rep["vqa_val_candidates"] = dict(Counter(c["kind"] for c in cv), pairs_in_file=len(pv))
    del qv, av
    used: set[int] = set()
    sets: dict[str, list] = {}
    sets["vqa_test"] = _finalize(_take_vqa(cv, cfg["vqa_test"], np.random.default_rng([seed, 1]), used), "vqa_test", "vqa")
    sets["vqa_dev"] = _finalize(_take_vqa(cv, cfg["vqa_dev"], np.random.default_rng([seed, 2]), used), "vqa_dev", "vqa")
    val_ids = sorted(stats_val)
    g = cfg["grounding_test"]
    sets["grounding_test"] = _finalize(eval_pairs(val_ids, stats_val, caps_val, cfg, g["presence"], g["counting"],
                                                  used, seed, tag=1), "grounding_test", "grounding")
    g = cfg["grounding_dev"]
    sets["grounding_dev"] = _finalize(eval_pairs(val_ids, stats_val, caps_val, cfg, g["presence"], g["counting"],
                                                 used, seed, tag=2), "grounding_dev", "grounding")
    rng = np.random.default_rng([seed, 3])
    free = [i for i in val_ids if i not in used and len(caps_val.get(i, [])) >= 1]
    dev_imgs = sorted(int(i) for i in rng.choice(free, cfg["caption_dev_images"], replace=False))
    used.update(dev_imgs)
    sets["captions_dev"] = []
    for j, i in enumerate(dev_imgs):
        c = caps_val[i][int(rng.integers(len(caps_val[i])))]
        sets["captions_dev"].append({"ex_id": f"cap_dev-{j:03d}", "image": f"val2014:{i}", "image_id": i,
                                     "ann_id": c["id"], "caption": " ".join(c["caption"].split())})
    del caps_val, stats_val
    # ---------------------------------------------------------------- train2014: training sets
    caps_tr = index_captions(_load("captions_train2014.json", raw))
    rng = np.random.default_rng([seed, 4])
    elig = sorted(i for i, v in caps_tr.items() if len(v) >= cfg["captions_per_train_image"])
    tr_imgs = sorted(int(i) for i in rng.choice(elig, cfg["caption_train_images"], replace=False))
    sets["captions_train"] = []
    for i in tr_imgs:
        for k in sorted(rng.choice(len(caps_tr[i]), cfg["captions_per_train_image"], replace=False)):
            c = caps_tr[i][int(k)]
            sets["captions_train"].append({"ex_id": f"cap_train-{len(sets['captions_train']):05d}",
                                           "image": f"train2014:{i}", "image_id": i, "ann_id": c["id"],
                                           "caption": " ".join(c["caption"].split())})
    qt, at = load_vqa(_load("v2_OpenEnded_mscoco_train2014_questions.json", raw),
                      _load("v2_mscoco_train2014_annotations.json", raw))
    pt = _dedupe_pairs(_load("v2_mscoco_train2014_complementary_pairs.json", raw))
    ct = candidate_pairs(pt, qt, at, "train2014", cfg["vqa_min_agree"])
    rep["vqa_train_candidates"] = dict(Counter(c["kind"] for c in ct), pairs_in_file=len(pt))
    del qt, at
    sets["vqa_train"] = _finalize(_take_vqa(ct, cfg["vqa_train"], np.random.default_rng([seed, 5]), None),
                                  "vqa_train", "vqa")
    _, stats_tr = index_instances(_load("instances_train2014.json", raw))
    universe = sorted(set(tr_imgs) | {e["image_id"] for p in sets["vqa_train"] for e in p["examples"]})
    pres, cnt, prep = train_pools(universe, stats_tr, caps_tr, cfg, seed)
    sets["presence_pool"] = _finalize(pres, "presence_pool", "grounding")
    sets["counting_pool"] = _finalize(cnt, "counting_pool", "grounding")
    rep["grounding_pools"] = prep
    rep["grounding_universe_images"] = len(universe)
    return sets, rep


def image_keys(sets: dict) -> dict[str, list[str]]:
    keys: dict[str, set[str]] = {"train": set(), "val": set()}
    for name, rows in sets.items():
        for r in rows:
            exs = r["examples"] if "examples" in r else [r]
            for e in exs:
                keys["train" if e["image"].startswith("train2014") else "val"].add(e["image"])
    return {k: sorted(v, key=lambda s: (s.split(":")[0], int(s.split(":")[1]))) for k, v in keys.items()}


def check_isolation(sets: dict) -> dict:
    """Eval sets: val2014 only, mutually image-disjoint, each image once; training: train2014 only."""
    def imgs(name):
        rows = sets[name]
        return [e["image"] for r in rows for e in (r["examples"] if "examples" in r else [r])]

    ev = ["vqa_test", "vqa_dev", "grounding_test", "grounding_dev", "captions_dev"]
    tr = ["captions_train", "vqa_train", "presence_pool", "counting_pool"]
    out = {}
    allev: list[str] = []
    for n in ev:
        im = imgs(n)
        assert all(i.startswith("val2014:") for i in im), n
        assert len(im) == len(set(im)), f"{n}: an image is used twice"
        allev += im
        out[n] = len(im)
    assert len(allev) == len(set(allev)), "evaluation sets share images"
    trimgs = {i for n in tr for i in imgs(n)}
    assert all(i.startswith("train2014:") for i in trimgs)
    assert not (trimgs & set(allev))
    for n in ev + ["vqa_train", "presence_pool", "counting_pool"]:
        for r in sets[n]:
            if "examples" in r:
                a, b = r["examples"]
                assert a["image"] != b["image"] and a["canonical"] != b["canonical"], r["pair_id"]
    out.update(eval_images_total=len(allev), train_images_total=len(trimgs), ok=True)
    return out


def write_manifests(sets: dict, rep: dict, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in sets.items():
        write_json(data_dir / f"{name}.json", rows)
    write_json(data_dir / "images.json", image_keys(sets))
    ids = {}
    for name, rows in sets.items():
        if rows and "examples" in rows[0]:
            ids[name] = {"pair_ids": [r["pair_id"] for r in rows],
                         "source_question_pairs": [r.get("source_pair") for r in rows],
                         "images": [[e["image"] for e in r["examples"]] for r in rows]}
        else:
            ids[name] = {"ex_ids": [r["ex_id"] for r in rows], "ann_ids": [r["ann_id"] for r in rows],
                         "images": [r["image"] for r in rows]}
    write_json(data_dir / "selected_ids.json", ids)
    rep["isolation"] = check_isolation(sets)
    rep["sizes"] = {n: len(r) for n, r in sets.items()}
    rep["kinds"] = {n: dict(Counter(p["kind"] for p in r)) for n, r in sets.items() if r and "kind" in r[0]}
    write_json(data_dir / "selection_report.json", rep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()
    cfg = load_data_config(a.data_config, a.set)
    scratch = Path(os.environ.get("PA3_SCRATCH", "/tmp/pa3_scratch"))
    data_dir = output_root() / "data"
    t0 = time.time()
    if not (data_dir / "selection_report.json").exists():
        sets, rep = select_all(scratch / "raw", cfg)
        rep["selection_seconds"] = time.time() - t0
        write_manifests(sets, rep, data_dir)
        print(f"[data] manifests written ({time.time() - t0:.0f}s): {rep['sizes']}", flush=True)
    if not a.no_images:
        keys = json.loads((data_dir / "images.json").read_text())
        st = ensure_images(keys["train"] + keys["val"], scratch / "images")
        write_json(data_dir / "image_fetch.json", st)
        print(f"[data] images ready: {st}", flush=True)


if __name__ == "__main__":
    main()
