"""Annotation-derived grounding questions (presence / counting) from COCO instances.

Presence pair: the same question on an image with the object (target "yes") and one without ("no").
Counting pair: the same question on two images with different unambiguous counts in 1..4 (digit targets).
Slot order inside every pair is randomised (seeded), so slot 0 is not always "yes" / the smaller count.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import cycle

import numpy as np

from .coco import CATEGORIES, count_ok, counting_question, presence_negative_ok, presence_positive_ok, \
    presence_question

COUNT_COMBOS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]


def _pair(kind: str, cat: str, split: str, ex: list[tuple[int, str]], rng: np.random.Generator) -> dict:
    if rng.random() < 0.5:
        ex = ex[::-1]
    q = presence_question(cat) if kind == "presence" else counting_question(cat)
    return {"kind": kind, "category": cat, "question": q,
            "examples": [{"image": f"{split}:{i}", "image_id": i, "target": t, "canonical": t, "answers": None}
                         for i, t in ex]}


def _draw_negative(cat: str, universe: list[int], stats: dict, caps: dict, rng: np.random.Generator,
                   excluded: set[int], tries: int = 2000) -> int | None:
    for _ in range(tries):
        i = universe[int(rng.integers(len(universe)))]
        if i in excluded:
            continue
        if presence_negative_ok(stats[i], cat, [c["caption"] for c in caps.get(i, [])]):
            return i
    return None


def train_pools(universe: list[int], stats: dict, caps: dict, cfg: dict, seed: int, split: str = "train2014"
                ) -> tuple[list[dict], list[dict], dict]:
    universe = sorted(universe)
    cap = cfg["pool_cap_per_category"]
    pres, cnt, report = [], [], {"presence_per_cat": {}, "counting_per_cat": {}}
    for ci, cat in enumerate(sorted(CATEGORIES)):
        rng = np.random.default_rng([seed, 11, ci])
        pos = [i for i in universe if presence_positive_ok(stats[i], cat, cfg["min_area"])]
        rng.shuffle(pos)
        n = 0
        for p in pos[:cap]:
            neg = _draw_negative(cat, universe, stats, caps, rng, {p})
            if neg is not None:
                pres.append(_pair("presence", cat, split, [(p, "yes"), (neg, "no")], rng))
                n += 1
        report["presence_per_cat"][cat] = n
        by = defaultdict(list)
        for i in universe:
            k = count_ok(stats[i], cat, cfg["min_area"], cfg["max_count"])
            if k is not None:
                by[k].append(i)
        ks = sorted(by)
        n = 0
        if len(ks) >= 2:
            imgs = [i for k in ks for i in by[k]]
            rng.shuffle(imgs)
            kof = {i: k for k in ks for i in by[k]}
            for a in imgs:
                if n >= cap:
                    break
                others = [k for k in ks if k != kof[a]]
                kb = others[int(rng.integers(len(others)))]
                b = by[kb][int(rng.integers(len(by[kb])))]
                cnt.append(_pair("counting", cat, split, [(a, str(kof[a])), (b, str(kb))], rng))
                n += 1
        report["counting_per_cat"][cat] = n
    report.update(n_presence=len(pres), n_counting=len(cnt))
    return pres, cnt, report


def eval_pairs(candidates: list[int], stats: dict, caps: dict, cfg: dict, n_presence: int, n_counting: int,
               used: set[int], seed: int, tag: int, split: str = "val2014") -> list[dict]:
    """Image-disjoint eval pairs (every image at most once; `used` is updated in place)."""
    rng = np.random.default_rng([seed, 21, tag])
    cands = sorted(candidates)
    cats = sorted(CATEGORIES)
    rng.shuffle(cats)
    out: list[dict] = []
    # presence: categories round-robin
    pos_lists = {}
    for cat in cats:
        lst = [i for i in cands if presence_positive_ok(stats[i], cat, cfg["min_area"])]
        rng.shuffle(lst)
        pos_lists[cat] = lst
    it = cycle(cats)
    guard = 0
    while sum(p["kind"] == "presence" for p in out) < n_presence:
        guard += 1
        assert guard < 100 * (n_presence + 1), "not enough presence candidates"
        cat = next(it)
        p = next((i for i in pos_lists[cat] if i not in used), None)
        if p is None:
            continue
        neg = _draw_negative(cat, cands, stats, caps, rng, used | {p})
        if neg is None:
            continue
        used.update((p, neg))
        out.append(_pair("presence", cat, split, [(p, "yes"), (neg, "no")], rng))
    # counting: count combinations round-robin, categories rotated for diversity
    by = {cat: defaultdict(list) for cat in cats}
    for i in cands:
        for cat in cats:
            k = count_ok(stats[i], cat, cfg["min_area"], cfg["max_count"])
            if k is not None:
                by[cat][k].append(i)
    for cat in cats:
        for k in by[cat]:
            rng.shuffle(by[cat][k])
    cat_order = list(cats)
    ci = 0
    combos = cycle(COUNT_COMBOS)
    guard = 0
    while sum(p["kind"] == "counting" for p in out) < n_counting:
        guard += 1
        assert guard < 100 * (n_counting + 1), "not enough counting candidates"
        ka, kb = next(combos)
        for j in range(len(cat_order)):
            cat = cat_order[(ci + j) % len(cat_order)]
            a = next((i for i in by[cat].get(ka, []) if i not in used), None)
            b = next((i for i in by[cat].get(kb, []) if i not in used and i != a), None)
            if a is not None and b is not None:
                used.update((a, b))
                out.append(_pair("counting", cat, split, [(a, str(ka)), (b, str(kb))], rng))
                ci = (ci + j + 1) % len(cat_order)
                break
    return out
