"""Data pipeline: normalisation/scoring, VQA filtering, seeded selection + isolation, sampler, schedule,
image-control conditions and the pair bootstrap (no model needed)."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest


def test_normalization_and_soft_score():
    from pa3vlm.data.normalize import normalize_answer as N, soft_score

    assert N("Two") == "2" and N("  The dog.  ") == "dog" and N("Yes!") == "yes" and N("dont") == "don't"
    assert N("10,000") == "10000" and N("3.5") == "3.5" and N(None) == ""
    refs = ["2"] * 3 + ["3"] * 7
    assert soft_score("2", refs) == pytest.approx(0.9)          # 3 matches: official leave-one-out average
    assert soft_score("2", ["2"] * 4 + ["x"] * 6) == pytest.approx(1.0)
    assert soft_score("2", ["2"] * 2 + ["x"] * 8) == pytest.approx(0.6)
    assert soft_score("2", ["2"] + ["x"] * 9) == pytest.approx(0.3)
    assert soft_score("", ["x"] * 10) == 0.0


def test_vqa_consensus_and_pairs():
    from pa3vlm.data.vqa import candidate_pairs, consensus

    assert consensus(["red"] * 5 + ["Red"] * 3 + ["blue"] * 2, 8) == ("red", "red")
    assert consensus(["Red"] * 5 + ["red"] * 3 + ["blue"] * 2, 8) == ("red", "Red")
    assert consensus(["two"] * 4 + ["2"] * 4 + ["3"] * 2, 8) == ("2", "2")   # tie on raw strings -> lexicographic
    assert consensus(["a"] * 7 + ["b"] * 3, 8) is None
    qs = {1: {"image_id": 10, "question": "Is it red?"}, 2: {"image_id": 11, "question": "Is it  red?"},
          3: {"image_id": 12, "question": "Is it red?"}, 4: {"image_id": 13, "question": "Is it blue?"}}
    an = {1: {"answers": ["yes"] * 10, "answer_type": "yes/no"}, 2: {"answers": ["no"] * 9 + ["yes"], "answer_type": "yes/no"},
          3: {"answers": ["yes"] * 10, "answer_type": "yes/no"}, 4: {"answers": ["no"] * 10, "answer_type": "yes/no"}}
    c = candidate_pairs([[1, 2], [1, 3], [1, 4]], qs, an, "val2014", 8)
    assert len(c) == 1 and [e["canonical"] for e in c[0]["examples"]] == ["yes", "no"]   # same answers / question differ
    assert len(c[0]["examples"][1]["answers"]) == 10


def test_selection_sizes_isolation_and_labels(out_root):
    from pa3vlm.data.coco import NO_COUNTING

    D = out_root / "data"
    rep = json.loads((D / "selection_report.json").read_text())
    assert rep["isolation"]["ok"]
    sz = rep["sizes"]
    assert sz["captions_train"] == 48 and sz["captions_dev"] == 6
    assert sz["vqa_train"] == 10 and sz["vqa_dev"] == 5 and sz["vqa_test"] == 8
    assert rep["kinds"]["grounding_test"] == {"presence": 3, "counting": 3}
    assert rep["kinds"]["vqa_test"] == {"yes/no": 3, "number": 2, "other": 3}
    test = json.loads((D / "grounding_test.json").read_text())
    pres = [p for p in test if p["kind"] == "presence"]
    assert {tuple(sorted(e["target"] for e in p["examples"])) for p in pres} == {("no", "yes")}
    pools = json.loads((D / "presence_pool.json").read_text()) + json.loads((D / "counting_pool.json").read_text())
    first = Counter(p["examples"][0]["target"] for p in pools if p["kind"] == "presence")
    assert first["yes"] > 0 and first["no"] > 0, "slot order must be randomised"
    for p in pools + test:
        a, b = p["examples"]
        assert a["image"] != b["image"] and a["target"] != b["target"]
        if p["kind"] == "counting":
            assert p["category"] not in NO_COUNTING and {a["target"], b["target"]} <= {"1", "2", "3", "4"}
            assert p["question"].startswith("How many ")
    vt = json.loads((D / "vqa_train.json").read_text())
    for p in vt:
        for e in p["examples"]:
            assert len(e["answers"]) == 10 and e["target"] in e["answers"]
    ids = json.loads((D / "selected_ids.json").read_text())
    assert len(ids["vqa_test"]["pair_ids"]) == 8 and ids["vqa_test"]["source_question_pairs"][0] is not None


def test_selection_is_deterministic(out_root, tmp_path):
    from pa3vlm.config import load_data_config
    from pa3vlm.data.splits import select_all
    from conftest import SMALL_DATA
    import os
    from pathlib import Path

    sets, _ = select_all(Path(os.environ["PA3_SCRATCH"]) / "raw", load_data_config(None, SMALL_DATA))
    for name in ("vqa_test", "grounding_test", "captions_train", "presence_pool"):
        assert sets[name] == json.loads((out_root / "data" / f"{name}.json").read_text()), name


def test_sampler_stage1_and_mixture():
    from pa3vlm.data.sampler import CurriculumSampler, StageSpec

    s = CurriculumSampler(6304, 48, {"presence": 30, "counting": 20, "vqa": 10}, 8, 2, passes=2)
    assert s.stage1_updates() == 6
    seen = Counter()
    for u in range(6):
        mbs, c = s.update(StageSpec(1, 6, None), u)
        assert [len(m) for m in mbs] == [8, 8] and c == {"caption": 16}
        seen.update(i for m in mbs for _, i in m)
    assert set(seen) == set(range(48)) and set(seen.values()) == {2}
    # each pass is a permutation; passes differ
    p0 = [i for u in range(3) for m in s.update(StageSpec(1, 6, None), u)[0] for _, i in m]
    p1 = [i for u in range(3, 6) for m in s.update(StageSpec(1, 6, None), u)[0] for _, i in m]
    assert sorted(p0) == sorted(p1) == list(range(48)) and p0 != p1
    mix = {"presence": 0.5, "counting": 0.25, "caption": 0.25}
    tot = Counter()
    for u in range(400):
        mbs, c = s.update(StageSpec(2, 400, mix), u)
        tot.update(c)
        for m in mbs:       # pairs complete and adjacent inside a microbatch
            for k in range(0, 8, 2):
                if m[k][0] != "caption":
                    assert m[k][:2] == m[k + 1][:2] and (m[k][2], m[k + 1][2]) == (0, 1)
        assert s.update(StageSpec(2, 400, mix), u) == (mbs, c)      # stateless
    n = sum(tot.values())
    assert n == 400 * 16
    for k, p in mix.items():
        assert abs(tot[k] / n - p) < 0.03, (k, tot[k] / n)


def test_lr_schedule():
    from pa3vlm.train.schedule import lr_at, warmup_updates

    assert [warmup_updates(T, 0.05) for T in (2250, 1500, 750)] == [113, 75, 38]
    assert lr_at(1, 2250, 1e-3) == pytest.approx(1e-3 / 113)
    assert lr_at(113, 2250, 1e-3) == pytest.approx(1e-3)
    assert lr_at(2250, 2250, 1e-3) == pytest.approx(1e-4)
    assert lr_at(750, 750, 1e-5) == pytest.approx(1e-6)
    v = [lr_at(t, 750, 1.0) for t in range(38, 751)]
    assert all(a >= b for a, b in zip(v, v[1:]))


def test_conditions_mapping():
    from pa3vlm.eval.conditions import images_for, mismatch_mapping

    pairs = [{"pair_id": f"p{i}", "examples": [{"ex_id": f"p{i}/0", "image": f"v:{2 * i}"},
                                               {"ex_id": f"p{i}/1", "image": f"v:{2 * i + 1}"}]} for i in range(9)]
    mm = mismatch_mapping(pairs, 6304, 1)
    assert mm == mismatch_mapping(pairs, 6304, 1)
    assert len(set(mm.values())) == 18
    for p in pairs:
        own = {e["image"] for e in p["examples"]}
        assert all(mm[e["ex_id"]] not in own for e in p["examples"])
    sw = images_for(pairs, "swap")
    assert sw[:2] == ["v:1", "v:0"] and images_for(pairs, "none") == [None] * 18
    assert images_for(pairs, "mismatched", mm) == [mm[e["ex_id"]] for p in pairs for e in p["examples"]]


def test_bootstrap_paired():
    from pa3vlm.eval.bootstrap import ci, diff_ci, resample_indices

    rng = np.random.default_rng(0)
    a = rng.random(50)
    b = a - 0.1
    idx = resample_indices(50, 4000, [6304, 1])
    assert (idx == resample_indices(50, 4000, [6304, 1])).all()
    p, lo, hi = diff_ci(a, b, idx)
    assert p == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)  # paired -> no spread
    p, lo, hi = ci(a, idx)
    assert lo < p < hi
