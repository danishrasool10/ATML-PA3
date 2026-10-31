"""End-to-end: every stage on synthetic exports (in-process), live generation with the fake backend, and checks of
the accounting rules (costs, N=1, oracle, consistency between offline SC and the adaptive fallback, live tokens)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pa3ttc import stages
from pa3ttc.data.load import Pool
from pa3ttc.io import read_json, read_jsonl
from pa3ttc.live import run as live_run


def _run_all(skip_live=False):
    for s in ("data", "splits", "scorer", "signals", "offline", "analysis", "adaptive", "compute"):
        assert stages.main(s) == 0
    if not skip_live:
        assert live_run.main() == 0
        assert stages.main("live_eval") == 0
    for s in ("qualitative", "report"):
        assert stages.main(s) == 0


def test_end_to_end(env):
    _run_all()
    out = env["out"]
    # ---- data: join despite shuffled rows, alignment, zero-length rollout, undefined v
    ali = read_json(out / "data" / "math_alignment.json")
    assert ali["alignment"]["misaligned_measurements"] == 0 and ali["alignment"]["zero_length_rollouts"] == 1
    assert ali["alignment"]["token_ids_len_minus_T"] == {"1": 16 * 32}
    mv = read_json(out / "data" / "mathvista_alignment.json")
    assert mv["columns"]["attention_metrics"] == {"img": "image_attention", "prm": "prompt_text_attention",
                                                  "gen": "generated_text_attention"}
    assert mv["alignment"]["attention_tokens_undefined_v"] == 1
    pool = Pool("math")
    assert pool.Q == 16 and pool.R == 32 and pool.rollout_ids[0].tolist() == [str(i) for i in range(32)]
    # ---- scorer
    man = read_json(out / "answers" / "scorer_manifest.json")
    assert man["math_verify"] and man["math"]["extractor_check"]["agreement"] == 1.0
    lab = np.load(out / "answers" / "math" / "labels.npz")
    q2 = pool.qids.index("test/algebra/2.json")                                # string order: index != 2
    assert pool.qids[:3] == ["test/algebra/0.json", "test/algebra/1.json", "test/algebra/10.json"]
    assert (lab["group"][q2, :3] == -1).all() and not lab["correct"][q2, :3].any()
    assert man["mathvista"]["metadata_source"] == "supplied files"
    # ---- offline
    po = pd.read_parquet(out / "offline" / "math" / "per_order.parquet")
    pqo = pd.read_parquet(out / "offline" / "math" / "per_question.parquet")
    orders = np.load(out / "splits" / "math_orders.npy")
    T = pool.T
    n1 = po[po["N"] == 1]
    assert n1.groupby(["qi", "order"])["rpos"].nunique().max() == 1          # N=1: all rules agree
    f = po[po["method"] == "first"]
    assert (f["rpos"].to_numpy() == orders[f["qi"], f["order"], 0]).all()
    assert (f["cost"].to_numpy() == T[f["qi"], orders[f["qi"], f["order"], 0]]).all()
    s8 = po[(po["method"] == "sc") & (po["N"] == 8)]
    exp = [T[q, orders[q, k, :8]].sum() for q, k in zip(s8["qi"], s8["order"])]
    assert (s8["cost"].to_numpy() == np.array(exp)).all()
    orc = po[po["method"] == "sc"].pivot_table(index=["qi", "order"], columns="N", values="oracle")
    assert (np.diff(orc.to_numpy().astype(int), axis=1) >= 0).all()             # nested prefixes
    assert ((pqo["acc"] <= pqo["oracle"] + 1e-12)).all()
    assert np.allclose(pqo["missed"] + pqo["no_correct"] + pqo["acc"], 1.0)
    assert set(po["method"].unique()) == {"first", "sc", "logp", "logp_rwc", "selfcert", "selfcert_rwc", "shortest"}
    pv = pd.read_parquet(out / "offline" / "mathvista" / "per_order.parquet")
    assert {"visconf", "visconf_rwc", "vision", "vision_rwc", "tokfuse", "tokfuse_rwc"} <= set(pv["method"].unique())
    bt = pd.read_csv(out / "offline" / "mathvista" / "bootstrap.csv")
    assert {"lo", "hi"} <= set(bt.columns) and (bt["lo"] <= bt["diff"] + 1e-12).all()
    # ---- adaptive: fallback == offline SC at 32; frozen settings meet the dev constraint
    for ds in ("math", "mathvista"):
        fz = read_json(out / "adaptive" / ds / "frozen.json")
        grid = pd.read_csv(out / "adaptive" / ds / "dev_grid.csv")
        for fam in ("agreement", "quality"):
            assert fz[fam]["acc"] >= fz["target_acc"] - 1e-12
            elig = grid[(grid["family"] == fam) & (grid["acc"] >= fz["target_acc"] - 1e-12)]
            assert fz[fam]["cost"] == pytest.approx(elig["cost"].min())
        s = pd.read_csv(out / "offline" / ds / "summary_dev.csv")
        sc32 = s[(s["method"] == "sc") & (s["N"] == 32)]["acc"].item()
        assert fz["sc32_dev_acc"] == pytest.approx(sc32)
        h = read_json(out / "adaptive" / ds / "heldout.json")
        assert sum(h["agreement"]["stop_distribution"].values()) == pytest.approx(1.0)
    # ---- analysis
    cv = read_json(out / "analysis" / "mathvista" / "conf_vision_summary.json")
    assert cv["identity_check"]["max_abs_error"] < 1e-9
    assert (out / "analysis" / "math" / "associations.csv").exists()
    # ---- live accounting
    qs = read_json(out / "live" / "questions.json")
    sp = read_json(out / "splits" / "math.json")
    assert [q["qid"] for q in qs] == sp["eval"][:3]
    rows = read_jsonl(out / "live" / "responses.jsonl")
    assert len(rows) == 15 and [r["slot"] for r in rows[:5]] == [1, 2, 3, 4, 0]
    assert len({r["seed"] for r in rows}) == 15
    by = {(r["qid"], r["slot"]): r for r in rows}
    for q in qs:
        r4 = by[(q["qid"], 4)]
        assert r4["messages"][2] == {"role": "assistant", "content": by[(q["qid"], 1)]["response"]}
        assert r4["messages"][3]["content"].startswith("Review the solution above.")
        assert r4["cap"] == 2048 and by[(q["qid"], 0)]["cap"] == 3072 and by[(q["qid"], 2)]["cap"] == 1024
        assert by[(q["qid"], 1)]["messages"][1]["content"].endswith(
            "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}.")
    lq = pd.read_csv(out / "live" / "per_question.csv", dtype={"qid": str}).set_index("qid")
    for q in qs:
        x = lq.loc[q["qid"]]
        assert x["refine_input_tokens"] == by[(q["qid"], 1)]["input_tokens"] + by[(q["qid"], 4)]["input_tokens"]
        assert x["refine_output_tokens"] == by[(q["qid"], 1)]["output_tokens"] + by[(q["qid"], 4)]["output_tokens"]
        assert x["parallel_output_tokens"] == sum(by[(q["qid"], s)]["output_tokens"] for s in (1, 2, 3))
        assert x["single_seconds"] == pytest.approx(by[(q["qid"], 0)]["seconds"])
        if by[(q["qid"], 4)]["extracted"] is None:
            assert not x["refine_correct"] and x["refine_fallback_answer"] == x["initial_answer"]
    ls = read_json(out / "live" / "summary.json")
    t = ls["transitions_vs_initial"]
    assert t["wrong_to_correct"] + t["correct_to_wrong"] + t["correct_to_correct"] + t["wrong_to_wrong"] == 3
    # ---- report
    figs = {p.name for p in (out / "report_assets" / "figures").glob("*.png")}
    assert {"math_accuracy_curves.png", "mathvista_accuracy_curves.png", "live_strategies.png",
            "adaptive_stopping_distribution.png", "mathvista_confidence_vision.png"} <= figs
    assert read_json(out / "_stages" / "report.json")["result"]["failed"] == {}
    assert (out / "qualitative" / "cases.md").exists()


def test_live_resume_and_pause(env, monkeypatch):
    for s in ("data", "splits"):
        stages.main(s)
    out = env["out"]
    monkeypatch.setenv("PA3_DEADLINE_UNIX", "0")                 # budget already used -> pause before any call
    assert live_run.main() == 3
    assert read_json(out / "live" / "status.json")["state"] == "paused"
    monkeypatch.delenv("PA3_DEADLINE_UNIX")
    assert live_run.main() == 0
    rows = read_jsonl(out / "live" / "responses.jsonl")
    # simulate a crash that lost the last two calls, then resume: only those are regenerated, identically
    lines = (out / "live" / "responses.jsonl").read_text().splitlines()
    (out / "live" / "responses.jsonl").write_text("\n".join(lines[:-2]) + "\n")
    assert live_run.main() == 0
    rows2 = read_jsonl(out / "live" / "responses.jsonl")
    assert len(rows2) == len(rows) == 15
    key = lambda r: (r["qid"], r["slot"], r["seed"], r["response"])  # noqa: E731
    assert sorted(map(key, rows)) == sorted(map(key, rows2))


def test_long_format_probability_table(tmp_path, monkeypatch):
    from conftest import make_data

    a = make_data(tmp_path / "a")
    b = make_data(tmp_path / "b", long_format=True)
    res = {}
    for name, root in (("a", a), ("b", b)):
        monkeypatch.setenv("PA3_OUTPUT_ROOT", str(tmp_path / f"out_{name}"))
        monkeypatch.setenv("PA3_SCRATCH", str(tmp_path / f"scr_{name}"))
        monkeypatch.setenv("PA3_CONFIG_OVERRIDES", json.dumps({"data_root": str(root), "n_questions_expected": 16}))
        stages.main("data")
        p = Pool("math")
        res[name] = (p.T.copy(), p.kl.copy(), p.logp.copy())
    for x, y in zip(res["a"], res["b"]):
        np.testing.assert_array_equal(x, y)
