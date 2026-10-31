"""Unit tests: extraction, answer equivalence, splits/orders, selection/tie rules, VisConf (fast == naive,
zero leakage), tokenwise identity, adaptive controller logic."""
from __future__ import annotations

import numpy as np
import pytest

from pa3ttc.adaptive.controllers import evaluate, features, stop_index
from pa3ttc.adaptive.tune import select_setting
from pa3ttc.answers import mathvista as mvn
from pa3ttc.answers.extract import last_boxed
from pa3ttc.answers.math_equiv import NO_ANSWER, group_and_grade
from pa3ttc.data.splits import make_orders, make_split
from pa3ttc.select.rules import avg_rank_desc, rwc, rwc_weights, select, vote, vote_margin
from pa3ttc.signals.visconf import VisConfIndex, naive_scores


# ----------------------------------------------------------------------------- extraction
@pytest.mark.parametrize("text,ans", [
    ("so $\\boxed{\\frac{1}{2}}$.", "\\frac{1}{2}"),
    ("\\boxed{1} then \\boxed{\\sqrt{x^{2}}}", "\\sqrt{x^{2}}"),
    ("\\boxed{a} and finally \\boxed{b", None),            # last box cut off by the cap
    ("answer \\boxed 5 done", "5"),
    ("\\fbox{7}", "7"),
    ("no box here", None),
    ("\\boxed{}", None),
    ("<think>\\boxed{2}</think> \\boxed{ {3} }", "{3}"),
])
def test_last_boxed(text, ans):
    assert last_boxed(text) == ans


# ----------------------------------------------------------------------------- math-verify grouping
def test_math_grouping_and_grading():
    r = group_and_grade(["0.5", "\\frac{1}{2}", None, "1/2", "3", "", "\\dfrac12", "Monday?"], "\\frac{1}{2}")
    g = r["groups"]
    assert g[2] == NO_ANSWER and g[5] == NO_ANSWER
    assert g[0] == g[1] == g[3] == g[6] != g[4]
    assert r["group_correct"][g[0]] and not r["group_correct"][g[4]]
    assert r["indiv_correct"] == [True, True, False, True, False, False, True, False]
    # groups are numbered by first appearance in rollout order
    assert g[0] == 0 and g[4] == 1


def test_math_equation_and_text_gold():
    r = group_and_grade(["x=3", "3", "4"], "3")
    assert r["groups"][0] == r["groups"][1] and r["group_correct"][r["groups"][0]]
    r = group_and_grade(["Monday", "\\text{Monday}", "Tuesday"], "\\text{Monday}")
    assert r["indiv_correct"][:2] == [True, True] and not r["indiv_correct"][2]


# ----------------------------------------------------------------------------- MathVista normalisation
def test_mathvista_rules():
    mc = {"question_type": "multi_choice", "answer_type": "text", "choices": '["red", "blue", "green"]'}
    assert mvn.normalize("B", mc)[0] == "MC:B"
    assert mvn.normalize("(b) blue", mc)[0] == "MC:B"
    assert mvn.normalize("blue", mc) == ("MC:B", "mc_exact")
    assert mvn.normalize("gren", mc) == ("MC:C", "mc_fuzzy")
    assert mvn.normalize("D", mc)[0] == "MC:B" or mvn.normalize("D", mc)[1] == "mc_fuzzy"   # out of range -> fuzzy
    assert mvn.normalize(None, mc) == (None, "na") and mvn.normalize("  ", mc) == (None, "na")
    assert mvn.gold_key("blue", mc) == "MC:B"
    it = {"question_type": "free_form", "answer_type": "integer"}
    assert mvn.normalize("5.0", it)[0] == mvn.normalize("5", it)[0] == "NUM:5"
    assert mvn.normalize("five?", it) == ("STR:five?", "str")
    fl = {"question_type": "free_form", "answer_type": "float", "precision": 1.0}
    assert mvn.normalize("1.53", fl)[0] == mvn.gold_key("1.5", fl) == "NUM:1.5"
    r = mvn.group_and_grade(["B", "blue", None, "A", "(B)"], "blue", mc)
    assert r["groups"] == [0, 0, NO_ANSWER, 1, 0] and r["group_correct"] == [True, False]
    assert mvn.levenshtein("kitten", "sitting") == 3


# ----------------------------------------------------------------------------- splits and orders
def test_split_lexicographic_and_fresh_rng():
    ids = [str(i) for i in range(1, 1001)]
    a = make_split(ids, 6304, 200)
    assert a["sorted"][:3] == ["1", "10", "100"]                    # string order
    exp = np.array(sorted(ids))[np.random.default_rng(6304).permutation(1000)]
    assert a["dev"] == exp[:200].tolist() and a["eval"] == exp[200:].tolist()
    b = make_split([f"x{i:04d}" for i in range(1000)], 6304, 200)
    assert a["permutation"] == b["permutation"]                     # fresh generator per dataset
    assert set(a["dev"]).isdisjoint(a["eval"]) and len(a["eval"]) == 800


def test_orders_stateless():
    o = make_orders(20, 1, 6304, 10, 32)
    o2 = make_orders(5, 1, 6304, 10, 32)
    assert (o[:5] == o2).all()
    assert all(sorted(o[3, k]) == list(range(32)) for k in range(10))
    assert not (make_orders(5, 0, 6304, 10, 32) == o2).all()
    r = np.random.default_rng([6304, 1, 4])
    assert (o[4, 0] == r.permutation(32)).all() and (o[4, 1] == r.permutation(32)).all()


# ----------------------------------------------------------------------------- selection rules
def test_vote_ties_earliest():
    assert vote([5, 7, 7, 5])[:2] == (5, 0)          # 2-2 tie -> answer whose earliest supporter is first
    assert vote([7, 5, 5, 7])[:2] == (7, 0)
    assert vote([1, 2, 3])[:2] == (1, 0)
    assert vote([NO_ANSWER, NO_ANSWER, 4])[0] == NO_ANSWER   # NO_ANSWER votes like any answer


def test_argmax_ties_and_nan():
    assert select("selfcert", [1, 2, 3], [0.5, 0.9, 0.9]) == 1
    assert select("selfcert", [1, 2, 3], [np.nan, -5.0, np.nan]) == 1
    assert select("shortest", [1, 2, 3], None, [30, 10, 10]) == 1
    for m in ("first", "sc", "selfcert", "selfcert_rwc", "shortest", "visconf_rwc"):
        assert select(m, [9], [0.1], [5]) == 0                       # N = 1


def test_rwc_average_ranks():
    s = [0.9, 0.5, 0.5, 0.1]
    assert avg_rank_desc(s).tolist() == [1.0, 2.5, 2.5, 4.0]
    w = rwc_weights(s)
    assert np.allclose(w, [1.0, 0.5, 0.5, 0.0])
    g, idx, tot = rwc([1, 2, 2, 1], s)          # W(1) = 1.0, W(2) = 1.0 -> tie -> earliest supporter (answer 1)
    assert (g, idx) == (1, 0) and tot == {1: 1.0, 2: 1.0}
    g, idx, tot = rwc([2, 1, 1, 2], [0.1, 0.9, 0.8, 0.2])    # ranks 4,1,2,3 -> W(1)=1+2/3, W(2)=1/3
    assert g == 1 and idx == 1
    assert vote_margin(tot) == pytest.approx((5 / 3 - 1 / 3) / 2)
    assert vote_margin({4: 2.0}) == 1.0
    assert np.allclose(rwc_weights([-np.inf, np.nan, 1.0]), [0.25, 0.25, 1.0])


# ----------------------------------------------------------------------------- VisConf
def _random_index(rng, n=12, with_zero=True):
    lens = rng.integers(0, 40, n)
    if with_zero:
        lens[3] = 0
    lens[5] = 1
    off = np.concatenate([[0], np.cumsum(lens)])
    k = rng.uniform(0, 10, off[-1]).astype(np.float32)
    k[off[6]:off[7]] = 4.0                         # constant segment (ties)
    k[: min(5, off[-1])] = np.round(k[: min(5, off[-1])])
    v = rng.uniform(0, 1, off[-1]).astype(np.float32)
    if lens[8] > 2:
        v[off[8] + 1] = np.nan
    return off, k, v


def test_visconf_fast_equals_naive():
    rng = np.random.default_rng(0)
    for trial in range(30):
        off, k, v = _random_index(rng)
        idx = VisConfIndex(off, k, v)
        n = len(off) - 1
        for _ in range(5):
            rows = rng.choice(n, size=int(rng.integers(2, n + 1)), replace=False)
            if np.diff(off)[rows].sum() == 0:
                continue
            segk = [k[off[r]:off[r + 1]].astype(np.float64) for r in rows]
            segv = [v[off[r]:off[r + 1]].astype(np.float64) for r in rows]
            Cn, Zn, Ztn, an, dn = naive_scores(segk, segv)
            V = np.array([np.nanmean(s) if np.isfinite(s).any() else np.nan for s in segv])
            C, Z, Zt, a, d = idx.scores(rows, V)
            assert a == pytest.approx(an) and d == pytest.approx(dn)
            np.testing.assert_allclose(C, Cn, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(Z, Zn, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(Zt, Ztn, rtol=1e-9, atol=1e-9)


def test_visconf_degenerate_spread():
    off = np.array([0, 4, 8])
    k = np.full(8, 3.0, np.float32)
    k[7] = 3.0 + 1e-9
    v = np.full(8, 0.5, np.float32)
    C, Z, Zt, a, d = VisConfIndex(off, k, v).scores([0, 1], np.array([0.5, 0.5]))
    Cn, Zn, Ztn, an, dn = naive_scores([k[:4].astype(float), k[4:].astype(float)], [v[:4].astype(float), v[4:].astype(float)])
    assert d == 1e-8
    np.testing.assert_allclose(C, Cn, atol=1e-6)


def test_visconf_zero_leakage():
    """Changing rollouts that are not in the candidate prefix never changes the prefix's scores."""
    rng = np.random.default_rng(1)
    off, k, v = _random_index(rng, n=16, with_zero=False)
    rows = np.array([2, 7, 9, 0])
    V = rng.uniform(0, 1, 4)
    base = VisConfIndex(off, k, v).scores(rows, V)
    k2, v2 = k.copy(), v.copy()
    for r in set(range(16)) - set(rows.tolist()):
        k2[off[r]:off[r + 1]] = rng.uniform(-50, 50, off[r + 1] - off[r])
        v2[off[r]:off[r + 1]] = rng.uniform(0, 1, off[r + 1] - off[r])
    other = VisConfIndex(off, k2, v2).scores(rows, V)
    for x, y in zip(base, other):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    # scores do not depend on the order of the candidate set either
    perm = VisConfIndex(off, k, v).scores(rows[::-1], V[::-1])
    np.testing.assert_array_equal(perm[0][::-1], base[0])


def test_tokenwise_identity():
    rng = np.random.default_rng(2)
    off = np.array([0, 30, 55, 90])
    k = rng.uniform(0, 8, 90)
    v = rng.uniform(0, 1, 90)
    idx = VisConfIndex(off, k, v)
    V = np.array([v[off[i]:off[i + 1]].mean() for i in range(3)])
    C, Z, Zt, a, d = idx.scores([0, 1, 2], V)
    for i in range(3):
        kk, vv = k[off[i]:off[i + 1]], v[off[i]:off[i + 1]]
        c = np.clip((kk - a) / d, 0, 1)
        cov = np.mean(c * vv) - c.mean() * vv.mean()
        assert Zt[i] - Z[i] == pytest.approx(cov, abs=1e-10)


# ----------------------------------------------------------------------------- adaptive controllers
def test_controller_stop_logic():
    # one question, one order: answers revealed in order; quality score = position-based
    R = 32
    group = np.array([[1, 1, 1, 2] + [1] * 4 + [2] * 8 + [3] * 16])
    correct = group == 2
    T = np.full((1, R), 10)
    orders = np.arange(R)[None, None, :]
    f = features(group, correct, T, lambda q, k, o, n: -np.arange(n, dtype=float), orders)
    # n=4: counts {1:3, 2:1} -> lead 0.75 unique; n=8: 7/8; n=16: {1:7,2:9}->2 leads 9/16; n=32: 3 leads 16/32
    assert np.allclose(f["lead_frac"][0, 0], [0.75, 7 / 8, 9 / 16, 0.5])
    st = stop_index(f, 0.75, None, "agreement")
    assert st[0, 0] == 0                                   # stops at n=4 with answer 1 (wrong)
    assert stop_index(f, 0.875, None, "agreement")[0, 0] == 1
    assert stop_index(f, 1.0, None, "agreement")[0, 0] == 3
    assert stop_index(f, None, None, "agreement")[0, 0] == 3
    # quality: RWC with scores favouring early rollouts agrees with SC at n=4; margin check
    assert f["agree_rwc"][0, 0, 0]
    m4 = f["margin"][0, 0, 0]
    assert stop_index(f, 0.75, m4 + 0.01, "quality")[0, 0] != 0
    r = evaluate(f, st)
    assert r["cost"] == 40 and r["rollouts"] == 4 and r["acc"] == 0.0
    # NO_ANSWER leader never stops
    g2 = np.full((1, R), NO_ANSWER)
    f2 = features(g2, g2 == 5, T, lambda q, k, o, n: np.zeros(n), orders)
    assert stop_index(f2, 0.5, None, "agreement")[0, 0] == 3


def test_selection_rule_ties():
    rows = [{"tau": 0.5, "gamma": None, "acc": 0.70, "cost": 100.0}, {"tau": 0.75, "gamma": None, "acc": 0.71, "cost": 100.0},
            {"tau": 0.625, "gamma": None, "acc": 0.71, "cost": 100.0}, {"tau": 1.0, "gamma": None, "acc": 0.60, "cost": 50.0},
            {"tau": None, "gamma": None, "acc": 0.72, "cost": 300.0}]
    assert select_setting(rows, 0.705)["tau"] == 0.625      # cheapest eligible; tie -> higher acc -> smaller tau
    assert select_setting(rows, 0.715)["tau"] is None        # only the fallback qualifies
