"""MATH answer groups and grades with math-verify (gold-free grouping; grading uses the same rule).

Grouping, per question, rollouts in rollout-ID order: an empty/missing extraction -> NO_ANSWER (-1).  Identical
normalised strings share a group without calling math-verify.  Otherwise the answer is parsed (as `$s$`, then
`\\boxed{s}`, then raw); it joins the first earlier group whose representative r satisfies verify(r, answer),
else it starts a new group.  A non-empty answer that math-verify cannot parse forms a string group keyed by its
lower-cased normalised string (approved decision 4).  Grade of a group = verify(gold, representative) (decision
3); every rollout is also graded on its own so the disagreement rate can be logged.
"""
from __future__ import annotations

import re

NO_ANSWER = -1


def norm_str(s) -> str | None:
    if s is None:
        return None
    if isinstance(s, float) and s != s:
        return None
    s = str(s).strip()
    while len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s)
    if not s or s.lower() in ("none", "nan", "null"):
        return None
    return s


def parse_ans(s: str, timeout: int = 5) -> list:
    from math_verify import parse

    for cand in (f"${s}$", f"\\boxed{{{s}}}", s):
        try:
            p = parse(cand, parsing_timeout=timeout)
        except Exception:  # noqa: BLE001
            p = []
        if p:
            return p
    return []


def _verify(gold, target, timeout: int = 5) -> bool:
    from math_verify import verify

    if not gold or not target:
        return False
    try:
        return bool(verify(gold, target, timeout_seconds=timeout))
    except Exception:  # noqa: BLE001
        return False


def group_and_grade(answers: list, gold, timeout: int = 5) -> dict:
    """answers: extracted strings in rollout order.  Returns per-rollout group ids, per-group representative /
    kind / grade, and per-rollout individual grades."""
    gold_n = norm_str(gold)
    gold_p = parse_ans(gold_n, timeout) if gold_n else []
    groups: list[int] = []
    reps: list[dict] = []          # {"answer", "kind", "parsed"|"key"}
    by_str: dict[str, int] = {}
    parsed_of: dict[str, list] = {}
    for a in answers:
        n = norm_str(a)
        if n is None:
            groups.append(NO_ANSWER)
            continue
        if n in by_str:
            groups.append(by_str[n])
            continue
        p = parse_ans(n, timeout)
        parsed_of[n] = p
        g = None
        if p:
            for gi, rep in enumerate(reps):
                if rep["kind"] == "expr" and _verify(rep["parsed"], p, timeout):
                    g = gi
                    break
            if g is None:
                reps.append({"answer": n, "kind": "expr", "parsed": p})
                g = len(reps) - 1
        else:
            key = "STR:" + n.lower()
            g = next((gi for gi, rep in enumerate(reps) if rep["kind"] == "str" and rep["key"] == key), None)
            if g is None:
                reps.append({"answer": n, "kind": "str", "key": key})
                g = len(reps) - 1
        by_str[n] = g
        groups.append(g)

    def grade(kind, parsed=None, s=None) -> bool:
        if gold_n is None:
            return False
        if kind == "expr":
            return _verify(gold_p, parsed, timeout) if gold_p else False
        return s is not None and s.lower() == gold_n.lower()

    rep_correct = [grade(r["kind"], r.get("parsed"), r["answer"]) for r in reps]
    indiv = []
    for a, g in zip(answers, groups):
        if g == NO_ANSWER:
            indiv.append(False)
            continue
        n = norm_str(a)
        p = parsed_of.get(n)
        if p is None:
            p = parse_ans(n, timeout)
            parsed_of[n] = p
        indiv.append(grade("expr", p) if p else grade("str", s=n))
    return {"groups": groups, "group_answer": [r["answer"] for r in reps], "group_kind": [r["kind"] for r in reps],
            "group_correct": rep_correct, "indiv_correct": indiv, "gold_parsed": bool(gold_p)}
