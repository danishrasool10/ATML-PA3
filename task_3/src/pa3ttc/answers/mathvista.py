"""MathVista answer normalisation (official `normalize_extracted_answer` rules) and grouping.

multi_choice: "(X)" -> X; a letter within range -> that option; otherwise the closest choice by Levenshtein
  distance (official `get_most_similar`, approved decision 5).  Answers are represented as option letters.
integer: str(int(float(x)));  float: str(round(float(x), precision));  list: str(x) without spaces;
text: stripped.  Missing extraction -> NO_ANSWER (never fuzzy-mapped).  A non-empty answer that fails the
numeric conversion forms a string group graded by string equality with the gold (decision 4).
Normalised-value equality is transitive, so group grade == individual grade.
"""
from __future__ import annotations

import json
import re

import numpy as np

NO_ANSWER = -1


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def as_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(c) for c in v]
        except json.JSONDecodeError:
            pass
        return [s]
    return [str(c) for c in x]


def _prec(p) -> int | None:
    try:
        p = float(p)
        return None if np.isnan(p) else int(p)
    except (TypeError, ValueError):
        return None


def normalize(extraction, meta: dict) -> tuple[str | None, str]:
    """-> (group key or None for NO_ANSWER, kind)."""
    if extraction is None or (isinstance(extraction, float) and np.isnan(extraction)):
        return None, "na"
    e = str(extraction).strip()
    if not e or e.lower() in ("none", "nan", "null"):
        return None, "na"
    qt, at = str(meta.get("question_type") or ""), str(meta.get("answer_type") or "")
    if qt == "multi_choice":
        choices = as_list(meta.get("choices"))
        letter = re.findall(r"\(([a-zA-Z])\)", e)
        if letter:
            e = letter[0].upper()
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        if e in letters:
            return "MC:" + e, "mc_letter"
        if not choices:
            return "STR:" + e.lower(), "str"
        d = [levenshtein(e, c) for c in choices]
        return "MC:" + letters[int(np.argmin(d))], ("mc_exact" if min(d) == 0 else "mc_fuzzy")
    try:
        if at == "integer":
            return "NUM:" + str(int(float(e))), "num"
        if at == "float":
            p = _prec(meta.get("precision"))
            return "NUM:" + str(round(float(e), p if p is not None else 2)), "num"
    except (ValueError, OverflowError):
        return "STR:" + e.lower(), "str"
    if at == "list":
        return "LIST:" + re.sub(r"\s+", "", e), "list"
    return "STR:" + e.lower(), "str"


def gold_key(gold, meta: dict) -> str | None:
    if str(meta.get("question_type") or "") == "multi_choice":
        choices = as_list(meta.get("choices"))
        g = str(gold).strip() if gold is not None else ""
        if g in choices:
            return "MC:" + chr(ord("A") + choices.index(g))
        return normalize(g, meta)[0]
    return normalize(gold, meta)[0]


def group_and_grade(answers: list, gold, meta: dict) -> dict:
    gk = gold_key(gold, meta)
    groups, keys, kinds, rep_answer = [], [], [], []
    seen: dict[str, int] = {}
    rkinds = []
    for a in answers:
        k, kind = normalize(a, meta)
        rkinds.append(kind)
        if k is None:
            groups.append(NO_ANSWER)
            continue
        if k not in seen:
            seen[k] = len(keys)
            keys.append(k)
            rep_answer.append(str(a).strip())
            kinds.append(kind)
        groups.append(seen[k])
    corr = [gk is not None and k == gk for k in keys]
    indiv = [False if g == NO_ANSWER else corr[g] for g in groups]
    return {"groups": groups, "group_answer": [k.split(":", 1)[1] for k in keys], "group_kind": kinds,
            "group_correct": corr, "indiv_correct": indiv, "rollout_kind": rkinds, "gold_key": gk}
