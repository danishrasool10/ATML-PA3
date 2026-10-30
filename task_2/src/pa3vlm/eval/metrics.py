"""Per-example scoring and per-(suite, condition, question type) summaries.

grounding: normalised exact match against the canonical answer ("yes"/"no"/digit).
vqa      : local soft-consensus accuracy against all ten (normalised) human answers.
Pair accuracy: both predictions equal their original canonical answers.  Identical-answer rate: both
predictions of a pair are the same normalised string.  Empty outputs stay in every denominator.
Under swaps we also record whether a prediction equals the *partner's* canonical answer (follows the new
image) or neither canonical answer (unsupported).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..data.normalize import normalize_answer, soft_score


def score(suite: str, pred_norm: str, ex: dict) -> float:
    if suite == "vqa":
        return soft_score(pred_norm, [normalize_answer(a) for a in ex["answers"]])
    return float(pred_norm == ex["canonical"])


def pair_table(records: list[dict]) -> dict[tuple, dict]:
    """Group example records (one condition) by pair -> dict with both slots."""
    out: dict[tuple, dict] = defaultdict(dict)
    for r in records:
        out[(r["suite"], r["condition"], r["pair_id"])][r["slot"]] = r
    for k, v in out.items():
        assert set(v) == {0, 1}, k
    return out


def pair_values(pair: dict) -> dict[str, float]:
    a, b = pair[0], pair[1]
    return {
        "acc": (a["score"] + b["score"]) / 2,
        "pair_acc": float(a["pred_norm"] == a["canonical"] and b["pred_norm"] == b["canonical"]),
        "identical": float(a["pred_norm"] == b["pred_norm"]),
        "empty": (float(a["pred_norm"] == "") + float(b["pred_norm"] == "")) / 2,
        "follows_partner": (float(a["pred_norm"] == a["partner_canonical"]) +
                            float(b["pred_norm"] == b["partner_canonical"])) / 2,
        "unsupported": sum(float(r["pred_norm"] not in (r["canonical"], r["partner_canonical"])) for r in (a, b)) / 2,
        "nll": (a["nll"] + b["nll"]) / 2,
        "gen_time": (a["gen_time"] + b["gen_time"]) / 2,
        "hit_cap": (float(not a["stopped"]) + float(not b["stopped"])) / 2,
    }


METRICS = ("acc", "pair_acc", "identical", "empty", "follows_partner", "unsupported", "nll", "gen_time", "hit_cap")


def pair_matrix(records: list[dict]) -> dict:
    """{(suite, condition): {"pair_ids": [...], "kinds": [...], metric: np.array[P]}} with pairs sorted by id."""
    pt = pair_table(records)
    groups: dict[tuple, list] = defaultdict(list)
    for (suite, cond, pid), pr in pt.items():
        groups[(suite, cond)].append((pid, pr))
    out = {}
    for k, lst in groups.items():
        lst.sort(key=lambda x: x[0])
        vals = [pair_values(pr) for _, pr in lst]
        out[k] = {"pair_ids": [p for p, _ in lst], "kinds": [pr[0]["kind"] for _, pr in lst],
                  **{m: np.array([v[m] for v in vals], dtype=np.float64) for m in METRICS}}
    return out
