"""Step 4 evaluation of the live calls (no gold answers are used to choose between outputs).

Strategies (costs are actual tokens/time of the calls they consume):
  single          slot 0
  parallel        slots 1-3, self-consistency over the three answers (tie -> earliest, i.e. slot 1)
  refine          slots 1 + 4 (shared initial response charged), returns the revised answer; NO_ANSWER if the
                  revision has no \\boxed{} (approved decision 6, main row)
  refine_fallback same calls; keeps the initial answer when the revision has none (comparative extra)
  initial_only    slot 1 alone (reference)
Answers are grouped/graded with the same math-verify rule as the offline scorer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..answers.math_equiv import NO_ANSWER, group_and_grade
from ..config import odir
from ..io import df_to_md, read_json, read_jsonl, write_json
from ..offline.bootstrap import boot_idx, paired
from ..select.rules import vote

CALLS = {"single": [0], "parallel": [1, 2, 3], "refine": [1, 4], "refine_fallback": [1, 4], "initial_only": [1]}


def run(cfg: dict) -> dict:
    out = odir("live")
    qs = read_json(out / "questions.json")
    calls = {(r["qid"], r["slot"]): r for r in read_jsonl(out / "responses.jsonl")}
    rows, incomplete = [], []
    for item in qs:
        qid, gold = item["qid"], item["gold"]
        if not all((qid, s) in calls for s in range(5)):
            incomplete.append(qid)
            continue
        a = {s: calls[(qid, s)]["extracted"] for s in range(5)}
        r = {"qid": qid, "live_idx": item["live_idx"], "gold": gold}
        g = group_and_grade([a[0]], gold, cfg["mv_timeout_s"])
        r["single_answer"], r["single_correct"] = a[0], g["groups"][0] != NO_ANSWER and g["group_correct"][g["groups"][0]]
        g = group_and_grade([a[1], a[2], a[3]], gold, cfg["mv_timeout_s"])
        win, idx, tot = vote(g["groups"])
        r["parallel_answer"] = a[[1, 2, 3][idx]]
        r["parallel_correct"] = win != NO_ANSWER and g["group_correct"][win]
        r["parallel_n_groups"] = len(tot)
        r["parallel_all_distinct_tie"] = len(tot) == 3
        g = group_and_grade([a[1], a[4]], gold, cfg["mv_timeout_s"])
        gi, gr = g["groups"]
        ok = lambda x: x != NO_ANSWER and g["group_correct"][x]  # noqa: E731
        r["initial_answer"], r["initial_correct"] = a[1], ok(gi)
        r["refine_answer"], r["refine_correct"] = a[4], ok(gr)
        r["refine_fallback_answer"] = a[4] if gr != NO_ANSWER else a[1]
        r["refine_fallback_correct"] = ok(gr) if gr != NO_ANSWER else ok(gi)
        r["revision_changed_answer"] = gi != gr
        r["initial_only_answer"], r["initial_only_correct"] = a[1], ok(gi)
        for s, ss in CALLS.items():
            cs = [calls[(qid, x)] for x in ss]
            r[f"{s}_input_tokens"] = sum(c["input_tokens"] for c in cs)
            r[f"{s}_output_tokens"] = sum(c["output_tokens"] for c in cs)
            r[f"{s}_output_tokens_retained"] = sum(c["output_tokens_retained"] for c in cs)
            r[f"{s}_seconds"] = sum(c["seconds"] for c in cs)
            r[f"{s}_truncated_calls"] = sum(bool(c["truncated"]) for c in cs)
            r[f"{s}_n_calls"] = len(cs)
            r[f"{s}_errors"] = sum(c.get("error") is not None for c in cs)
            pk = [c["peak_gpu_mem_gib"] for c in cs if c.get("peak_gpu_mem_gib") is not None]
            r[f"{s}_peak_gpu_mem_gib"] = max(pk) if pk else np.nan
            r[f"{s}_extracted"] = r[f"{s}_answer"] is not None
        r["refine_final_call_truncated"] = bool(calls[(qid, 4)]["truncated"])
        r["single_final_call_truncated"] = bool(calls[(qid, 0)]["truncated"])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_question.csv", index=False)
    if df.empty:
        write_json(out / "summary.json", {"incomplete": incomplete, "n": 0})
        return {"n": 0}
    idx = boot_idx(len(df), cfg["n_boot"], cfg["seed"])
    summ = []
    for s in CALLS:
        summ.append({"strategy": s, "accuracy": df[f"{s}_correct"].mean(),
                     "extraction_rate": df[f"{s}_extracted"].mean(),
                     "truncation_rate_calls": df[f"{s}_truncated_calls"].sum() / df[f"{s}_n_calls"].sum(),
                     "mean_input_tokens": df[f"{s}_input_tokens"].mean(),
                     "mean_output_tokens": df[f"{s}_output_tokens"].mean(),
                     "mean_output_tokens_retained": df[f"{s}_output_tokens_retained"].mean(),
                     "mean_seconds": df[f"{s}_seconds"].mean(), "total_seconds": df[f"{s}_seconds"].sum(),
                     "max_peak_gpu_mem_gib": df[f"{s}_peak_gpu_mem_gib"].max(),
                     "errors": int(df[f"{s}_errors"].sum()), "output_allowance": {
                         "single": 3072, "parallel": 3072, "refine": 3072, "refine_fallback": 3072,
                         "initial_only": 1024}[s]})
    sdf = pd.DataFrame(summ)
    ic, rc = df["initial_correct"].astype(bool), df["refine_correct"].astype(bool)
    trans = {"wrong_to_correct": int((~ic & rc).sum()), "correct_to_wrong": int((ic & ~rc).sum()),
             "correct_to_correct": int((ic & rc).sum()), "wrong_to_wrong": int((~ic & ~rc).sum()),
             "answer_changed": int(df["revision_changed_answer"].sum()),
             "revision_no_answer": int((~df["refine_extracted"].astype(bool)).sum()),
             "revision_truncated": int(df["refine_final_call_truncated"].sum())}
    cmp = {f"{a}_minus_{b}": paired(df[f"{a}_correct"].astype(float).to_numpy(),
                                     df[f"{b}_correct"].astype(float).to_numpy(), idx)
           for a, b in [("parallel", "single"), ("refine", "single"), ("refine", "parallel"),
                        ("refine", "initial_only"), ("refine_fallback", "refine")]}
    summary = {"n_questions": len(df), "incomplete": incomplete, "transitions_vs_initial": trans,
               "bootstrap": cmp, "parallel_three_way_ties": int(df["parallel_all_distinct_tie"].sum()),
               "backend": read_json(out / "backend.json") if (out / "backend.json").exists() else None}
    sdf.to_csv(out / "summary.csv", index=False)
    write_json(out / "summary.json", {**summary, "table": summ})
    (out / "summary.md").write_text(
        f"## Live strategies on {len(df)} MATH held-out questions (Qwen2.5-3B-Instruct, FP16, batch 1)\n\n"
        + df_to_md(sdf, floatfmt=".3f") + "\n### Revision transitions vs the shared initial answer\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in trans.items()) + "\n\n### Paired bootstrap (accuracy differences)\n\n"
        + "\n".join(f"- {k}: {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" for k, v in cmp.items()) + "\n")
    print(f"[live-eval] {summary['transitions_vs_initial']}", flush=True)
    return summary
