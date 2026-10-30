"""Report figures/tables for Task 2 from $PA3_OUTPUT_ROOT (runs/vt196, runs/vt49, data, features).

    python scripts/make_report_assets.py

Writes report_assets/{figures,tables}/: learning curves with stage boundaries, final-result tables (copied
from each run's summary.md + a 196-vs-49 comparison), resources (feature caching separated from training),
the qualitative example sheet and its selection rule.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TASK / "src"), str(TASK.parent / "common")]

from pa3common.utils import output_root, write_json  # noqa: E402
from pa3vlm.eval import qualitative as Q  # noqa: E402

RUNS = ("vt196", "vt49")


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def curves(run: str, R: Path, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr, dv = jl(R / "runs" / run / "train_log.jsonl"), jl(R / "runs" / run / "dev_metrics.jsonl")
    if not tr:
        return
    cfg = json.loads((R / "runs" / run / "config.json").read_text())
    ends = np.cumsum([s["updates"] for s in cfg["stages"]])
    u = np.array([r["update"] for r in tr])
    loss = np.array([r["loss"] for r in tr])
    k = 25
    sm = np.convolve(loss, np.ones(k) / k, mode="valid") if len(loss) >= k else loss
    du = [r["update"] for r in dv]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    ax = ax.ravel()
    ax[0].plot(u, loss, alpha=0.25, lw=0.6, label="per update")
    ax[0].plot(u[len(u) - len(sm):], sm, lw=1.5, label=f"{k}-update mean")
    ax[0].set_title("training loss (example-weighted CE)")
    ax[1].plot(du, [r["caption_loss"] for r in dv], "o-", label="correct image")
    ax[1].plot(du, [r["caption_loss_noimage"] for r in dv], "s--", label="no image")
    ax[1].set_title("dev caption reference loss")
    for i, suite in ((2, "grounding"), (3, "vqa")):
        for cond, st in (("correct", "o-"), ("swap", "s--"), ("none", "^:")):
            ax[i].plot(du, [r[f"{suite}/all/nll_{cond}"] for r in dv], st, label=cond)
        ax[i].set_title(f"dev {suite}: reference-answer NLL")
    for suite, st in (("grounding", "o-"), ("vqa", "s-")):
        ax[4].plot(du, [r[f"{suite}/all/delta_nll_swap"] for r in dv], st, label=f"{suite} ΔNLL swap")
        ax[5].plot(du, [r[f"{suite}/all/acc_greedy"] for r in dv], st, label=f"{suite} greedy acc (correct img)")
    ax[4].axhline(0, color="k", lw=0.5)
    ax[4].set_title("image-swap penalty ΔNLL = NLL(partner) − NLL(correct)")
    ax[5].set_title("dev greedy accuracy")
    for a in ax:
        for e in ends[:-1]:
            a.axvline(e, color="grey", ls=":", lw=1)
        a.set_xlabel("update")
        a.legend(fontsize=7)
    fig.suptitle(f"{run}: learning curves (dotted lines = stage boundaries {', '.join(map(str, ends[:-1]))})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{run}_learning_curves.png", dpi=110)
    plt.close(fig)


def _n(x) -> str:
    return f"{x:,}" if isinstance(x, int) else "?"


def resources(R: Path) -> tuple[dict, str]:
    res: dict = {"feature_cache": [json.loads(p.read_text()) for p in sorted((R / "features").glob("cache_shard*.json"))]}
    for f in res["feature_cache"]:
        f.pop("preprocessing", None)
    md = ["| run | stage | trainable params | connector | LoRA | train time (h) | s/update | peak GPU mem (GiB) |",
          "|---|---|---|---|---|---|---|---|"]
    for run in RUNS:
        tr, ev = jl(R / "runs" / run / "train_log.jsonl"), jl(R / "runs" / run / "events.jsonl")
        if not tr:
            continue
        starts = {e["stage"]: e for e in ev if e.get("event") == "stage_start"}
        dv = jl(R / "runs" / run / "dev_metrics.jsonl")
        rr = {"stages": {}, "dev_eval_s": sum(r.get("seconds", 0) for r in dv)}
        for s in sorted({r["stage"] for r in tr}):
            rows = [r for r in tr if r["stage"] == s]
            secs = sum(r["seconds"] for r in rows)
            mem = max((r.get("peak_mem_gib", 0) for r in rows), default=0)
            st = starts.get(s, {})
            rr["stages"][s] = {"updates": len(rows), "train_s": secs, "peak_mem_gib": mem,
                               "trainable": st.get("trainable_params"), "skipped": sum(r["skipped"] for r in rows)}
            md.append(f"| {run} | {s} | {_n(st.get('trainable_params'))} | {_n(st.get('connector_params'))} | "
                      f"{_n(st.get('lora_params'))} | {secs / 3600:.2f} | {secs / max(1, len(rows)):.2f} | {mem:.2f} |")
        fe = R / "runs" / run / "final_eval"
        if (fe / "eval_resources.json").exists():
            rr["final_eval"] = json.loads((fe / "eval_resources.json").read_text())
            gt = [r["gen_time"] for f in fe.glob("*__*.jsonl") for r in jl(f)]
            rr["final_eval"]["gen_ms_per_answer_mean"] = 1000 * float(np.mean(gt)) if gt else None
        cal = R / "runs" / run / "calibration.json"
        if cal.exists():
            c = json.loads(cal.read_text())
            rr["calibration"] = {k: c[k] for k in ("scale", "text_median_norm", "visual_median_norm_before")}
        res[run] = rr
    fc = res["feature_cache"]
    if fc:
        md.append(f"\nFeature caching (separate from training): {sum(f['n_images'] for f in fc)} images, "
                  f"max shard wall time {max(f['seconds'] for f in fc) / 60:.1f} min on {len(fc)} GPU(s), "
                  f"{sum(f['bytes'] for f in fc) / 2 ** 30:.2f} GiB FP16, peak GPU mem "
                  f"{max(f['memory'].get('peak_alloc_gib', 0) for f in fc):.2f} GiB.")
    for run in RUNS:
        if run in res:
            md.append(f"\n{run}: dev evaluation total {res[run]['dev_eval_s'] / 60:.1f} min; "
                      f"calibration {res[run].get('calibration')}; final eval {res[run].get('final_eval', {}).get('gen_ms_per_answer_mean')} ms/answer")
    return res, "\n".join(md)


def comparison(R: Path) -> str:
    rows = ["| suite | metric | condition | vt196 | vt49 |", "|---|---|---|---|---|"]
    S = {r: json.loads((R / "runs" / r / "final_eval" / "summary.json").read_text())
         for r in RUNS if (R / "runs" / r / "final_eval" / "summary.json").exists()}
    if len(S) < 2:
        return ""
    for suite in ("grounding", "vqa"):
        for cond in ("correct", "swap", "mismatched", "none"):
            for m in ("acc", "pair_acc"):
                cells = []
                for r in RUNS:
                    v = S[r]["suites"][suite]["all"]["conditions"][cond][m]
                    cells.append(f"{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]")
                rows.append(f"| {suite} | {m} | {cond} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    R = output_root()
    A = R / "report_assets"
    (A / "figures").mkdir(parents=True, exist_ok=True)
    (A / "tables").mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        curves(run, R, A / "figures")
        sm = R / "runs" / run / "final_eval" / "summary.md"
        if sm.exists():
            (A / "tables" / f"{run}_final_results.md").write_text(f"## {run} final test results\n\n" + sm.read_text())
    res, md = resources(R)
    write_json(A / "tables" / "resources.json", res)
    (A / "tables" / "resources.md").write_text(md)
    cmp = comparison(R)
    if cmp:
        (A / "tables" / "vt196_vs_vt49.md").write_text("## 196 vs 49 visual tokens (accuracy %, 95% CI)\n\n" + cmp)
    fe = R / "runs" / "vt196" / "final_eval"
    if fe.exists():
        chosen = Q.select(Q.load_records(fe), 6304)
        write_json(A / "tables" / "qualitative_examples.json", {"rule": Q.__doc__, "examples": chosen})
        img_dir = Path(os.environ.get("PA3_SCRATCH", "/tmp/pa3_scratch")) / "images"
        for suite in ("grounding", "vqa"):
            Q.render([c for c in chosen if c["suite"] == suite], img_dir, A / "figures" / f"qualitative_{suite}.png")
    print(f"[report] assets in {A}", flush=True)


if __name__ == "__main__":
    main()
