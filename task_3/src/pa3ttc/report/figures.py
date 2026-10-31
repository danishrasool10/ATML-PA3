"""Report figures (PNG) and the combined summary.md.  Every figure is built only from saved outputs, so this
step can be rerun on its own; a missing input skips that figure."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import odir
from ..io import df_to_md, read_json

SURF, INK, INK2, GRID, ORACLE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8985"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FIG_METHODS = {"math": ["first", "sc", "logp", "selfcert", "selfcert_rwc", "shortest"],
               "mathvista": ["first", "sc", "selfcert", "selfcert_rwc", "visconf", "visconf_rwc", "vision_rwc"]}
COLOR = {m: SERIES[i] for i, m in enumerate(["first", "sc", "selfcert", "selfcert_rwc", "visconf", "visconf_rwc",
                                              "vision_rwc", "logp"])}
COLOR["shortest"] = SERIES[4]
DSNAME = {"math": "MATH (Qwen2.5-3B-Instruct)", "mathvista": "MathVista (Qwen2.5-VL-3B-Instruct)"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
                         "lines.linewidth": 1.8, "lines.markersize": 5})
    return plt


def fig_dir() -> Path:
    return odir("report_assets", "figures")


def acc_curves(ds: str):
    from ..offline.runner import LABEL

    plt = _plt()
    s = pd.read_csv(odir("offline", ds) / "summary_eval.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for m in FIG_METHODS[ds]:
        x = s[s["method"] == m].sort_values("N")
        axes[0].plot(x["N"], x["acc"], marker="o", color=COLOR[m], label=LABEL[m])
        axes[1].plot(x["cost"], x["acc"], marker="o", color=COLOR[m], label=LABEL[m])
    o = s[s["method"] == "sc"].sort_values("N")
    axes[0].plot(o["N"], o["oracle"], ls="--", color=ORACLE, label="Oracle coverage (upper bound)")
    ad = odir("adaptive", ds) / "heldout.json"
    if ad.exists():
        h = read_json(ad)
        for fam, mk in (("agreement", "D"), ("quality", "s")):
            axes[1].plot(h[fam]["cost"], h[fam]["acc"], marker=mk, ls="none", ms=8, color=INK,
                         label=f"Adaptive ({fam})")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    axes[0].set_xlabel("Candidate budget N")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Mean generated tokens per question (offline cost proxy)")
    for a in axes:
        a.set_ylabel("Accuracy (held-out)")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    extra = [(h, l) for h, l in zip(h1, l1) if l not in l0]
    fig.legend(h0 + [h for h, _ in extra], l0 + [l for _, l in extra], loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"{DSNAME[ds]}: accuracy vs budget and vs token cost", color=INK)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(fig_dir() / f"{ds}_accuracy_curves.png", dpi=150)
    plt.close(fig)


def decomposition(ds: str):
    plt = _plt()
    s = pd.read_csv(odir("offline", ds) / "summary_eval.csv")
    meths = ["sc", "selfcert_rwc"] + (["visconf_rwc"] if ds == "mathvista" else [])
    fig, axes = plt.subplots(1, len(meths), figsize=(4.2 * len(meths), 3.8), sharey=True)
    from ..offline.runner import LABEL

    for a, m in zip(np.atleast_1d(axes), meths):
        x = s[s["method"] == m].sort_values("N")
        lab = [str(n) for n in x["N"]]
        b = np.zeros(len(x))
        for col, c, name in (("acc", SERIES[2], "Correct"), ("missed", SERIES[3], "Correct available, missed"),
                             ("no_correct", ORACLE, "No correct candidate")):
            a.bar(lab, x[col], bottom=b, color=c, label=name, edgecolor=SURF, linewidth=1.5)
            b += x[col].to_numpy()
        a.set_title(LABEL[m], fontsize=10)
        a.set_xlabel("N")
    np.atleast_1d(axes)[0].set_ylabel("Fraction of held-out questions")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{DSNAME[ds]}: sampling vs selection failures", color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir() / f"{ds}_failure_decomposition.png", dpi=150)
    plt.close(fig)


def marginal():
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, ds in zip(axes, ("math", "mathvista")):
        m = pd.read_csv(odir("compute", ds) / "marginal_samples.csv")
        a.plot(m["n"], m["p_new_group"], color=SERIES[0], label="adds a new answer group")
        a.plot(m["n"], m["p_new_answered_group"], color=SERIES[1], label="adds a new non-missing answer")
        a.plot(m["n"], m["p_first_correct_appears"], color=SERIES[2], label="first correct answer appears")
        a.set_title(DSNAME[ds], fontsize=10)
        a.set_xlabel("Sample index n (revealed order)")
    axes[0].set_ylabel("Probability (held-out, averaged over orders)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir() / "marginal_value_of_samples.png", dpi=150)
    plt.close(fig)


def stopping():
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for a, ds in zip(axes, ("math", "mathvista")):
        h = read_json(odir("adaptive", ds) / "heldout.json")
        ns = ["4", "8", "16", "32"]
        w = 0.38
        for i, fam in enumerate(("agreement", "quality")):
            d = h[fam]["stop_distribution"]
            a.bar(np.arange(4) + (i - 0.5) * w, [d.get(n, d.get(int(n), 0)) for n in ns], width=w - 0.04,
                  color=SERIES[i], label=f"{fam} (τ={h[fam]['setting']['tau']}, γ={h[fam]['setting']['gamma']})")
        a.set_xticks(np.arange(4), ns)
        a.set_xlabel("Rollouts consumed at stop")
        a.set_title(DSNAME[ds], fontsize=10)
        a.legend(fontsize=8)
    axes[0].set_ylabel("Fraction of (question, order) runs")
    fig.tight_layout()
    fig.savefig(fig_dir() / "adaptive_stopping_distribution.png", dpi=150)
    plt.close(fig)


def quartiles():
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, ds in zip(axes, ("math", "mathvista")):
        t = pd.read_csv(odir("analysis", ds) / "associations.csv")
        a.barh(t["signal"], t["top_minus_bottom_quartile_mixed"], color=SERIES[0])
        a.axvline(0, color=INK2, lw=0.8)
        a.set_title(DSNAME[ds], fontsize=10)
        a.set_xlabel("P(correct | top quartile) - P(correct | bottom quartile)\n(mixed held-out questions, N=32)")
    fig.tight_layout()
    fig.savefig(fig_dir() / "score_quartile_gaps.png", dpi=150)
    plt.close(fig)


def conf_vision():
    plt = _plt()
    df = pd.read_parquet(odir("analysis", "mathvista") / "conf_vision_per_rollout.parquet")
    perq = df.groupby("qid")["rho_c_v"].mean().dropna()
    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4])
    ax_h = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1, 1], sharex=ax_c)
    ax_h.hist(perq, bins=40, color=SERIES[0], edgecolor=SURF)
    ax_h.axvline(0, color=INK2, lw=0.8)
    ax_h.set_xlabel("Within-rollout corr(c_t, v_t), mean per question")
    ax_h.set_ylabel("Held-out questions")
    tj = read_json(odir("analysis", "mathvista") / "trajectories.json")["series"][:2]
    for i, s in enumerate(tj):
        x = np.arange(len(s["c_roll"])) * s["step"]
        lab = f"{s['which'].replace('_', ' ')} rollout ({'correct' if s['correct'] else 'wrong'})"
        ax_c.plot(x, s["c_roll"], color=SERIES[i], label=lab)
        ax_v.plot(x, s["v_roll"], color=SERIES[i], label=lab)
    ax_c.set_ylabel("calibrated c_t")
    ax_v.set_ylabel("visual share v_t")
    ax_v.set_xlabel("Generated token position")
    if tj:
        ax_c.set_title(f"Illustrative question {tj[0]['qid']} ({tj[0]['bucket'].replace('_', ' ')}); "
                       "rolling mean, window 16", fontsize=9)
        ax_c.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir() / "mathvista_confidence_vision.png", dpi=150)
    plt.close(fig)


def live():
    plt = _plt()
    s = pd.read_csv(odir("live") / "summary.csv")
    s = s[s["strategy"].isin(["single", "parallel", "refine"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for i, (_, r) in enumerate(s.iterrows()):
        for a, col, lab in ((axes[0], "mean_output_tokens", "Mean output tokens per question"),
                            (axes[1], "mean_seconds", "Mean seconds per question (T4, batch 1)")):
            a.plot(r[col], r["accuracy"], marker="o", ms=9, ls="none", color=SERIES[i], label=r["strategy"])
            a.set_xlabel(lab)
    axes[0].set_ylabel("Accuracy (50 MATH questions)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir() / "live_strategies.png", dpi=150)
    plt.close(fig)


def summary_md() -> None:
    R = odir()
    parts = ["# PA3 Task 3 — results summary\n"]
    for rel in ["offline/math/summary_eval.md", "offline/mathvista/summary_eval.md", "analysis/math/associations.md",
                "analysis/mathvista/associations.md", "adaptive/math/heldout.md", "adaptive/mathvista/heldout.md",
                "compute/math/compute.md", "compute/mathvista/compute.md", "live/summary.md"]:
        p = R / rel
        if p.exists():
            parts.append(p.read_text())
    for rel in ["answers/scorer_manifest.json", "analysis/mathvista/conf_vision_summary.json"]:
        p = R / rel
        if p.exists():
            j = json.loads(p.read_text())
            j.pop("rules", None)
            parts.append(f"\n## {rel}\n\n```json\n{json.dumps(j, indent=1)[:6000]}\n```\n")
    for ds in ("math", "mathvista"):
        p = R / "offline" / ds / "bootstrap.csv"
        if p.exists():
            b = pd.read_csv(p)
            b = b[b["N"].isin([8, 32])][["N", "A", "B", "a", "b", "diff", "lo", "hi"]]
            parts.append(f"\n## {ds}: paired bootstrap (held-out, 1,000 question resamples), N=8 and N=32\n\n" + df_to_md(b))
    tdir = odir("report_assets", "tables")
    for rel in ["offline/math/summary_eval.md", "offline/mathvista/summary_eval.md", "adaptive/math/heldout.md",
                "adaptive/mathvista/heldout.md", "live/summary.md"]:
        p = R / rel
        if p.exists():
            (tdir / rel.replace("/", "_")).write_text(p.read_text())
    (odir("report_assets") / "summary.md").write_text("\n".join(parts))


def run(cfg: dict) -> dict:
    done, failed = [], {}
    jobs = [("acc_math", lambda: acc_curves("math")), ("acc_mathvista", lambda: acc_curves("mathvista")),
            ("decomp_math", lambda: decomposition("math")), ("decomp_mathvista", lambda: decomposition("mathvista")),
            ("marginal", marginal), ("stopping", stopping), ("quartiles", quartiles), ("conf_vision", conf_vision),
            ("live", live), ("summary", summary_md)]
    for name, fn in jobs:
        try:
            fn()
            done.append(name)
        except Exception as ex:  # noqa: BLE001
            failed[name] = f"{type(ex).__name__}: {ex}"
            traceback.print_exc()
    print(f"[report] done={done} failed={list(failed)}", flush=True)
    return {"done": done, "failed": failed}
