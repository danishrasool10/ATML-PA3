"""Build report tables and figures from whatever outputs exist (re-runnable after every stage).

Tables (report_assets/tables): benchmarks.csv/.md (7 conditions x 4 benchmarks, primary score,
raw/normalised accuracy, delta vs baseline in pp, GSM8K truncation/extraction), resources.csv/.md,
diagnostics_weights.md, diagnostics_activations.md, regularization.md.
Figures (report_assets/figures): sft_loss.png, lr.png, l2sp_penalty.png, benchmark_deltas.png,
activation_ratios.png.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[2] / "common")]
from pa3.eval.conditions import ORDER  # noqa: E402
from pa3.utils.common import find_artifact, output_root  # noqa: E402

RUNS = ["full_ft", "full_ft_l2sp", "lora_r4", "lora_r16", "oplora_r16"]
RUN_LABEL = {"full_ft": "Full FT", "full_ft_l2sp": "Full FT + L2-SP", "lora_r4": "LoRA-4",
             "lora_r16": "LoRA-16", "oplora_r16": "OPLoRA-16"}
COND_LABEL = {"baseline": "Baseline", "fullft_u0625": "Full FT (mid)", "fullft_u1250": "Full FT",
              "l2sp_u1250": "Full FT + L2-SP", "lora4_u1250": "LoRA-4", "lora16_u1250": "LoRA-16",
              "oplora16_u1250": "OPLoRA-16"}
TASKS = [("gsm8k_cot", "GSM8K"), ("hellaswag", "HellaSwag"), ("boolq", "BoolQ"), ("arc_easy", "ARC-Easy")]
# reference categorical palette (light), fixed order; text never wears series colour
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False,
        "lines.linewidth": 1.6, "figure.dpi": 150,
    })


def _ema(xs, a=0.05):
    out, m = [], None
    for x in xs:
        m = x if m is None else (1 - a) * m + a * x
        out.append(m)
    return out


def _md_table(header: list[str], rows: list[list]) -> str:
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join("" if v is None else str(v) for v in r) + " |\n"
    return s


def fmt(v, nd=1, pct=True):
    if v is None:
        return "—"
    return f"{100 * v:.{nd}f}" if pct else f"{v:.{nd}f}"


def benchmarks(root: Path, tdir: Path) -> dict:
    aggs = {}
    for c in ORDER:
        p = find_artifact(f"eval/{c}/aggregate.json")
        if p:
            aggs[c] = json.loads(p.read_text())
    base = aggs.get("baseline", {}).get("primary", {})
    out_rows, md_rows = [], []
    for c in ORDER:
        if c not in aggs:
            continue
        pr = aggs[c]["primary"]
        a = aggs[c].get("gsm8k_audit", {})
        row = {"condition": c}
        md = [COND_LABEL[c]]
        for t, _ in TASKS:
            v = pr.get(t, {}).get("value")
            b = base.get(t, {}).get("value")
            d = None if (v is None or b is None) else 100 * (v - b)
            row[f"{t}_primary"] = v
            row[f"{t}_delta_pp"] = d
            if t != "gsm8k_cot":
                row[f"{t}_acc_raw"] = pr.get(t, {}).get("acc_raw")
                row[f"{t}_acc_norm"] = pr.get(t, {}).get("acc_norm")
            else:
                row["gsm8k_strict"] = pr.get(t, {}).get("strict_match")
            md.append(f"{fmt(v)} ({'' if d is None or d < 0 else '+'}{'—' if d is None else f'{d:.1f}'})")
        for t in ("hellaswag", "arc_easy"):
            md.append(fmt(row.get(f"{t}_acc_raw")))
        row.update({"gsm8k_truncated_rate": a.get("truncated_rate"),
                    "gsm8k_flex_unextractable_rate": a.get("flexible_unextractable_rate"),
                    "gsm8k_mean_gen_tokens": a.get("mean_gen_tokens")})
        md += [fmt(a.get("truncated_rate")), fmt(a.get("flexible_unextractable_rate")),
               fmt(a.get("mean_gen_tokens"), 0, pct=False)]
        out_rows.append(row)
        md_rows.append(md)
    if out_rows:
        keys = sorted({k for r in out_rows for k in r}, key=lambda k: (k != "condition", k))
        with open(tdir / "benchmarks.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(out_rows)
        hdr = ["Condition", "GSM8K flex-EM (Δpp)", "HellaSwag acc_norm (Δpp)", "BoolQ acc (Δpp)",
               "ARC-E acc_norm (Δpp)", "HellaSwag acc (raw)", "ARC-E acc (raw)", "GSM8K truncated %",
               "GSM8K unextractable %", "GSM8K mean gen tok"]
        (tdir / "benchmarks.md").write_text(
            "Scores in %, Δ vs baseline in percentage points.\n\n" + _md_table(hdr, md_rows))
    return {r["condition"]: r for r in out_rows}


def resources(tdir: Path) -> None:
    rows = []
    for r in RUNS:
        p = find_artifact(f"runs/{r}/resources.json")
        if not p:
            continue
        res = json.loads(p.read_text())
        peak = max((m.get("peak_alloc_gib", 0) for m in res.get("peak_memory_per_rank", [])), default=None)
        ck = res.get("checkpoints", [])
        snap = sum(e.get("file_bytes", 0) for e in res.get("exports", {}).values())
        extra = ""
        if "l2sp" in res:
            l = res["l2sp"]
            extra = (f"θ0 {l['ref_dtype']} {l['ref_storage_bytes'] / 2**30:.2f} GiB "
                     f"({'host' if l['ref_on_host'] else 'GPU'}); penalty {l['penalty_seconds_total'] / 60:.1f} min")
        if "bases_meta" in res:
            b = res["bases_meta"]
            extra = (f"bases {int(b['basis_bytes_fp32']) / 2**20:.0f} MiB; SVD {float(b['svd_seconds']):.1f} s; "
                     f"export {res.get('exports', {}).get('1250', {}).get('export_seconds', 0):.2f} s")
        rows.append([RUN_LABEL[r], f"{res['trainable_params']:,}", f"{res['train_seconds'] / 3600:.2f}",
                     f"{peak:.2f}" if peak else "—", f"{res['tokens_per_second']:.0f}",
                     " / ".join(f"u{e['update']}: {e['bytes'] / 2**30:.2f}" for e in ck) or "—",
                     f"{snap / 2**30:.3f}", len(res.get("skipped_updates", [])), extra])
    if rows:
        hdr = ["Run", "Trainable params", "Train h (2×T4)", "Peak GPU mem GiB (max rank)", "tok/s",
               "Recovery ckpt GiB", "Eval snapshot GiB", "Skipped updates", "Method overhead"]
        (tdir / "resources.md").write_text(_md_table(hdr, rows))


def curves(fdir: Path, tdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    logs = {r: _rows(p) for r in RUNS if (p := find_artifact(f"runs/{r}/train_log.jsonl"))}
    if not logs:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, r in enumerate(RUNS):
        if r not in logs:
            continue
        u = [x["update"] for x in logs[r]]
        y = [x["sft_loss"] for x in logs[r]]
        ax.plot(u, y, color=SERIES[i], alpha=0.18, linewidth=0.8)
        ax.plot(u, _ema(y), color=SERIES[i], label=RUN_LABEL[r])
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("SFT loss (token-mean CE)")
    ax.set_title("SFT loss (faint: per update; solid: EMA 0.05)", loc="left", color=INK)
    ax.legend(loc="upper right")
    ax.set_xlim(0, max(x["update"] for L in logs.values() for x in L) * 1.02)
    fig.tight_layout()
    fig.savefig(fdir / "sft_loss.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    for i, r in enumerate(RUNS):
        if r in logs:
            ax = axes[0] if r.startswith("full") else axes[1]
            ax.plot([x["update"] for x in logs[r]], [x["lr"] for x in logs[r]], color=SERIES[i], label=RUN_LABEL[r])
    axes[0].set_title("Full FT variants (peak 2e-5)", loc="left")
    axes[1].set_title("Adapters (peak 2e-4)", loc="left")
    for ax in axes:
        ax.set_xlabel("Optimizer update")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Learning rate")
    fig.tight_layout()
    fig.savefig(fdir / "lr.png")
    plt.close(fig)

    if "full_ft_l2sp" in logs:
        L = logs["full_ft_l2sp"]
        u = [x["update"] for x in L]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].plot(u, [x["lambda_omega"] for x in L], color=SERIES[1])
        axes[0].set_title("λΩ (penalty term in the objective)", loc="left")
        axes[1].plot(u, [x["displacement_l2"] for x in L], color=SERIES[1])
        axes[1].set_title("‖θ − θ0‖₂ during L2-SP training", loc="left")
        for ax in axes:
            ax.set_xlabel("Optimizer update")
        fig.tight_layout()
        fig.savefig(fdir / "l2sp_penalty.png")
        plt.close(fig)
        last = L[-1]
        disp = find_artifact("diagnostics/displacement.json")
        dv = json.loads(disp.read_text())["values"] if disp else {}
        rows = [["Full FT", "0", "—", "—", fmt(dv.get("fullft_u0625"), 3, False), fmt(dv.get("fullft_u1250"), 3, False)],
                ["Full FT + L2-SP", "1e-3", f"{last['omega']:.4e}", f"{last['lambda_omega']:.4e}", "—",
                 fmt(dv.get("l2sp_u1250"), 3, False)]]
        (tdir / "regularization.md").write_text(
            _md_table(["Run", "λ", "Ω (final)", "λΩ (final)", "‖θ−θ0‖ mid", "‖θ−θ0‖ final"], rows)
            + f"\nFinal-update SFT loss (L2-SP): {last['sft_loss']:.4f}; total objective {last['total_loss']:.4f}.\n")


def bench_figure(bench: dict, fdir: Path) -> None:
    if len(bench) < 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    conds = [c for c in ORDER if c in bench and c != "baseline"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, (t, name) in zip(axes, TASKS):
        vals = [bench[c].get(f"{t}_delta_pp") or 0.0 for c in conds]
        ax.barh(range(len(conds)), vals, color=SERIES[0], height=0.6)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_title(f"{name} Δ (pp)", loc="left")
        for y, v in enumerate(vals):
            ax.annotate(f"{v:+.1f}", (v, y), xytext=(3 if v >= 0 else -3, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=7, color=INK2)
        ax.grid(axis="y", visible=False)
        lo, hi = min(vals + [0]), max(vals + [0])
        pad = 0.25 * max(1.0, hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
    axes[0].set_yticks(range(len(conds)), [COND_LABEL[c] for c in conds])
    axes[0].invert_yaxis()
    fig.suptitle("Change from shared baseline (primary metric, percentage points)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(fdir / "benchmark_deltas.png")
    plt.close(fig)


def diagnostics(fdir: Path, tdir: Path) -> None:
    wp, ap = find_artifact("diagnostics/weights.csv"), find_artifact("diagnostics/activations.csv")
    if wp:
        W = list(csv.DictReader(open(wp)))
        rows = [[COND_LABEL.get(r["condition"], r["condition"]), r["layer"], f"{float(r['rho_k']):.4f}",
                 f"{float(r['rho_k_random_baseline']):.4f}", f"{float(r['eta']):.2e}"] for r in W]
        (tdir / "diagnostics_weights.md").write_text(
            _md_table(["Condition", "Layer", "ρ_k (k=128)", "ρ_k random-matrix ref", "η"], rows))
    if ap:
        A = list(csv.DictReader(open(ap)))
        rows = [[COND_LABEL.get(r["condition"], r["condition"]), r["layer"], r["group"], f"{float(r['median']):.3e}",
                 f"{float(r['mean']):.3e}", f"[{float(r['q25']):.2e}, {float(r['q75']):.2e}]"] for r in A]
        (tdir / "diagnostics_activations.md").write_text(
            _md_table(["Condition", "Layer", "Probe group", "median R(z)", "mean R(z)", "IQR"], rows))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _style(plt)
        conds = [c for c in ORDER if any(r["condition"] == c for r in A)]
        layers = sorted({int(r["layer"]) for r in A})
        fig, axes = plt.subplots(1, len(layers), figsize=(3.6 * len(layers), 3.2), sharey=True)
        axes = axes if len(layers) > 1 else [axes]
        for ax, li in zip(axes, layers):
            for gi, (g, lab) in enumerate((("math", "GSM8K probes"), ("nonmath", "HellaSwag probes"))):
                ys = []
                for c in conds:
                    hit = [r for r in A if r["condition"] == c and int(r["layer"]) == li and r["group"] == g]
                    ys.append(float(hit[0]["median"]) if hit else float("nan"))
                xs = [i + (gi - 0.5) * 0.25 for i in range(len(conds))]
                ax.scatter(xs, ys, s=28, color=SERIES[gi], label=lab, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.set_yscale("log")
            ax.set_title(f"down_proj, layer {li}", loc="left")
            ax.set_xticks(range(len(conds)), [COND_LABEL[c] for c in conds], rotation=40, ha="right", fontsize=7)
        axes[0].set_ylabel("median R(z) = ‖ΔW z‖ / ‖W0 z‖")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fdir / "activation_ratios.png")
        plt.close(fig)


def main() -> None:
    root = output_root()
    tdir, fdir = root / "report_assets" / "tables", root / "report_assets" / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    bench = benchmarks(root, tdir)
    resources(tdir)
    curves(fdir, tdir)
    bench_figure(bench, fdir)
    diagnostics(fdir, tdir)
    print("report assets:", sorted(str(p.relative_to(root)) for p in (root / "report_assets").rglob("*.*")))


if __name__ == "__main__":
    main()
