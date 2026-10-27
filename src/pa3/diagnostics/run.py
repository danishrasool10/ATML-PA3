"""Run diagnostics for every available condition (weights, activations, displacement, OPLoRA checks).

    python -m pa3.diagnostics.run cache         # cache starting-model z for the probe set (FP32)
    python -m pa3.diagnostics.run analyze       # rho_k, eta, R(z), displacement, OPLoRA checks
"""
from __future__ import annotations

import argparse
import csv
import json

import torch

from ..eval.conditions import CONDITIONS, ORDER, weights_path
from ..models.load import load_model, load_tokenizer
from ..models.oplora import load_bases
from ..utils.common import find_artifact, output_root, read_json, write_json
from .activations import LAYERS, build_probe_prompts, cache_down_proj_inputs, save_cache
from .weights import (activation_ratios, adapter_displacement, deltas_for_condition, down_name,
                      full_displacement, oplora_checks_all, weight_metrics)

DIAG_CONDS = [c for c in ORDER if c != "baseline"]


def cmd_cache(a) -> None:
    out = output_root() / "diagnostics" / "z_cache.pt"
    if out.exists() and not a.force:
        print(f"{out} exists")
        return
    tok = load_tokenizer(a.model_id)
    probe = read_json(find_artifact("data/probe_ids.json"))
    prompts = build_probe_prompts(tok, probe)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.model_id, torch.float32)
    Z = cache_down_proj_inputs(model, prompts, device)
    save_cache(out, Z, prompts)
    print(f"cached z {tuple(Z.shape)} -> {out}")


def _available(name: str) -> bool:
    try:
        weights_path(CONDITIONS[name])
        return True
    except FileNotFoundError:
        return False


def cmd_analyze(a) -> None:
    root = output_root() / "diagnostics"
    base = load_model(a.model_id, torch.float32)  # exact upcast of the bf16 checkpoint
    bases = load_bases(find_artifact(a.bases_path))
    cache = torch.load(find_artifact("diagnostics/z_cache.pt"), weights_only=False)
    Z, prompts = cache["Z"], cache["prompts"]
    groups = [p["group"] for p in prompts]
    names = [down_name(li) for li in LAYERS]
    W0 = {n: base.get_submodule(n).weight.detach().float() for n in names}

    wrows, arows, disp, per_example = [], [], {}, []
    for c in DIAG_CONDS:
        if not _available(c):
            print(f"[diag] skip {c}: weights not available")
            continue
        cond = CONDITIONS[c]
        dWs = deltas_for_condition(cond, W0, names)
        for li, n in zip(LAYERS, names):
            m = weight_metrics(dWs[n], W0[n], bases[n]["U"], bases[n]["V"])
            wrows.append({"condition": c, "layer": li, **m})
            R = activation_ratios(dWs[n], W0[n], Z[:, LAYERS.index(li)])
            for g in ("math", "nonmath"):
                sel = torch.tensor([gg == g for gg in groups])
                r = R[sel]
                arows.append({"condition": c, "layer": li, "group": g, "n": int(sel.sum()),
                              "mean": float(r.mean()), "median": float(r.median()),
                              "q25": float(r.quantile(0.25)), "q75": float(r.quantile(0.75)),
                              "max": float(r.max())})
            for i, p in enumerate(prompts):
                per_example.append({"condition": c, "layer": li, "task": p["task"], "doc_id": p["doc_id"],
                                    "group": p["group"], "R": float(R[i])})
        disp[c] = full_displacement(cond, base) if cond.kind == "full" else adapter_displacement(cond)
        if c == "oplora16_u1250":
            write_json(root / "oplora_checks.json", oplora_checks_all(cond, base, bases))
        print(f"[diag] {c} done (displacement {disp[c]:.4f})")

    root.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("weights.csv", wrows), ("activations.csv", arows), ("activations_per_example.csv", per_example)):
        if rows:
            with open(root / fname, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    write_json(root / "displacement.json", {
        "note": "Full FT variants: ||theta - theta0||_2 over all unique params (FP32). Adapters: "
                "Frobenius norm of the merged correction over all targeted matrices.", "values": disp})
    print(json.dumps(disp, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "analyze"])
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--bases_path", default="bases/oplora_bases_k128.safetensors")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    {"cache": cmd_cache, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
