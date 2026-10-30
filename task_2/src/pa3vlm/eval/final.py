"""Final test evaluation of a finished run under four image conditions, plus summary tables with pair-bootstrap CIs.

    python -m pa3vlm.eval.final --config vt196

Greedy decoding, thinking disabled (template), 16 new tokens, batch 8.  Questions and original references
are fixed across conditions; the mismatched mapping is fixed (seeded) before generation and saved.
Per-example records (raw generation, normalised answer, score, reference NLL, time) are written per
condition; finished conditions are skipped on rerun.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pa3common.lora import load_lora_state
from pa3common.utils import gpu_mem_stats, output_root, read_json, write_json

from ..config import load_config
from ..models.build import build, device_dtype
from ..train.loop import attach_lora
from .bootstrap import ci, diff_ci, resample_indices
from .conditions import CONDITIONS, images_for, mismatch_mapping
from .dev import generate_examples, nll_examples
from .metrics import METRICS, pair_matrix, score

SUITES = (("grounding_test", "grounding", 1), ("vqa_test", "vqa", 2))


def load_run_weights(vlm, path: Path, cfg: dict) -> dict:
    w = torch.load(path, map_location="cpu", weights_only=False)
    if w["lora"]:
        attach_lora(vlm, cfg)
        load_lora_state(vlm.lm, w["lora"])
    vlm.connector.load_state_dict(w["connector"])
    return w["meta"]


def run_conditions(cfg: dict, weights: str = "final.pt") -> Path:
    device, dtype = device_dtype(cfg)
    run = output_root() / "runs" / cfg["run_name"]
    out = run / "final_eval"
    out.mkdir(parents=True, exist_ok=True)
    tok, fmt, bundle, store, vlm = build(cfg, device, dtype, grad_ckpt=False)
    meta = load_run_weights(vlm, run / "weights" / weights, cfg)
    vlm.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for name, suite, tag in SUITES:
        pairs, encs = bundle.suite(name)
        mm_path = output_root() / "eval_mappings" / f"{name}_mismatched.json"
        mm = mismatch_mapping(pairs, cfg["seed"], tag)
        if mm_path.exists():
            assert read_json(mm_path) == mm, "mismatched mapping changed"
        else:
            write_json(mm_path, mm)
        exs = [(p, s, e) for p in pairs for s, e in enumerate(p["examples"])]
        for cond in CONDITIONS:
            f = out / f"{name}__{cond}.jsonl"
            if f.exists():
                continue
            t0 = time.time()
            imgs = images_for(pairs, cond, mm)
            gens = generate_examples(vlm, encs, imgs, store.get, store.n_vis, fmt, cfg["eval_bs"],
                                     cfg["max_new_tokens"], device)
            nll = nll_examples(vlm, encs, imgs, store.get, store.n_vis, fmt.pad_id, cfg["eval_bs"], device)
            recs = []
            for (p, s, e), gen, img, nl in zip(exs, gens, imgs, nll):
                partner = p["examples"][1 - s]
                recs.append({"suite": suite, "condition": cond, "pair_id": p["pair_id"], "slot": s, "ex_id": e["ex_id"],
                             "kind": p["kind"], "question": p["question"], "image_used": img, "own_image": e["image"],
                             "partner_image": partner["image"], "target": e["target"], "canonical": e["canonical"],
                             "partner_canonical": partner["canonical"], "answers": e["answers"], **gen,
                             "score": score(suite, gen["pred_norm"], e), "nll": float(nl)})
            tmp = f.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in recs))
            tmp.replace(f)
            print(f"[final] {cfg['run_name']} {name} {cond}: acc={np.mean([r['score'] for r in recs]):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    write_json(out / "eval_resources.json", {"memory": gpu_mem_stats(device), "weights": weights, "meta": meta,
                                             "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"})
    return out


def _fmt(p, lo, hi, pct=True):
    k = 100 if pct else 1
    return f"{p * k:.1f} [{lo * k:.1f}, {hi * k:.1f}]" if pct else f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def summarize(run_dir: Path, seed: int, n_boot: int) -> dict:
    out = run_dir / "final_eval"
    recs = [json.loads(l) for f in sorted(out.glob("*__*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    pm = pair_matrix(recs)
    summary: dict = {"n_boot": n_boot, "suites": {}}
    md = []
    for name, suite, tag in SUITES:
        if (suite, "correct") not in pm:
            continue
        base = pm[(suite, "correct")]
        kinds = sorted(set(base["kinds"]))
        S: dict = {}
        for ki, sub in enumerate(["all"] + kinds):
            m = np.ones(len(base["kinds"]), bool) if sub == "all" else np.array(base["kinds"]) == sub
            idx = resample_indices(int(m.sum()), n_boot, [seed, 41, tag, ki])
            T: dict = {"n_pairs": int(m.sum()), "conditions": {}, "advantages": {}}
            for cond in CONDITIONS:
                if (suite, cond) not in pm:
                    continue
                M = pm[(suite, cond)]
                assert M["pair_ids"] == base["pair_ids"], "pairs must match across conditions"
                T["conditions"][cond] = {mt: ci(M[mt][m], idx) for mt in METRICS}
            for cond in ("swap", "mismatched", "none"):
                if (suite, cond) not in pm:
                    continue
                M = pm[(suite, cond)]
                T["advantages"][cond] = {"acc": diff_ci(base["acc"][m], M["acc"][m], idx),
                                         "pair_acc": diff_ci(base["pair_acc"][m], M["pair_acc"][m], idx),
                                         "nll": diff_ci(M["nll"][m], base["nll"][m], idx)}
            S[sub] = T
        summary["suites"][suite] = S
        # ---------- markdown
        md.append(f"### {suite} ({S['all']['n_pairs']} pairs, {2 * S['all']['n_pairs']} questions) — "
                  "mean [95% pair-bootstrap CI]\n")
        md.append("| condition | accuracy % | pair acc % | identical % | empty % | follows partner % | unsupported % | "
                  "ref NLL | ms/answer |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for cond, v in S["all"]["conditions"].items():
            md.append(f"| {cond} | {_fmt(*v['acc'])} | {_fmt(*v['pair_acc'])} | {_fmt(*v['identical'])} | "
                      f"{_fmt(*v['empty'])} | {_fmt(*v['follows_partner'])} | {_fmt(*v['unsupported'])} | "
                      f"{_fmt(*v['nll'], pct=False)} | {v['gen_time'][0] * 1000:.1f} |")
        md.append("\n**By question type (accuracy %)**\n")
        md.append("| type | pairs | " + " | ".join(CONDITIONS) + " |")
        md.append("|---|---|" + "---|" * len(CONDITIONS))
        for sub in ["all"] + kinds:
            cells = [_fmt(*S[sub]["conditions"][c]["acc"]) if c in S[sub]["conditions"] else "—" for c in CONDITIONS]
            md.append(f"| {sub} | {S[sub]['n_pairs']} | " + " | ".join(cells) + " |")
        md.append("\n**Correct-image advantage** (accuracy / pair accuracy: correct − control, pp; "
                  "NLL: control − correct)\n")
        md.append("| type | control | Δ accuracy | Δ pair acc | Δ NLL |")
        md.append("|---|---|---|---|---|")
        for sub in ["all"] + kinds:
            for cond, v in S[sub]["advantages"].items():
                md.append(f"| {sub} | {cond} | {_fmt(*v['acc'])} | {_fmt(*v['pair_acc'])} | "
                          f"{_fmt(*v['nll'], pct=False)} |")
        md.append("")
    write_json(out / "summary.json", summary)
    (out / "summary.md").write_text("\n".join(md))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--weights", default="final.pt")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    out = run_conditions(cfg, a.weights)
    summarize(out.parent, cfg["seed"], cfg["bootstrap"])
    print(f"[final] {cfg['run_name']}: summary written", flush=True)


if __name__ == "__main__":
    main()
