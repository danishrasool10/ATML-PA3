"""Evaluate one condition: lm-eval benchmarks + qualitative generations (+ merge check).

    CUDA_VISIBLE_DEVICES=0 python -m pa3.eval.run --condition baseline
    python -m pa3.eval.run --condition oplora16_u1250 --limit 8      # smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..models.load import load_tokenizer
from ..utils.common import find_artifact, output_root, write_json
from .conditions import CONDITIONS, build_eval_model
from .harness import run_lm_eval
from .merge import merge_check
from .qualitative import generate_responses, load_questions


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_name", default=None, help="defaults to eval/<condition> (smoke: eval_smoke/...)")
    ap.add_argument("--questions", default=None, help="path to first_plot_questions.yaml")
    ap.add_argument("--skip_lmeval", action="store_true")
    ap.add_argument("--skip_qual", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--gen_batch", type=int, default=64)
    ap.add_argument("--ll_batch", type=int, default=32)
    ap.add_argument("--bases_path", default="bases/oplora_bases_k128.safetensors")
    a = ap.parse_args(argv)

    cond = CONDITIONS[a.condition]
    root = output_root()
    sub = a.out_name or (f"eval_smoke/{cond.name}" if a.limit else f"eval/{cond.name}")
    out_dir = root / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(a.model_id)
    timing: dict = {}

    if cond.kind == "adapter" and not (out_dir / "merge_check.json").exists():
        t0 = time.perf_counter()
        from ..config import TARGET_MODULES
        mc = merge_check(cond, a.model_id, tok, device, a.bases_path, TARGET_MODULES)
        timing["merge_check_seconds"] = time.perf_counter() - t0
        write_json(out_dir / "merge_check.json", mc)
        print(f"[{cond.name}] merge check: {json.dumps(mc)}")

    t0 = time.perf_counter()
    model = build_eval_model(cond, a.model_id, device)
    timing["build_seconds"] = time.perf_counter() - t0

    if not a.skip_lmeval:  # parts already finished (part_gsm8k.json / part_ll.json) are reused
        t0 = time.perf_counter()
        agg = run_lm_eval(model, tok, out_dir, limit=a.limit, gen_batch=a.gen_batch, ll_batch=a.ll_batch,
                          tasks=a.tasks)
        timing["lm_eval_seconds"] = time.perf_counter() - t0
        print(f"[{cond.name}] primary: {json.dumps(agg['primary'])}")
        if "gsm8k_audit" in agg:
            print(f"[{cond.name}] gsm8k audit: {json.dumps(agg['gsm8k_audit'])}")

    qpath = Path(a.questions) if a.questions else find_artifact("first_plot_questions.yaml")
    qual_dir = root / ("qualitative_smoke" if a.limit else "qualitative") / "per_condition"
    qfile = qual_dir / f"{cond.name}.jsonl"
    if not a.skip_qual and not qfile.exists():
        if qpath is None or not Path(qpath).exists():
            print("[qualitative] first_plot_questions.yaml not found; skipping")
        else:
            qs = load_questions(qpath)
            if a.limit:
                qs = qs[: min(len(qs), 3)]
            t0 = time.perf_counter()
            rows = generate_responses(model, tok, qs, cond.name, 512, device)
            timing["qualitative_seconds"] = time.perf_counter() - t0
            qual_dir.mkdir(parents=True, exist_ok=True)
            with open(qfile, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{cond.name}] wrote {len(rows)} qualitative responses")
    if device.type == "cuda":
        timing["peak_mem_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        timing["device"] = torch.cuda.get_device_name(device)
    tf = out_dir / "timing.json"
    prev = json.loads(tf.read_text()) if tf.exists() else {}
    write_json(tf, {**prev, **timing})


if __name__ == "__main__":
    main()
