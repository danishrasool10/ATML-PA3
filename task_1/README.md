# PA3 Task 1 — Math fine-tuning of Qwen3-0.6B: Full FT, L2-SP, LoRA-4/16, OPLoRA-16

CS-6304 / EE-5102 Advanced Topics in ML (LUMS), Danish Rasool (28100233).

## Layout

```
configs/      base.yaml (shared recipe) + one yaml per run
src/pa3/
  config.py               RunConfig dataclass, yaml + CLI overrides
  data/                   prepare.py (40k eligible MetaMathQA, decontam, pre-tokenise), formatting.py (chat
                          template, thinking off), dataset.py (masking, deterministic DDP batch plan), normalize.py
  models/                 load.py, lora.py (LoRALinear for LoRA + OPLoRA, export/merge), oplora.py (bases,
                          constraint checks), l2sp.py (bf16-exact theta0, Omega, penalty gradient)
  train/                  loop.py (torchrun entry), loss.py (chunked masked CE), optim.py, checkpoint.py
  eval/                   conditions.py (7 conditions), harness.py (lm-eval + GSM8K audit + exact truncation),
                          merge.py (merged vs unmerged logits), qualitative.py (24 prompts, 42-row sheet), run.py
  diagnostics/            probe_set.py, activations.py (hooks), weights.py (rho_k, eta, R(z)), run.py
  smoke.py, orchestrate.py   S0 smoke checks and the S0/S1/S2/S3 stage drivers
scripts/      compute_bases.py, make_report_assets.py, build_notebook.py
tests/        CPU test suite (tiny 28-layer Qwen3 + real tokenizer; DDP via gloo)
notebooks/pa3_task1_kaggle.ipynb   writes everything above to disk and runs a selected stage
```

All artifacts live under `$PA3_OUTPUT_ROOT` (`/kaggle/working/pa3_outputs` on Kaggle):
`data/`, `bases/`, `runs/<run>/{train_log.jsonl, resources.json, checkpoints.jsonl, weights_uXXXX/}`,
`eval/<condition>/{aggregate.json, samples_*.jsonl, gsm8k_audit.jsonl, merge_check.json}`,
`qualitative/`, `diagnostics/`, `report_assets/`, `smoke/`, `logs/`, `decisions.json`.

## Key implementation decisions (approved)

* **Batching.** Effective batch 32 = 2 GPUs x microbatch 2 x accumulation 8 (DDP, `no_sync` on the
  first 7 microsteps). Update k uses examples [32k, 32k+32) in source order; rank r takes pairs
  [8r, 8r+8). Each microbatch's summed token CE is divided by N_k (supervised tokens in all 32
  examples, from the manifest) and multiplied by the world size, so the synced gradient equals
  single-GPU accumulation 16 (verified: rel. error ~1e-7 on CPU for all methods).
* **Masking.** Prompt = chat template with `enable_thinking=False` and generation prompt (ends with the
  empty think block); supervised = response tokens + `<|im_end|>`; prompt/padding masked.
* **Precision.** FP16 autocast + dynamic GradScaler; Full FT keeps FP32 master weights, adapters are
  FP32 on a frozen FP16 backbone; FP32 AdamW states; unscale -> (L2-SP grad) -> clip 1.0 -> step.
  Skipped updates are logged, do not advance AdamW's step count, but do advance the budget index
  and the LR schedule (HF cosine, 38 warmup, decays to 0; note HF's first update has lr = 0).
* **L2-SP.** theta0 stored in bf16 (exact for the bf16 checkpoint); Omega over all unique params
  (tied embedding once), not averaged; lambda*(theta - theta0) added after unscale, so clipping acts on
  the total gradient; Omega, lambda*Omega, total loss and ||theta - theta0|| logged every update.
  Fallbacks: ZeRO-1, then theta0 in pinned host memory (S0 decides and records in decisions.json).
* **OPLoRA.** FP32 thin SVD of each starting weight once (k=128); effective factors recomputed each
  forward in FP32 (no dense projectors); projected factors exported and used for merging/diagnostics;
  constraint + Proposition-2 checks in `diagnostics/oplora_checks.json`.
* **Checkpoints.** Recovery at updates 312/625/937/1250 with model, optimizer, scheduler, scaler,
  per-rank RNG and data position; rolling (only newest kept, sizes recorded first); FP32 weight
  snapshots kept at 625 (Full FT) and 1250 (all); final optimizer state dropped after completion.

## Running on Kaggle

See the first cell of the notebook. Stage order: S0 (smoke, ~1 h) -> S1 -> S2 -> S3, each a separate
session; after each stage turn the version's output into a dataset and attach it to the next session.

## Tests (CPU)

```
PA3_TEST_TOKENIZER=<dir with Qwen3 tokenizer files or omit to download> python -m pytest -q tests
```
