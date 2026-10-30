# PA3 Task 2 — Vision-language model and visual grounding

SigLIP (`google/siglip-base-patch16-224`, frozen, 196 cached patch features) → 2-layer MLP connector
(768→1024→1024, trainable, fixed calibration scale *s*) → `Qwen/Qwen3-0.6B` (frozen FP16; LoRA r16 on
q/k/v/o in Stages 2–3). Three-stage curriculum (2,250 + 1,500 + 750 updates, effective batch 16) and a
grounding evaluation under four image conditions. Approved design: project doc `claude/task2_design_decisions.md`.

## Run on Kaggle
Open `notebooks/pa3_task2_kaggle.ipynb` (self-contained: it writes every file of this folder and `common/`
with `%%writefile` cells). Settings: **GPU T4 ×2**, **Internet on**, then **Save & Run All (Commit)**.
GPU0 trains the 196-token model and GPU1 the optional 49-token model at the same time.
If `status.json` says `paused`, make the output a dataset, attach it and run again (work resumes exactly).
If a download host is unreachable, attach a dataset with the COCO 2014 / VQAv2 JSONs (and optionally the
`train2014`/`val2014` image folders); they are found automatically under `/kaggle/input`.

Regenerate the notebook after code changes: `python task_2/scripts/build_notebook.py`.

## Layout
```
configs/        base.yaml (shared recipe), vt196.yaml, vt49.yaml, data.yaml (selection sizes)
src/pa3vlm/
  data/         download (attached-or-download), coco (category wording, label filters), vqa (pair filter),
                grounding (presence/counting pairs), splits (seeded selection + isolation checks),
                normalize (VQA normalisation + soft score), prompts (chat template with image slot),
                collate (masks/positions/labels), sampler (stateless curriculum), bundle (lazy tokenisation)
  features/     siglip_cache (FP16 memmap, LN applied once), store (+2x2 pooling)
  models/       connector, calibrate, vlm (embedding splice, supervised-only LM head), build
  train/        loop (3 stages, AMP, checkpoints, dev eval, pause/resume), schedule, checkpoint
  eval/         dev, generate (KV-cache greedy), conditions, metrics, bootstrap, final, qualitative
  orchestrate.py  Kaggle pipeline (prefetch → data → audit → features → preflight → train → eval → report)
scripts/        build_notebook.py, label_audit.py, make_report_assets.py
tests/          CPU tests (tiny 28-layer Qwen3, synthetic COCO/VQA): 23 tests
```

## Outputs (`/kaggle/working/pa3_task2_outputs`)
`data/` manifests, selected IDs, selection report, label audit · `features/` caching cost ·
`runs/<vt196|vt49>/` config, calibration, train_log, dev_metrics, events, checkpoints log, stage weights,
`final_eval/` (per-condition generations, summary.json/.md) · `report_assets/` figures and tables ·
`/kaggle/working/results_task2.zip` — made by the notebook's last cell (everything except resumable
checkpoints), with a download link printed under the cell.

## CPU tests
`PA3_TEST_TOKENIZER=<dir with the Qwen3 tokenizer> python -m pytest task_2/tests -q` (defaults to the hub id).
