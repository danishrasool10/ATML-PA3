# Task 2 results (Kaggle 2x T4, one session, status: complete)

Exact copy of `pa3_task2_outputs/` from the Kaggle run (the `results_task2.zip` produced by the notebook), unchanged.
Resumable checkpoints were never part of the zip; the trained connector + LoRA weights are included.
The run completed every step; the final `status.json` write raised a bookkeeping error that has since been fixed
(`task_2/src/pa3vlm/orchestrate.py`), so `status.json` stops at the report step.

| Path | Contents |
|---|---|
| `status.json` | pipeline steps, timings, return codes |
| `data/` | selection manifests (captions, presence/counting pools, VQAv2 pairs, dev/test suites), `selected_ids.json`, `selection_report.json` (candidate counts, isolation checks), `image_fetch.json`, `label_audit/` (contact sheets + automated checks) |
| `features/` | SigLIP caching cost per GPU shard (time, images/s, peak memory, LN check, FP16 stats) |
| `preflight/<run>/` | 3-update-per-stage timing and memory check, calibration from that check |
| `runs/<vt196\|vt49>/` | `config.json`, `calibration.json` (scale s and its medians), `train_log.jsonl` (loss, lr, grad norm, mixture per update), `dev_metrics.jsonl` (12 dev evaluations), `events.jsonl` (trainable params per stage, runtime, memory), `checkpoints.jsonl`, `status.json` |
| `runs/<run>/weights/` | `stage1.pt`, `stage2.pt`, `stage3.pt`, `final.pt` (connector state incl. scale + LoRA factors; load with `pa3vlm.eval.final.load_run_weights`) |
| `runs/<run>/final_eval/` | per-condition generations `{grounding,vqa}_test__{correct,swap,mismatched,none}.jsonl` (raw + normalised answer, score, reference NLL, time), `summary.json` / `summary.md` (4,000-resample pair-bootstrap CIs), `eval_resources.json` |
| `eval_mappings/` | the fixed mismatched-image mapping per test suite |
| `report_assets/` | figures (learning curves, qualitative sheets) and tables (final results, 196 vs 49, resources, qualitative examples) |
| `logs/` | subprocess logs for every step |

Headline (vt196, correct image): grounding 71.9% [63.3, 80.5] (presence 89.1%, counting 54.7%), VQAv2 soft accuracy
59.4% [52.6, 66.2]; vt49: 68.0% and 53.0%. Full analysis: project doc "PA3 Task 2 — VLM Results and Analysis".
