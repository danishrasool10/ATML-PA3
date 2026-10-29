# Task 1 results (Kaggle 2x T4, two sessions, status: complete)

Exact copy of `pa3_outputs/` from the Kaggle runs, minus model weights / checkpoints / OPLoRA bases /
tokenised `.npy` data (those stay in the Kaggle output datasets). The per-example lm-eval dumps
`eval/*/samples_*.jsonl` are gzipped (`.jsonl.gz`) to keep the repository small; content is unchanged.

| Path | Contents |
|---|---|
| `status.json`, `decisions.json`, `code_commit.txt` | completion status, preflight decisions (L2-SP used ZeRO-1), code commits |
| `data/report.json`, `data/probe_ids.json` | MetaMathQA filtering counts + hash, activation-probe IDs |
| `runs/<run>/` | `train_log.jsonl` (loss, lr, grad norm, Ω/λΩ for L2-SP per update), `resources.json`, `checkpoints.jsonl`, `config.json` |
| `eval/<condition>/` | `aggregate.json` (all metrics), `gsm8k_audit.jsonl` (per-example extracted answers + exact truncation flags), `merge_check.json` (adapters), `samples_*.jsonl.gz` |
| `diagnostics/` | `weights.csv` (ρ_k, η), `activations*.csv` (R(z)), `displacement.json`, `oplora_checks.json` |
| `qualitative/` | `responses.jsonl` (168), `responses_grouped.md`, `annotation_sheet.csv` (42 rows, auto-flags pre-filled) |
| `report_assets/` | tables (`*.md`, `benchmarks.csv`) and figures (`*.png`) |
| `logs/`, `smoke/` | subprocess logs, FP16 sanity check |

Note on GSM8K: the official metric is lm-eval `flexible-extract` (last number in the response). The
baseline often writes its answer in bold/`\boxed{}` and keeps writing, so re-scoring by each response's
own final answer gives baseline 47.2% (vs 43.3% flexible); fine-tuned models are identical under both
(they end with "The answer is: N"). `strict-match` is 0 for every model because its regex expects
"The answer is N" without the colon that MetaMathQA-style answers use.
