# PA3 Task 3 — Test-time compute: sampling, selection, refinement and adaptive inference

Offline analysis of the supplied `ATML-PA3-Data` rollouts (32 per question; Qwen2.5-3B-Instruct on a 1,000-question
MATH subset, Qwen2.5-VL-3B-Instruct on MathVista testmini) plus a 50-question live experiment with
`Qwen/Qwen2.5-3B-Instruct` (FP16, batch 1). Approved design: see the Task 3 design summary (2026-10-31).

## Run on Kaggle
Open `notebooks/pa3_task3_kaggle.ipynb` (self-contained: it writes every file of this folder and the needed
`common/` modules with `%%writefile` cells). **Add Input → ATML-PA3-Data** (expected at `/kaggle/input/atml-pa3-data`,
otherwise found anywhere under `/kaggle/input`), **GPU T4 ×2**, **Internet on**, then **Save & Run All (Commit)**.
Only GPU0 is used (live generation, in the background); the offline stages run at the same time on the other CPU
cores. The last cell writes `/kaggle/working/results_task3.zip` and shows a download link, also after a failure.

The **DATA CHECK** cell (runs ~1 min after start) lists everything under `/kaggle/input` and the tables it will use;
if the data is not found it stops there, writes `data/input_listing.txt` into the results zip and the run ends with an
error (it no longer "finishes" with nothing done). Accepted layouts: the task's file names, shard files/folders,
both models in one folder with `math_`/`mathvista_` prefixes, an unextracted .zip/.tar, or a Hugging Face dataset repo
(`data_hf_repo` in the config). Any failed stage makes the final cell raise after the zip is written.

Column names are auto-detected (`outputs/data/<ds>_columns.json` shows what was picked and every column in the
files). If a pick is wrong, set it in `configs/task3.yaml → columns`, e.g. `columns: {math: {text: response}}`.

Regenerate the notebook after code changes: `python task_3/scripts/build_notebook.py`.
CPU tests: `cd task_3 && python -m pytest -q tests` (~20 s; needs torch + transformers for the HF-backend test).

## Pipeline (`src/pa3ttc/orchestrate.py`, one subprocess per stage, `python -m pa3ttc.stages <stage>`)
| stage | spec step | what it does |
|---|---|---|
| data | 1 | find the two model folders, join the 3 parquets on (dataset, sample_id, rollout_id), check token-list alignment, cache ragged token arrays |
| splits | — | sort IDs as strings, `default_rng(6304)` per dataset → 200 dev / 800 held-out; 10 orders per question (`default_rng([6304, d, q])`); first 50 MATH held-out IDs for the live run |
| *live (background, GPU0)* | 4 | 5 seeded calls per question (parallel ×3, revision, single), resumable `responses.jsonl` |
| scorer | 1 | answer groups + correctness (math-verify / MathVista rules), NO_ANSWER handling, frozen `scorer_manifest.json` |
| signals | 2 | L, K, V per rollout; VisConf / tokenwise fusion recalibrated on every candidate prefix |
| offline | 1–2 | every rule × N ∈ {1,2,4,8,16,32} × 10 orders; accuracy, oracle coverage, token cost, distinct answers, failure decomposition, paired bootstrap |
| analysis | 3 | length/score/correctness associations, quartile gaps, confidence–vision correlations, Z_tok − Z = Cov check, trajectories |
| adaptive | 5 | agreement and quality-aware stopping: dev grid → frozen settings → held-out evaluation |
| compute | 6 | marginal value of each extra sample, coverage-vs-selection gap |
| live_eval | 4 | accuracy, tokens, runtime, truncation/extraction, revision transitions |
| qualitative, report | 6 | seeded example cases, figures, tables, `report_assets/summary.md` |

## Layout
```
configs/task3.yaml     seeds, budgets, grids, live settings, optional column overrides
src/pa3ttc/
  data/      discover (folders + README), load (column resolution, Arrow list → ragged arrays, join, alignment), splits
  answers/   extract (balanced-brace last \boxed), math_equiv (math-verify grouping/grading), mathvista (official rules), scorer
  signals/   rollout (L, K, V), visconf (exact sorted-prefix-sum VisConf, no leakage), stage
  select/    rules (FirstAns, SC, argmax, shortest, RWC with average ranks, tie rules, vote margin)
  offline/   runner (methods × budgets × orders), bootstrap (paired, question-level)
  analysis/  signal_assoc, conf_vision, compute_alloc
  adaptive/  controllers (features, stopping, evaluation), tune (dev selection, freeze, held-out)
  live/      backend (HF FP16 / fake), run (worker), evaluate
  report/    figures, qualitative
  stages.py  orchestrate.py
scripts/build_notebook.py
tests/      unit tests, end-to-end pipeline on synthetic exports, live resume, real HF generation with a tiny Qwen2
```

## Outputs (`/kaggle/working/pa3_task3_outputs`)
`data/` README copy, column resolution, alignment reports · `splits/` split IDs, orders, candidate rollout IDs, live IDs ·
`answers/` labels, groups, scorer manifest · `signals/` rollout scores, VisConf per prefix ·
`offline/<ds>/` per_order (selected rollout/answer/cost per question × order × N × method), per_question, summaries,
bootstrap · `analysis/` · `adaptive/<ds>/` dev grid, frozen.json, held-out results, recoverable early stops ·
`compute/` · `live/` questions, responses.jsonl (prompts, responses, seeds, lengths, truncation, time, memory),
per-question table, summary · `qualitative/cases.md` · `report_assets/` figures, tables, summary.md ·
`status.json`, `logs/`.
