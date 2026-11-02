# PA3 Task 3 — Test-time compute: sampling, selection, refinement and adaptive inference

Offline analysis of the supplied `ATML-PA3-Data` rollouts (32 per question; Qwen2.5-3B-Instruct on a 1,000-question
MATH subset, Qwen2.5-VL-3B-Instruct on MathVista testmini) plus a 50-question live experiment with
`Qwen/Qwen2.5-3B-Instruct` (FP16, batch 1). Approved design: see the Task 3 design summary (2026-10-31).

## Run on Kaggle
1. Push this repo to GitHub (the notebook clones `danishrasool10/ATML-PA3` and refuses to run if the clone's
   `pa3ttc.__version__` differs from the notebook's `EXPECTED_CODE_VERSION`, i.e. if the push was forgotten).
2. Upload the supplied `ATML-PA3-Data` folder (README.MD, `qwen25-math/`, `qwen25vl-mathvista/`) as a Kaggle dataset and
   attach it (*Add Input*). It is found anywhere under `/kaggle/input`.
3. Open `notebooks/pa3_task3_kaggle.ipynb` (7 cells) → **GPU T4 ×2**, **Internet on** → **Save & Run All**.

Offline Steps 1–3, 5, 6 take ~10 min on two CPU cores; the live Step 4 runs concurrently on GPU0 (~2–4 h).
The last cell writes `/kaggle/working/results_task3.zip` with a download link, also after a failure.
Regenerate the notebook after changing `__version__`: `python task_3/scripts/build_notebook.py`.
CPU tests: `cd task_3 && python -m pytest -q tests` (~30 s).

## Data (ATML-PA3-Data/README.MD)
Read with the exact documented schema (any missing column stops the data stage with a clear message):
generations `question_text, generated_text, generated_token_ids, n_generated_text_tokens, hit_max_new_tokens,
gold_answer, extracted_answer`; probability_metrics `logp, kl_u_p`; attention_metrics `img_attn_total,
prompt_text_attn_total, generated_text_attn_total`; joined on `(dataset, sample_id, rollout_id)`. T =
`n_generated_text_tokens` is verified against every per-token list. MathVista choices come from `question_text`
(gold = option letter); answer type / precision from the released testmini metadata (AI4Math/MathVista on the Hub),
with a recorded fallback to the gold format if the Hub is unreachable. Live generation uses the README's pinned
model revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` and its extraction rule (last literal `\boxed{`).

## Pipeline (`src/pa3ttc/orchestrate.py`, one subprocess per stage, `python -m pa3ttc.stages <stage>`)
| stage | spec step | what it does |
|---|---|---|
| data | 1 | find `qwen25-math/` and `qwen25vl-mathvista/`, join on (dataset, sample_id, rollout_id), verify token-list alignment, cache token arrays |
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
  data/      discover (finds the two folders + README), load (documented schema, join, alignment), splits
  answers/   extract (README rule), math_equiv (math-verify grouping/grading), mathvista (official rules + metadata), scorer
  signals/   rollout (L, K, V), visconf (exact sorted-prefix-sum VisConf, no leakage), stage
  select/    rules (FirstAns, SC, argmax, shortest, RWC with average ranks, tie rules, vote margin)
  offline/   runner (methods × budgets × orders), bootstrap (paired, question-level)
  analysis/  signal_assoc, conf_vision, compute_alloc
  adaptive/  controllers (features, stopping, evaluation), tune (dev selection, freeze, held-out)
  live/      backend (HF FP16 / fake), run (worker), evaluate
  report/    figures, qualitative
  stages.py  orchestrate.py
scripts/build_notebook.py   writes the 7-cell notebook
tests/      unit tests, end-to-end pipeline on synthetic exports with the real schema, live resume, real HF generation with a tiny Qwen2
```

## Outputs (`/kaggle/working/pa3_task3_outputs`)
`data/` README copy, column resolution, alignment reports · `splits/` split IDs, orders, candidate rollout IDs, live IDs ·
`answers/` labels, groups, scorer manifest · `signals/` rollout scores, VisConf per prefix ·
`offline/<ds>/` per_order (selected rollout/answer/cost per question × order × N × method), per_question, summaries,
bootstrap · `analysis/` · `adaptive/<ds>/` dev grid, frozen.json, held-out results, recoverable early stops ·
`compute/` · `live/` questions, responses.jsonl (prompts, responses, seeds, lengths, truncation, time, memory),
per-question table, summary · `qualitative/cases.md` · `report_assets/` figures, tables, summary.md ·
`status.json`, `logs/`.
