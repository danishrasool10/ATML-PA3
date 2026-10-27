# ATML PA3 — CS-6304 / EE-5102 (LUMS) · Danish Rasool (28100233)

```
common/            shared, task-agnostic code (import name: pa3common)
  pa3common/utils.py       seeding, torch.distributed helpers, JSONL logging, artifact lookup
  pa3common/kaggle.py      subprocess streaming, GPU job queue, session deadline, scratch disk, retries
  pa3common/nccl_check.py  2-GPU NCCL sanity check (torchrun)
task_1/            LLM fine-tuning: Full FT, L2-SP, LoRA-4/16, OPLoRA-16 on Qwen3-0.6B (see task_1/README.md)
  notebooks/pa3_task1_kaggle.ipynb   the Kaggle notebook (clones this repo and runs everything)
```

New tasks go in `task_N/` and put `common/` on `sys.path` (the Task 1 notebook shows how).
