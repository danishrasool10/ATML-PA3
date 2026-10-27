"""Download everything once, with retries, before any parallel job starts (avoids two processes
racing on the same Hugging Face cache entry and avoids mid-run network failures)."""
from __future__ import annotations

import argparse

from pa3common.kaggle import retry


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--skip_train_data", action="store_true")
    a = ap.parse_args(argv)

    from huggingface_hub import snapshot_download

    retry(lambda: snapshot_download(a.model_id), what="model download")
    print(f"[prefetch] model {a.model_id} cached", flush=True)
    from datasets import load_dataset

    retry(lambda: load_dataset("openai/gsm8k", "main", split="test"), what="gsm8k")
    if not a.skip_train_data:
        retry(lambda: load_dataset("meta-math/MetaMathQA", split="train"), what="MetaMathQA")
    print("[prefetch] training data cached", flush=True)
    from lm_eval.tasks import TaskManager, get_task_dict

    def tasks():
        td = get_task_dict(["gsm8k_cot", "hellaswag", "boolq", "arc_easy"], TaskManager())
        for name, t in td.items():
            n = len(list(t.eval_docs))
            print(f"[prefetch] lm-eval {name}: {n} eval docs", flush=True)

    retry(tasks, what="lm-eval task datasets")


if __name__ == "__main__":
    main()
