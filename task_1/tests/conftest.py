"""Shared fixtures: a tiny random Qwen3 (28 layers, same module names) + the real Qwen3 tokenizer.

Tokenizer source: $PA3_TEST_TOKENIZER (a dir with tokenizer.json/tokenizer_config.json) or the
hub id Qwen/Qwen3-0.6B (on Kaggle).  The tiny model is saved in bf16 like the hub checkpoint.
"""
from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT.parent / "common"
sys.path[:0] = [str(ROOT / "src"), str(COMMON)]

TINY = dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=28,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
            tie_word_embeddings=True, rms_norm_eps=1e-6, rope_theta=1_000_000.0)
N_UPDATES = 4
EFF = 32


def _tokenizer_src() -> str:
    return os.environ.get("PA3_TEST_TOKENIZER", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory) -> Path:
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    d = tmp_path_factory.mktemp("tiny_qwen3")
    torch.manual_seed(0)
    cfg = Qwen3Config(**TINY)
    m = Qwen3ForCausalLM(cfg)
    with torch.no_grad():  # larger weights so updates/projections are numerically meaningful
        for p in m.parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.05)
    m.to(torch.bfloat16).save_pretrained(d)
    tok = AutoTokenizer.from_pretrained(_tokenizer_src())
    tok.save_pretrained(d)
    return d


@pytest.fixture(scope="session")
def tok(tiny_dir):
    from pa3.models.load import load_tokenizer

    return load_tokenizer(str(tiny_dir))


def synthetic_rows(n_good: int, seed: int = 0) -> tuple[list[dict], list[str]]:
    """MetaMathQA-like rows with planted ineligible examples; returns (rows, gsm8k_test_questions)."""
    rnd = random.Random(seed)
    gsm = ["Janet’s ducks lay 16 eggs per day. How many eggs does she sell?",
           "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?"]
    rows = []
    words = "apple banana seven twelve add subtract total each per the of and so then answer".split()
    i = 0
    while sum(1 for r in rows if r.get("_good")) < n_good:
        q = " ".join(rnd.choice(words) for _ in range(rnd.randint(5, 40))) + f" ({i})?"
        a = " ".join(rnd.choice(words) for _ in range(rnd.randint(3, 60))) + f" The answer is: {i}"
        rows.append({"query": q, "response": a, "original_question": q, "type": "x", "_good": True})
        if i % 17 == 3:
            rows.append({"query": "  ", "response": a, "original_question": q, "_good": False})  # empty
        if i % 23 == 5:  # GSM8K test match via original_question after NFKC/case/whitespace normalisation
            rows.append({"query": "Rephrased: " + q, "response": a,
                         "original_question": "  JANET'S ducks lay 16 eggs\nper day.  How many eggs does she sell?",
                         "_good": False})
        if i % 29 == 7:
            rows.append({"query": q, "response": " ".join(["seven"] * 400), "original_question": q, "_good": False})
        i += 1
    return rows, gsm


@pytest.fixture(scope="session")
def out_root(tmp_path_factory, tiny_dir, tok) -> Path:
    """Prepared data for N_UPDATES updates under a fresh PA3_OUTPUT_ROOT."""
    from pa3.data.prepare import save_arrays, select_and_tokenize

    root = tmp_path_factory.mktemp("pa3_outputs")
    rows, gsm = synthetic_rows(N_UPDATES * EFF)
    gsm[0] = gsm[0].replace("’", "'")  # NFKC does not map curly quotes; keep exact-match semantics
    arrays, report = select_and_tokenize(tok, rows, gsm, n_examples=N_UPDATES * EFF, max_len=256, chunk=37)
    save_arrays(arrays, report, root / "data")
    os.environ["PA3_OUTPUT_ROOT"] = str(root)
    return root


def write_cfg(root: Path, tiny_dir: Path, **kw) -> Path:
    import yaml

    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    base.update({"model_id": str(tiny_dir), "n_examples": N_UPDATES * EFF, "num_updates": N_UPDATES,
                 "warmup_updates": 1, "max_len": 256, "recovery_updates": [], "ce_chunk": 37})
    base.update(kw)
    p = root / f"cfg_{kw.get('run_name', 'x')}.yaml"
    p.write_text(yaml.safe_dump(base))
    return p


def run_train(root: Path, cfg_path: Path, nproc: int, *sets: str, check: bool = True):
    import subprocess

    env = {**os.environ, "PA3_OUTPUT_ROOT": str(root), "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(COMMON)]),
           "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
           "-m", "pa3.train.loop", "--config", str(cfg_path), "--set", *sets]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if check and r.returncode != 0:
        raise RuntimeError(r.stdout[-3000:] + "\n" + r.stderr[-6000:])
    return r


@pytest.fixture(scope="session")
def bases_rel(out_root, tiny_dir) -> str:
    from pa3.config import TARGET_MODULES
    from pa3.models.load import load_model
    from pa3.models.oplora import compute_bases, save_bases

    rel = "bases/oplora_bases_k16.safetensors"
    m = load_model(str(tiny_dir), torch.float32)
    t, meta = compute_bases(m, 16, TARGET_MODULES, torch.device("cpu"))
    save_bases(t, meta, out_root / rel)
    return rel


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(ROOT / ".pytest_cache", ignore_errors=True)
