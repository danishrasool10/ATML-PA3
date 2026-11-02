"""Synthetic exports with exactly the schema of ATML-PA3-Data/README.MD (folder names, column names and types,
MATH ids "test:<n>", MathVista choices inside question_text with the gold given as the option letter)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

TASK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TASK / "src"), str(TASK.parent / "common")]

from pa3ttc.answers.extract import last_boxed  # noqa: E402

MATH_GOLD = ["\\frac{1}{2}", "3", "2\\sqrt{3}", "x^2+1", "10", "\\text{Monday}", "7", "(1,2)"]
MATH_EQUIV = {"\\frac{1}{2}": ["0.5", "\\dfrac12", "1/2"], "3": ["3.0", "x=3"], "2\\sqrt{3}": ["\\sqrt{12}"],
              "x^2+1": ["1+x^2"], "10": ["10^\\circ"], "\\text{Monday}": ["Monday"], "7": ["7"], "(1,2)": ["(1,2)"]}


def _lists(rows, col, typ):
    return pa.array([r[col] for r in rows], type=pa.list_(typ))


def _write(rows, path, scalars: dict, lists: dict):
    cols = {c: pa.array([r[c] for r in rows], type=t) for c, t in scalars.items()}
    cols.update({c: _lists(rows, c, t) for c, t in lists.items()})
    pq.write_table(pa.table(cols), path)


GEN_SCALARS = {"dataset": pa.string(), "model": pa.string(), "sample_id": pa.string(), "rollout_id": pa.int32(),
               "question_text": pa.string(), "n_generated_text_tokens": pa.int32(), "hit_max_new_tokens": pa.bool_(),
               "generated_text": pa.string(), "gold_answer": pa.string(), "extracted_answer": pa.string()}
KEY_SCALARS = {"dataset": pa.string(), "sample_id": pa.string(), "rollout_id": pa.int32()}


def make_data(root: Path, Q: int = 16, R: int = 32, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    base = Path(root) / "ATML-PA3-Data"
    # ---------------- MATH (LLM)
    d = base / "qwen25-math"
    d.mkdir(parents=True, exist_ok=True)
    gen, prob = [], []
    for q in range(Q):
        sid = f"test:{q * 7}"
        gold = MATH_GOLD[q % len(MATH_GOLD)]
        p_corr = [0.0, 1.0, 0.3, 0.6, 0.9][q % 5]
        for r in range(R):
            T = int(rng.integers(5, 60)) if not (q == 1 and r == 5) else 0
            u = rng.random()
            if q == 2 and r < 3:
                text = "<think>cut off before the box"
            elif u < p_corr:
                text = f"<think>w {q}-{r}</think> so $\\boxed{{{rng.choice([gold] + MATH_EQUIV[gold])}}}$."
            else:
                text = f"<think>w</think> \\boxed{{{int(rng.integers(20, 24))}}}"
            kl = rng.uniform(8, 40, T) + (2.0 if u < p_corr else 0.0)
            gen.append({"dataset": "math", "model": "Qwen/Qwen2.5-3B-Instruct", "sample_id": sid, "rollout_id": r,
                        "question_text": f"Problem {q}: compute something.", "n_generated_text_tokens": T,
                        "hit_max_new_tokens": T == 59, "generated_text": text, "gold_answer": gold,
                        "extracted_answer": last_boxed(text), "generated_token_ids": rng.integers(0, 1000, T).tolist()})
            prob.append({"dataset": "math", "sample_id": sid, "rollout_id": r,
                         "logp": (-rng.exponential(0.5, T)).tolist(), "kl_u_p": kl.tolist()})
    _write(gen, d / "generations.parquet", GEN_SCALARS, {"generated_token_ids": pa.int32()})
    order = rng.permutation(len(prob))                      # different row order than generations
    _write([prob[i] for i in order], d / "probability_metrics.parquet", KEY_SCALARS,
           {"logp": pa.float32(), "kl_u_p": pa.float32()})
    # ---------------- MathVista (VLM)
    d = base / "qwen25vl-mathvista"
    d.mkdir(parents=True, exist_ok=True)
    gen, prob, att = [], [], []
    for q in range(Q):
        sid = str(q + 1)
        kind = q % 3
        if kind == 0:   # multiple choice: choices in question_text, gold = letter
            qt, gold = "Which colour is the sky?\nA. red\nB. blue\nC. green", "B"
            right, wrong = ["B"], ["A", "C", "E"]
        elif kind == 1:
            qt, gold, right, wrong = "How many apples?", "5", ["5", "5.0"], ["4", "6", "five apples"]
        else:
            qt, gold, right, wrong = "What is the length?", "1.5", ["1.5", "1.50", "1.53"], ["1.4", "2"]
        p_corr = [0.2, 0.8, 0.5, 1.0][q % 4]
        for r in range(R):
            T = int(rng.integers(5, 60))
            ext = None if r == 7 else (rng.choice(right) if rng.random() < p_corr else rng.choice(wrong))
            gen.append({"dataset": "mathvista", "model": "Qwen2.5-VL-3B-Instruct", "sample_id": sid, "rollout_id": r,
                        "question_text": qt, "n_generated_text_tokens": T, "hit_max_new_tokens": False,
                        "generated_text": f"... {ext}", "gold_answer": gold, "extracted_answer": ext,
                        "generated_token_ids": rng.integers(0, 1000, T).tolist()})
            prob.append({"dataset": "mathvista", "sample_id": sid, "rollout_id": r,
                         "logp": (-rng.exponential(0.5, T)).tolist(), "kl_u_p": rng.uniform(4, 40, T).tolist()})
            img, prm = rng.uniform(0.0, 0.3, T), rng.uniform(0.02, 0.6, T)
            gn = rng.uniform(0.0, 0.7, T)
            gn[0] = 0.0                                      # first prediction: no generated-text attention
            att.append({"dataset": "mathvista", "sample_id": sid, "rollout_id": r, "n_image_tokens": 300,
                        "n_prompt_text_tokens": 40, "n_generated_text_tokens": list(range(T)),
                        "img_attn_total": img.tolist(), "prompt_text_attn_total": prm.tolist(),
                        "generated_text_attn_total": gn.tolist()})
    _write(gen, d / "generations.parquet", GEN_SCALARS, {"generated_token_ids": pa.int32()})
    _write(prob, d / "probability_metrics.parquet", KEY_SCALARS, {"logp": pa.float32(), "kl_u_p": pa.float32()})
    order = rng.permutation(len(att))
    _write([att[i] for i in order], d / "attention_metrics.parquet",
           {**KEY_SCALARS, "n_image_tokens": pa.int32(), "n_prompt_text_tokens": pa.int32()},
           {"n_generated_text_tokens": pa.int32(), "img_attn_total": pa.float32(),
            "prompt_text_attn_total": pa.float32(), "generated_text_attn_total": pa.float32()})
    (base / "README.MD").write_text("synthetic test data\n")
    return Path(root)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    data = make_data(tmp_path / "data")
    out = tmp_path / "out"
    monkeypatch.setenv("PA3_OUTPUT_ROOT", str(out))
    monkeypatch.setenv("PA3_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("PA3_CONFIG_OVERRIDES", json.dumps({
        "data_root": str(data), "data_fallback_root": str(data), "n_questions_expected": 16, "n_dev": 6,
        "n_live": 3, "n_boot": 50, "mathvista_metadata_hf": None}))
    monkeypatch.setenv("PA3_LIVE_BACKEND", "fake")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("PA3_CPUS", raising=False)
    return {"data": data, "out": out, "tmp": tmp_path}
