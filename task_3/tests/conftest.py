"""Synthetic ATML-PA3-Data exports (same file layout and column names as described in the task) for CPU tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TASK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TASK / "src"), str(TASK.parent / "common")]

MATH_GOLD = ["\\frac{1}{2}", "3", "2\\sqrt{3}", "x^2+1", "10", "\\text{Monday}", "7", "(1,2)"]
MATH_EQUIV = {"\\frac{1}{2}": ["0.5", "\\dfrac12", "1/2"], "3": ["3.0", "x=3"], "2\\sqrt{3}": ["\\sqrt{12}"],
              "x^2+1": ["1+x^2"], "10": ["10^\\circ"], "\\text{Monday}": ["Monday"], "7": ["7"], "(1,2)": ["(1,2)"]}


def _rollout_tokens(rng, T, n_vocab=50):
    return {"logp": (-rng.exponential(0.5, T)).astype(np.float32).tolist(),
            "kl": rng.uniform(2, 10, T).astype(np.float32).tolist(),
            "ids": rng.integers(0, n_vocab, T).tolist()}


def make_data(root: Path, Q: int = 16, R: int = 32, seed: int = 0, long_format: bool = False,
              t_range: tuple = (5, 60)) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    # ---------------- MATH (LLM)
    d = root / "Qwen2.5-3B-Instruct" / "MATH"
    d.mkdir(parents=True, exist_ok=True)
    gen, prob = [], []
    for q in range(Q):
        sid = f"test/algebra/{q}.json"
        gold = MATH_GOLD[q % len(MATH_GOLD)]
        p_corr = [0.0, 1.0, 0.3, 0.6, 0.9][q % 5]
        for r in range(R):
            T = int(rng.integers(*t_range)) if not (q == 1 and r == 5) else 0
            u = rng.random()
            if q == 2 and r < 3:
                ext = None
            elif u < p_corr:
                ext = rng.choice([gold] + MATH_EQUIV[gold])
            else:
                ext = str(int(rng.integers(20, 24)))
            tok = _rollout_tokens(rng, T)
            if ext is not None and u < p_corr:     # make correct rollouts a bit more confident
                tok["kl"] = (np.array(tok["kl"]) + 1.0).tolist()
            text = f"<think>reasoning {q}-{r}</think> The answer is $\\boxed{{{ext}}}$." if ext else "<think>cut"
            gen.append({"dataset": "math", "sample_id": sid, "rollout_id": r, "response": text,
                        "token_ids": tok["ids"] + [151645], "num_tokens": T, "extracted_answer": ext,
                        "gold_answer": gold, "problem": f"Problem {q}: compute something."})
            prob.append({"dataset": "math", "sample_id": sid, "rollout_id": r, "logp": tok["logp"], "kl_u_p": tok["kl"]})
    pd.DataFrame(gen).to_parquet(d / "generations.parquet", index=False)
    pr = pd.DataFrame(prob).sample(frac=1.0, random_state=1)     # different row order than generations
    if long_format:
        rows = [{"dataset": x["dataset"], "sample_id": x["sample_id"], "rollout_id": x["rollout_id"], "token_index": t,
                 "logp": lp, "kl_u_p": kl} for x in pr.to_dict("records")
                for t, (lp, kl) in enumerate(zip(x["logp"], x["kl_u_p"]))]
        pd.DataFrame(rows).sample(frac=1.0, random_state=2).to_parquet(d / "probability_metrics.parquet", index=False)
    else:
        pr.to_parquet(d / "probability_metrics.parquet", index=False)
    # ---------------- MathVista (VLM)
    d = root / "Qwen2.5-VL-3B-Instruct" / "MathVista"
    d.mkdir(parents=True, exist_ok=True)
    gen, prob, att = [], [], []
    for q in range(Q):
        sid = str(q + 1)
        kind = q % 3
        if kind == 0:
            meta = {"question_type": "multi_choice", "answer_type": "text", "choices": ["red", "blue", "green"],
                    "precision": None}
            gold, wrong = "blue", ["A", "(C) green", "gren", "C"]
            right = ["B", "(B) blue", "blue"]
        elif kind == 1:
            meta = {"question_type": "free_form", "answer_type": "integer", "choices": None, "precision": None}
            gold, wrong, right = "5", ["4", "6", "five?"], ["5", "5.0"]
        else:
            meta = {"question_type": "free_form", "answer_type": "float", "choices": None, "precision": 1.0}
            gold, wrong, right = "1.5", ["1.4", "2"], ["1.5", "1.50", "1.53"]
        p_corr = [0.2, 0.8, 0.5, 1.0][q % 4]
        for r in range(R):
            T = int(rng.integers(*t_range))
            ext = (rng.choice(right) if rng.random() < p_corr else rng.choice(wrong)) if r != 7 else None
            tok = _rollout_tokens(rng, T)
            img = rng.uniform(0.0, 0.3, T)
            prm = rng.uniform(0.2, 0.5, T)
            gn = rng.uniform(0.2, 0.5, T)
            if q == 3 and r == 2 and T > 3:
                img[1] = prm[1] = gn[1] = 0.0          # undefined v at one token
            gen.append({"dataset": "mathvista", "sample_id": sid, "rollout_id": r, "response": f"... {ext}",
                        "token_ids": tok["ids"], "num_tokens": T, "extracted_answer": ext, "gold_answer": gold,
                        "question": f"Q{q}", **meta})
            prob.append({"dataset": "mathvista", "sample_id": sid, "rollout_id": r, "logp": tok["logp"],
                         "kl_u_p": tok["kl"]})
            att.append({"dataset": "mathvista", "sample_id": sid, "rollout_id": r, "image_attention": img.tolist(),
                        "prompt_text_attention": prm.tolist(), "generated_text_attention": gn.tolist()})
    g = pd.DataFrame(gen)
    g["choices"] = g["choices"].apply(lambda c: json.dumps(c) if c is not None else None)
    g.to_parquet(d / "generations.parquet", index=False)
    pd.DataFrame(prob).to_parquet(d / "probability_metrics.parquet", index=False)
    pd.DataFrame(att).sample(frac=1.0, random_state=3).to_parquet(d / "attention_metrics.parquet", index=False)
    (root / "README.MD").write_text("synthetic test data\n")
    return root


@pytest.fixture()
def env(tmp_path, monkeypatch):
    data = make_data(tmp_path / "data")
    out = tmp_path / "out"
    monkeypatch.setenv("PA3_OUTPUT_ROOT", str(out))
    monkeypatch.setenv("PA3_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("PA3_CONFIG_OVERRIDES", json.dumps({
        "data_root": str(data), "n_questions_expected": 16, "n_dev": 6, "n_live": 3, "n_boot": 50}))
    monkeypatch.setenv("PA3_LIVE_BACKEND", "fake")
    monkeypatch.delenv("PA3_CPUS", raising=False)
    return {"data": data, "out": out, "tmp": tmp_path}
