"""Fixtures: a tiny random Qwen3 (28 layers, real module names, real vocab size) + the Qwen3 tokenizer, and a
synthetic COCO/VQAv2 universe for the data pipeline and an end-to-end CPU training run.

Tokenizer source: $PA3_TEST_TOKENIZER (dir with tokenizer files) or the hub id Qwen/Qwen3-0.6B (on Kaggle).
"""
from __future__ import annotations

import json
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
# Stage budgets for the CPU run: 16 train images x 3 captions = 48 = 3 updates/pass -> stage 1 = 6 updates
SMALL_DATA = ["caption_train_images=16", "caption_dev_images=6",
              'vqa_train={"yes/no": 4, "number": 2, "other": 4}', 'vqa_dev={"yes/no": 2, "number": 1, "other": 2}',
              'vqa_test={"yes/no": 3, "number": 2, "other": 3}', "grounding_dev={presence: 2, counting: 2}",
              "grounding_test={presence: 3, counting: 3}", "pool_cap_per_category=20"]


def _tokenizer_src() -> str:
    return os.environ.get("PA3_TEST_TOKENIZER", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory) -> Path:
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    d = tmp_path_factory.mktemp("tiny_qwen3")
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(Qwen3Config(**TINY))
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.05)
    m.to(torch.bfloat16).save_pretrained(d)
    AutoTokenizer.from_pretrained(_tokenizer_src()).save_pretrained(d)
    return d


@pytest.fixture(scope="session")
def tok(tiny_dir):
    from pa3vlm.models.vlm import load_tokenizer

    return load_tokenizer(str(tiny_dir))


# ----------------------------------------------------------------------------- synthetic COCO / VQAv2
CATS = ["dog", "cat", "car", "chair", "cup", "bottle", "person", "bench"]


def _coco(split: str, ids: list[int], rnd: random.Random) -> tuple[dict, dict]:
    from pa3vlm.data.coco import CATEGORIES

    names = sorted(CATEGORIES)
    cid = {n: i + 1 for i, n in enumerate(names)}
    images = [{"id": i, "file_name": f"COCO_{split}_{i:012d}.jpg", "width": 640, "height": 480} for i in ids]
    anns, caps, aid = [], [], 1
    for i in ids:
        present = rnd.sample(CATS, rnd.randint(1, 3))
        words = []
        for c in present:
            n = rnd.randint(1, 4)
            for _ in range(n):
                area = rnd.choice([5000.0, 8000.0, 12000.0] + ([300.0] if rnd.random() < 0.1 else []))
                anns.append({"id": aid, "image_id": i, "category_id": cid[c], "area": area,
                             "bbox": [10, 10, 50, 50], "iscrowd": 1 if rnd.random() < 0.03 else 0})
                aid += 1
            words.append(f"{n} {c}")
        for k in range(5):
            caps.append({"id": i * 10 + k, "image_id": i, "caption": f" A photo  showing {' and '.join(words)} ({k})."})
    inst = {"images": images, "annotations": anns, "categories": [{"id": cid[n], "name": n} for n in names]}
    return inst, {"images": images, "annotations": caps}


def _vqa(split: str, ids: list[int], rnd: random.Random, n_pairs: int, qid0: int) -> tuple[dict, dict, list]:
    qs, anns, pairs = [], [], []
    templates = {"yes/no": ("Is the {} visible?", ["yes", "no"]), "number": ("How many {} are there?", ["2", "3", "four"]),
                 "other": ("What color is the {}?", ["red", "blue", "green"])}
    q = qid0
    for j in range(n_pairs):
        kind = ["yes/no", "number", "other"][j % 3]
        t, vals = templates[kind]
        text = t.format(rnd.choice(CATS))
        a, b = rnd.sample(ids, 2)
        va, vb = rnd.sample(vals, 2)
        bad = j % 7 == 6                      # low agreement -> must be filtered
        for img, v in ((a, va), (b, vb)):
            agree = 6 if bad else rnd.choice([8, 9, 10])
            raw = [v] * agree + ["maybe"] * (10 - agree)
            if v == "red":
                raw = ["red"] * (agree - 3) + ["Red"] * 3 + ["maybe"] * (10 - agree)
            qs.append({"question_id": q, "image_id": img, "question": text})
            anns.append({"question_id": q, "image_id": img, "answer_type": kind,
                         "answers": [{"answer": x, "answer_id": k + 1} for k, x in enumerate(raw)]})
            q += 1
        pairs.append([q - 2, q - 1])
    return {"questions": qs}, {"annotations": anns}, pairs


def make_raw(raw: Path, seed: int = 0) -> None:
    rnd = random.Random(seed)
    raw.mkdir(parents=True, exist_ok=True)
    tr, va = list(range(1, 81)), list(range(100001, 100241))
    for split, ids in (("train2014", tr), ("val2014", va)):
        inst, caps = _coco(split, ids, rnd)
        (raw / f"instances_{split}.json").write_text(json.dumps(inst))
        (raw / f"captions_{split}.json").write_text(json.dumps(caps))
        qj, aj, pr = _vqa(split, ids, rnd, 60, 1_000_000 if split == "train2014" else 5_000_000)
        (raw / f"v2_OpenEnded_mscoco_{split}_questions.json").write_text(json.dumps(qj))
        (raw / f"v2_mscoco_{split}_annotations.json").write_text(json.dumps(aj))
        (raw / f"v2_mscoco_{split}_complementary_pairs.json").write_text(json.dumps(pr))


@pytest.fixture(scope="session")
def out_root(tmp_path_factory) -> Path:
    """Synthetic raw annotations -> manifests under a fresh PA3_OUTPUT_ROOT."""
    from pa3vlm.config import load_data_config
    from pa3vlm.data.splits import select_all, write_manifests

    root = tmp_path_factory.mktemp("pa3_t2_out")
    scratch = tmp_path_factory.mktemp("pa3_t2_scratch")
    make_raw(scratch / "raw")
    os.environ["PA3_OUTPUT_ROOT"] = str(root)
    os.environ["PA3_SCRATCH"] = str(scratch)
    os.environ["PA3_INPUT_ROOTS"] = str(scratch / "nonexistent")
    cfg = load_data_config(None, SMALL_DATA)
    sets, rep = select_all(scratch / "raw", cfg)
    write_manifests(sets, rep, root / "data")
    return root


def run_cfg(root: Path, tiny_dir: Path, name: str, **kw) -> Path:
    import yaml

    from pa3vlm.config import load_config

    cfg = load_config("vt196")
    cfg.update({"run_name": name, "model_id": str(tiny_dir), "features": "random", "dtype": "float32",
                "dev_every": 5, "ckpt_every": 2, "calib_images": 8, "bootstrap": 200})
    cfg["stages"][0]["updates"], cfg["stages"][1]["updates"], cfg["stages"][2]["updates"] = 6, 4, 3
    cfg.update(kw)
    p = root / f"cfg_{name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def run_module(root: Path, module: str, *args: str, check: bool = True):
    import subprocess

    env = {**os.environ, "PA3_OUTPUT_ROOT": str(root), "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(COMMON)]),
           "OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-m", module, *args], env=env, capture_output=True, text=True, timeout=3600)
    if check and r.returncode != 0:
        raise RuntimeError(r.stdout[-3000:] + "\n" + r.stderr[-6000:])
    return r


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(ROOT / ".pytest_cache", ignore_errors=True)
