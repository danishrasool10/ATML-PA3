"""Shared evaluation runners and the development evaluation.

Development metrics (logged before training, every 500 updates and at stage ends):
- caption reference loss on the 96 development captions (training objective: content + <|im_end|>),
  with the correct image and (extra) without the image;
- reference-answer NLL (termination token excluded; token-mean per example, then mean over examples) on
  both development suites under correct / complementary-swap / no-image inputs, and the image-swap
  penalty  dNLL = NLL(y | q, I_partner) - NLL(y | q, I_correct)  (positive = the correct image helps);
- (extra) greedy accuracy with the correct image;
- the median per-token norm of the scaled connector output on development images (drift monitor).
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from ..data.collate import collate
from ..data.normalize import normalize_answer
from ..data.prompts import Encoded
from ..models.vlm import VLM, per_example_mean
from .conditions import images_for
from .generate import timed_greedy
from .metrics import score


def amp_ctx(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()


@torch.no_grad()
def nll_examples(vlm: VLM, encs: list[Encoded], images: list, get_feats, n_vis: int, pad_id: int, bs: int,
                 device: torch.device, exclude_term: bool = True) -> np.ndarray:
    out = []
    for i in range(0, len(encs), bs):
        b = collate(encs[i:i + bs], images[i:i + bs], get_feats, n_vis, pad_id, "right", True).to(device)
        with amp_ctx(device):
            nll, rows, term = vlm.supervised_nll(b)
        keep = ~term if exclude_term else None
        out.append(per_example_mean(nll, rows, len(encs[i:i + bs]), keep).cpu())
    return torch.cat(out).numpy().astype(np.float64)


@torch.no_grad()
def generate_examples(vlm: VLM, encs: list[Encoded], images: list, get_feats, n_vis: int, fmt, bs: int,
                      max_new: int, device: torch.device) -> list[dict]:
    res = []
    for i in range(0, len(encs), bs):
        chunk = encs[i:i + bs]
        b = collate(chunk, images[i:i + bs], get_feats, n_vis, fmt.pad_id, "left", False).to(device)
        with amp_ctx(device):
            seqs, stopped, secs = timed_greedy(vlm, b, max_new, fmt.eos_ids, fmt.pad_id, device)
        for s, st in zip(seqs, stopped):
            raw = fmt.tok.decode(s, skip_special_tokens=True)
            res.append({"pred_raw": raw, "pred_norm": normalize_answer(raw), "stopped": bool(st),
                        "n_new_tokens": len(s) + int(st), "gen_time": secs / len(chunk)})
    return res


def dev_eval(vlm: VLM, bundle, get_feats, n_vis: int, fmt, device: torch.device, bs: int, max_new: int,
             update: int, stage: int) -> dict:
    was_training = vlm.training
    vlm.eval()
    row: dict = {"update": update, "stage": stage}
    caps = bundle.captions_dev()
    imgs = [e.image for e in caps]
    row["caption_loss"] = float(nll_examples(vlm, caps, imgs, get_feats, n_vis, fmt.pad_id, bs, device, False).mean())
    row["caption_loss_noimage"] = float(nll_examples(vlm, caps, [None] * len(caps), get_feats, n_vis, fmt.pad_id, bs,
                                                     device, False).mean())
    for name, suite in (("grounding_dev", "grounding"), ("vqa_dev", "vqa")):
        pairs, encs = bundle.suite(name)
        kinds = np.array([p["kind"] for p in pairs for _ in p["examples"]])
        exs = [e for p in pairs for e in p["examples"]]
        nl = {}
        for cond in ("correct", "swap", "none"):
            nl[cond] = nll_examples(vlm, encs, images_for(pairs, cond), get_feats, n_vis, fmt.pad_id, bs, device)
        gens = generate_examples(vlm, encs, images_for(pairs, "correct"), get_feats, n_vis, fmt, bs, max_new, device)
        acc = np.array([score(suite, g["pred_norm"], e) for g, e in zip(gens, exs)])
        for sub in ["all"] + sorted(set(kinds)):
            m = np.ones(len(kinds), bool) if sub == "all" else kinds == sub
            key = f"{suite}/{sub}"
            for cond, v in nl.items():
                row[f"{key}/nll_{cond}"] = float(v[m].mean())
            row[f"{key}/delta_nll_swap"] = float((nl["swap"] - nl["correct"])[m].mean())
            row[f"{key}/delta_nll_none"] = float((nl["none"] - nl["correct"])[m].mean())
            row[f"{key}/acc_greedy"] = float(acc[m].mean())
    with torch.no_grad(), amp_ctx(device):
        f = get_feats(sorted(set(imgs))[:32]).to(device)
        row["connector_norm_median"] = float(vlm.connector(f).float().norm(dim=-1).median())
    if was_training:
        vlm.train()
    return row
