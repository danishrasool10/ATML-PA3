"""Connector output-magnitude calibration (training data only).

Rule: s = median L2 norm of Qwen input-embedding rows over the text tokens that occur in the Stage-1
training sequences (prefix + prompt + caption + <|im_end|>, frequency weighted)
      / median per-token L2 norm of the *initialised* connector output on `n` seeded training images.
s is stored as the connector's fixed `scale` buffer (not trained).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


@torch.no_grad()
def calibrate(connector, embed_weight: torch.Tensor, token_counts: Counter, get_feats, image_keys: list[str],
              n_images: int, seed: int, device: torch.device, chunk: int = 64) -> dict:
    ids = np.array(sorted(token_counts), dtype=np.int64)
    cnt = np.array([token_counts[i] for i in ids], dtype=np.float64)
    tnorm = embed_weight[torch.from_numpy(ids).to(embed_weight.device)].float().norm(dim=-1).cpu().numpy()
    text_med = weighted_median(tnorm, cnt)
    rng = np.random.default_rng([seed, 7])
    keys = sorted(rng.choice(sorted(set(image_keys)), size=min(n_images, len(set(image_keys))), replace=False).tolist())
    norms = []
    for i in range(0, len(keys), chunk):
        f = get_feats(keys[i:i + chunk]).to(device, torch.float32)
        norms.append(connector.raw(f).float().norm(dim=-1).flatten().cpu())
    vn = torch.cat(norms).numpy()
    vis_med = float(np.median(vn))
    s = text_med / vis_med
    connector.scale.fill_(s)
    return {"rule": "s = weighted median ||E[t]|| over Stage-1 training text tokens / median per-token "
                    "||g_phi(F)|| of the initialised connector on seeded training images",
            "scale": s, "text_median_norm": text_med, "visual_median_norm_before": vis_med,
            "visual_median_norm_after": vis_med * s, "n_text_tokens": int(cnt.sum()), "n_unique_text_tokens": len(ids),
            "text_norm_quantiles": {q: weighted_quantile(tnorm, cnt, q) for q in (0.1, 0.5, 0.9)},
            "visual_norm_quantiles_before": {q: float(np.quantile(vn, q)) for q in (0.1, 0.5, 0.9)},
            "n_images": len(keys), "n_visual_tokens": int(vn.size), "image_keys": keys}


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, q * c[-1])])
