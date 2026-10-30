"""Read-only access to the cached SigLIP features (+ optional 2x2 average pooling for 49 tokens)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def pool2x2(x: torch.Tensor) -> torch.Tensor:
    """[N, 196, D] (14x14 row-major) -> [N, 49, D]: mean of each spatial 2x2 block, row-major 7x7."""
    n, t, d = x.shape
    g = int(round(t ** 0.5))
    assert g * g == t and g % 2 == 0
    return x.view(n, g // 2, 2, g // 2, 2, d).mean(dim=(2, 4)).reshape(n, (g // 2) ** 2, d)


class FeatureStore:
    def __init__(self, feat_dir: str | Path, pool: int = 1):
        feat_dir = Path(feat_dir)
        self.pool = pool
        self.arrays, self.index = [], {}
        metas = sorted(feat_dir.glob("shard*of*.json"))
        assert metas, f"no feature shards in {feat_dir}"
        for s, m in enumerate(metas):
            meta = json.loads(m.read_text())
            self.arrays.append(np.load(m.with_suffix(".npy"), mmap_mode="r"))
            for r, k in enumerate(meta["keys"]):
                self.index[k] = (s, r)
        self.n_vis = 196 // (pool * pool)

    def __contains__(self, key: str) -> bool:
        return key in self.index

    def get(self, keys: list[str]) -> torch.Tensor:
        """FP32 tensor [len(keys), n_vis, 768]."""
        x = torch.from_numpy(np.stack([self.arrays[s][r] for s, r in (self.index[k] for k in keys)])).float()
        return pool2x2(x) if self.pool == 2 else x


class RandomFeatures:
    """Deterministic stand-in features keyed by image id (CPU tests only)."""

    def __init__(self, pool: int = 1, dim: int = 768):
        self.pool, self.dim = pool, dim
        self.n_vis = 196 // (pool * pool)

    def __contains__(self, key: str) -> bool:
        return True

    def get(self, keys: list[str]) -> torch.Tensor:
        out = []
        for k in keys:
            g = torch.Generator().manual_seed(_kseed(k))
            out.append(torch.randn(196, self.dim, generator=g))
        x = torch.stack(out)
        return pool2x2(x) if self.pool == 2 else x


def _kseed(k: str) -> int:
    import hashlib

    return int(hashlib.sha256(k.encode()).hexdigest()[:8], 16)
