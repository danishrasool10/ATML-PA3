"""Image-control conditions.  Questions and original references stay fixed; only the image input changes.

correct    : the example's own image
swap       : the partner image of the same complementary pair (different reference answer)
mismatched : the image in the same slot of another pair of the same suite (seeded derangement of pairs,
             fixed before generation and reused everywhere; never either image of the original pair;
             every image is used exactly once)
none       : visual tokens omitted (masks and positions rebuilt; text unchanged)
"""
from __future__ import annotations

import numpy as np

CONDITIONS = ("correct", "swap", "mismatched", "none")


def derangement(n: int, seed_parts: list[int]) -> list[int]:
    """Sattolo's algorithm: a uniformly random cyclic permutation, hence no fixed points (n >= 2)."""
    assert n >= 2
    rng = np.random.default_rng(seed_parts)
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(i))
        p[i], p[j] = p[j], p[i]
    assert all(p[i] != i for i in range(n))
    return p


def mismatch_mapping(pairs: list[dict], seed: int, tag: int) -> dict[str, str]:
    """ex_id -> image key of the mismatched control."""
    perm = derangement(len(pairs), [seed, 31, tag])
    out = {}
    for i, p in enumerate(pairs):
        other = pairs[perm[i]]
        own = {e["image"] for e in p["examples"]}
        for s, e in enumerate(p["examples"]):
            img = other["examples"][s]["image"]
            assert img not in own
            out[e["ex_id"]] = img
    assert len(set(out.values())) == len(out), "each image is used once"
    return out


def images_for(pairs: list[dict], cond: str, mismatch: dict[str, str] | None = None) -> list[str | None]:
    imgs: list[str | None] = []
    for p in pairs:
        a, b = p["examples"]
        for e, partner in ((a, b), (b, a)):
            if cond == "correct":
                imgs.append(e["image"])
            elif cond == "swap":
                imgs.append(partner["image"])
            elif cond == "mismatched":
                imgs.append(mismatch[e["ex_id"]])
            elif cond == "none":
                imgs.append(None)
            else:
                raise ValueError(cond)
    return imgs
