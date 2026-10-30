"""Curriculum sampler (stateless: every batch is a pure function of (seed, stage, update)).

Stage 1: `passes` seeded permutations of the caption examples; pass p uses default_rng([seed, 1, p]);
         an update takes 16 consecutive examples of one pass (18,000 / 16 = 1,125 updates per pass).
Stages 2/3: an update = 8 draws, a microbatch = 4 draws; draw k of stage s uses default_rng([seed, s, k]):
         pick a component with the stage probabilities, then a uniform pair from that pool (both examples,
         kept together) or two uniform caption examples (caption replay).
Resuming only needs the global update counter, so there is no sampler RNG state to drift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# item = ("caption", index) or (pool_name, pair_index, slot)
Item = tuple


@dataclass
class StageSpec:
    index: int          # 1, 2, 3
    updates: int
    mix: dict | None    # None for stage 1


class CurriculumSampler:
    def __init__(self, seed: int, n_captions: int, pool_sizes: dict[str, int], micro_bs: int, grad_accum: int,
                 passes: int = 2):
        assert micro_bs % 2 == 0, "pairs must fit in a microbatch"
        self.seed, self.n_cap, self.pool_sizes = seed, n_captions, pool_sizes
        self.mb, self.acc, self.passes = micro_bs, grad_accum, passes
        self.eff = micro_bs * grad_accum
        self._perm: dict[int, np.ndarray] = {}

    def stage1_updates(self) -> int:
        assert self.n_cap % self.eff == 0, "caption count must be a multiple of the effective batch"
        return self.passes * self.n_cap // self.eff

    def _perm_for(self, p: int) -> np.ndarray:
        if p not in self._perm:
            self._perm[p] = np.random.default_rng([self.seed, 1, p]).permutation(self.n_cap)
        return self._perm[p]

    def draw(self, stage: int, k: int, mix: dict) -> tuple[str, list[Item]]:
        rng = np.random.default_rng([self.seed, stage, k])
        names = sorted(mix)
        probs = np.array([mix[n] for n in names], dtype=float)
        comp = names[int(rng.choice(len(names), p=probs / probs.sum()))]
        if comp == "caption":
            idx = rng.integers(self.n_cap, size=2)
            return comp, [("caption", int(idx[0])), ("caption", int(idx[1]))]
        j = int(rng.integers(self.pool_sizes[comp]))
        return comp, [(comp, j, 0), (comp, j, 1)]

    def update(self, spec: StageSpec, u: int) -> tuple[list[list[Item]], dict]:
        """Microbatches (lists of items) for local update u (0-based) of a stage, plus per-component example counts."""
        assert 0 <= u < spec.updates
        counts: dict[str, int] = {}
        if spec.mix is None:
            per_pass = self.n_cap // self.eff
            p, off = divmod(u, per_pass)
            idx = self._perm_for(p)[off * self.eff:(off + 1) * self.eff]
            items = [("caption", int(i)) for i in idx]
            counts["caption"] = self.eff
        else:
            dpu = self.eff // 2
            items = []
            for d in range(dpu):
                comp, its = self.draw(spec.index, u * dpu + d, spec.mix)
                counts[comp] = counts.get(comp, 0) + 2
                items += its
        mbs = [items[m * self.mb:(m + 1) * self.mb] for m in range(self.acc)]
        return mbs, counts
