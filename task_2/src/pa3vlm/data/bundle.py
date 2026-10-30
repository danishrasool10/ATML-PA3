"""Manifests + lazy, cached tokenisation of training items and evaluation suites."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .prompts import ChatFormat, Encoded, encode_caption, encode_pair

POOLS = {"presence": "presence_pool", "counting": "counting_pool", "vqa": "vqa_train"}


class Bundle:
    def __init__(self, data_dir: str | Path, fmt: ChatFormat):
        self.dir = Path(data_dir)
        self.fmt = fmt
        self._m: dict[str, list] = {}
        self._cap: dict[int, Encoded] = {}
        self._pair: dict[tuple[str, int], list[Encoded]] = {}

    def manifest(self, name: str) -> list:
        if name not in self._m:
            self._m[name] = json.loads((self.dir / f"{name}.json").read_text())
        return self._m[name]

    @property
    def n_captions(self) -> int:
        return len(self.manifest("captions_train"))

    def pool_sizes(self) -> dict[str, int]:
        return {k: len(self.manifest(v)) for k, v in POOLS.items()}

    def caption(self, i: int) -> Encoded:
        if i not in self._cap:
            self._cap[i] = encode_caption(self.fmt, self.manifest("captions_train")[i])
        return self._cap[i]

    def pair(self, pool: str, j: int) -> list[Encoded]:
        key = (pool, j)
        if key not in self._pair:
            self._pair[key] = encode_pair(self.fmt, self.manifest(POOLS[pool])[j])
        return self._pair[key]

    def item(self, it: tuple) -> Encoded:
        return self.caption(it[1]) if it[0] == "caption" else self.pair(it[0], it[1])[it[2]]

    def stage1_token_counts(self) -> Counter:
        c: Counter = Counter()
        for i in range(self.n_captions):
            e = self.caption(i)
            c.update(e.prefix)
            c.update(e.suffix)
            c.update(e.target)
        return c

    def captions_dev(self) -> list[Encoded]:
        return [encode_caption(self.fmt, r) for r in self.manifest("captions_dev")]

    def suite(self, name: str) -> tuple[list[dict], list[Encoded]]:
        """Pairs and their examples in (pair, slot) order."""
        pairs = self.manifest(name)
        encs = [e for p in pairs for e in encode_pair(self.fmt, p)]
        return pairs, encs
