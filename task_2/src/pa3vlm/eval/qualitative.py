"""Qualitative complementary-pair examples with a selection rule fixed before looking at outputs.

Rule: for each suite, classify every test pair (main run) into buckets using the correct and swap
conditions, then pick up to K pairs per bucket with a seeded RNG (seed 6304) from the bucket's pairs
sorted by pair id:
  grounded   : both answers right with the correct images AND both follow the partner's answer when swapped
  same_answer: identical answer for both images of the pair (correct condition)
  unsupported: at least one swapped prediction matches neither the own nor the partner reference
A pair may fall into several buckets; each pair is shown at most once (first bucket in this order).
"""
from __future__ import annotations

import json
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np

BUCKETS = ("grounded", "same_answer", "unsupported")


def load_records(final_dir: Path) -> list[dict]:
    return [json.loads(l) for f in sorted(final_dir.glob("*__*.jsonl")) for l in f.read_text().splitlines() if l.strip()]


def by_pair(recs: list[dict]) -> dict:
    out: dict = defaultdict(lambda: defaultdict(dict))
    for r in recs:
        out[(r["suite"], r["pair_id"])][r["condition"]][r["slot"]] = r
    return out


def buckets(pr: dict) -> list[str]:
    c, s = pr.get("correct", {}), pr.get("swap", {})
    out = []
    if c and s and all(c[i]["pred_norm"] == c[i]["canonical"] for i in (0, 1)) and \
            all(s[i]["pred_norm"] == s[i]["partner_canonical"] for i in (0, 1)):
        out.append("grounded")
    if c and c[0]["pred_norm"] == c[1]["pred_norm"]:
        out.append("same_answer")
    if s and any(s[i]["pred_norm"] not in (s[i]["canonical"], s[i]["partner_canonical"]) for i in (0, 1)):
        out.append("unsupported")
    return out


def select(recs: list[dict], seed: int, k: int = 2) -> list[dict]:
    bp = by_pair(recs)
    chosen, seen = [], set()
    for si, suite in enumerate(sorted({s for s, _ in bp})):
        for bi, b in enumerate(BUCKETS):
            cands = sorted(pid for (s, pid), pr in bp.items() if s == suite and b in buckets(pr) and (s, pid) not in seen)
            rng = np.random.default_rng([seed, 51, si, bi])
            pick = sorted(rng.choice(cands, size=min(k, len(cands)), replace=False).tolist()) if cands else []
            for pid in pick:
                seen.add((suite, pid))
                chosen.append({"suite": suite, "pair_id": pid, "bucket": b, "conditions": {
                    c: {str(sl): {kk: r[kk] for kk in ("question", "image_used", "own_image", "target", "canonical",
                                                       "pred_raw", "pred_norm", "score", "answers")}
                        for sl, r in v.items()} for c, v in bp[(suite, pid)].items()}})
    return chosen


def render(chosen: list[dict], img_dir: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from ..data.download import image_path

    if not chosen:
        return
    n = len(chosen)
    fig, axes = plt.subplots(n, 2, figsize=(11, 4.3 * n))
    axes = np.atleast_2d(axes)
    for r, ex in enumerate(chosen):
        cond = ex["conditions"]
        for s in (0, 1):
            ax = axes[r, s]
            c = cond["correct"][str(s)]
            try:
                ax.imshow(Image.open(image_path(img_dir, c["own_image"])).convert("RGB"))
            except OSError:
                ax.text(0.5, 0.5, "image unavailable", ha="center")
            ax.axis("off")
            refs = c["canonical"] if c["answers"] is None else ", ".join(sorted(set(c["answers"])))
            lines = [f"[{ex['bucket']}] {ex['pair_id']} slot {s}", f"Q: {c['question']}", f"ref: {refs}"]
            for cn in ("correct", "swap", "mismatched", "none"):
                if cn in cond:
                    lines.append(f"{cn}: {cond[cn][str(s)]['pred_raw']!r}")
            ax.set_title("\n".join(textwrap.shorten(x, 70) for x in lines), fontsize=8, loc="left")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
