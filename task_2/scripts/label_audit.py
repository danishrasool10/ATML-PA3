"""Label audit for the evaluation grounding pairs, made before any model is trained (independent of predictions).

    python scripts/label_audit.py

Writes data/label_audit/page_XX.png (each image with its question, target and the annotated boxes of the
asked category) and data/label_audit/summary.json (automated checks: per-pair categories, counts, instance
areas, crowd flags, caption mentions for negatives).
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TASK / "src"), str(TASK.parent / "common")]

from pa3common.utils import output_root, write_json  # noqa: E402
from pa3vlm.data.coco import captions_mention, index_captions  # noqa: E402
from pa3vlm.data.download import ensure_annotation, image_path  # noqa: E402


def main(per_page: int = 12) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    R = output_root()
    scratch = Path(os.environ.get("PA3_SCRATCH", "/tmp/pa3_scratch"))
    out = R / "data" / "label_audit"
    out.mkdir(parents=True, exist_ok=True)
    inst = json.loads(ensure_annotation("instances_val2014.json", scratch / "raw").read_text())
    caps = index_captions(json.loads(ensure_annotation("captions_val2014.json", scratch / "raw").read_text()))
    cname = {c["id"]: c["name"] for c in inst["categories"]}
    boxes = defaultdict(list)
    for a in inst["annotations"]:
        boxes[(a["image_id"], cname[a["category_id"]])].append((a["bbox"], a.get("iscrowd", 0), a["area"]))
    del inst
    pairs = []
    for name in ("grounding_dev", "grounding_test"):
        pairs += json.loads((R / "data" / f"{name}.json").read_text())
    summary = []
    for p in pairs:
        for e in p["examples"]:
            b = boxes.get((e["image_id"], p["category"]), [])
            summary.append({"pair_id": p["pair_id"], "kind": p["kind"], "category": p["category"], "image": e["image"],
                            "target": e["target"], "n_noncrowd": sum(1 for _, c, _ in b if not c),
                            "n_crowd": sum(1 for _, c, _ in b if c), "areas": sorted(round(x) for _, _, x in b),
                            "caption_mentions_category": captions_mention(p["category"],
                                                                          [c["caption"] for c in caps.get(e["image_id"], [])])})
    flags = [s for s in summary if (s["target"] == "no" and (s["n_noncrowd"] or s["caption_mentions_category"]))
             or (s["kind"] == "counting" and str(s["n_noncrowd"]) != s["target"])]
    write_json(out / "summary.json", {"n_pairs": len(pairs), "flags": flags, "examples": summary})
    for pg in range(0, len(pairs), per_page):
        chunk = pairs[pg:pg + per_page]
        rows = (len(chunk) + 1) // 2
        fig, axes = plt.subplots(rows, 4, figsize=(16, 3.6 * rows))
        axes = axes.reshape(rows, 4)
        for i, p in enumerate(chunk):
            for s, e in enumerate(p["examples"]):
                ax = axes[i // 2, (i % 2) * 2 + s]
                try:
                    ax.imshow(Image.open(image_path(scratch / "images", e["image"])).convert("RGB"))
                except OSError:
                    ax.text(0.5, 0.5, "missing", ha="center")
                for (x, y, w, h), crowd, _ in boxes.get((e["image_id"], p["category"]), []):
                    ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, lw=1.5, ec="red" if crowd else "lime"))
                ax.set_title(textwrap.shorten(f"{p['pair_id']}/{s} {p['question']} -> {e['target']}", 60), fontsize=7)
                ax.axis("off")
        for j in range(len(chunk), rows * 2):
            for s in (0, 1):
                axes[j // 2, (j % 2) * 2 + s].axis("off")
        fig.tight_layout()
        fig.savefig(out / f"page_{pg // per_page:02d}.png", dpi=70)
        plt.close(fig)
    print(f"[audit] {len(pairs)} pairs, {len(flags)} automated flags -> {out}", flush=True)


if __name__ == "__main__":
    main()
