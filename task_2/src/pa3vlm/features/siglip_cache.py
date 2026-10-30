"""Frozen SigLIP feature cache (run once, before training; the process exits so SigLIP leaves the GPU).

    python -m pa3vlm.features.siglip_cache --shard 0 --nshards 2     # one process per GPU

Features = the 196 final patch tokens in 14x14 row-major order with SigLIP's final LayerNorm applied
exactly once: we compute post_layernorm(hidden_states[-1]) explicitly and check it equals HF's
last_hidden_state (which is already post-LN), so the cache is correct whichever way HF behaves.
Encoder runs in FP32; features are stored as FP16 in a .npy memmap [N, 196, 768].
Preprocessing is deterministic PIL code driven by the checkpoint's preprocessor_config.json
(RGB, resize to 224x224 with its resample filter, rescale 1/255, normalise with its mean/std).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from pa3common.utils import gpu_mem_stats, output_root, write_json

from ..data.download import image_path


def load_preproc(vision_id: str) -> dict:
    if Path(vision_id).is_dir():
        cfg = json.loads((Path(vision_id) / "preprocessor_config.json").read_text())
    else:
        from huggingface_hub import hf_hub_download

        cfg = json.loads(Path(hf_hub_download(vision_id, "preprocessor_config.json")).read_text())
    size = cfg.get("size", {"height": 224, "width": 224})
    return {"height": int(size.get("height", 224)), "width": int(size.get("width", 224)),
            "resample": int(cfg.get("resample", 3)), "rescale": float(cfg.get("rescale_factor", 1 / 255)),
            "mean": [float(x) for x in cfg.get("image_mean", [0.5] * 3)],
            "std": [float(x) for x in cfg.get("image_std", [0.5] * 3)],
            "do_resize": cfg.get("do_resize", True), "do_normalize": cfg.get("do_normalize", True)}


def preprocess(path: Path, pp: dict) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if pp["do_resize"]:
            im = im.resize((pp["width"], pp["height"]), resample=pp["resample"])
        x = np.asarray(im, dtype=np.float32) * pp["rescale"]
    if pp["do_normalize"]:
        x = (x - np.array(pp["mean"], np.float32)) / np.array(pp["std"], np.float32)
    return x.transpose(2, 0, 1).copy()


class _DS(torch.utils.data.Dataset):
    def __init__(self, keys, img_dir, pp):
        self.keys, self.img_dir, self.pp = keys, img_dir, pp

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return torch.from_numpy(preprocess(image_path(self.img_dir, self.keys[i]), self.pp))


@torch.no_grad()
def encode(model, pix: torch.Tensor, check: bool = False) -> tuple[torch.Tensor, float | None]:
    out = model(pixel_values=pix, output_hidden_states=True)
    vm = model.vision_model if hasattr(model, "vision_model") else model
    feats = vm.post_layernorm(out.hidden_states[-1])
    diff = (feats - out.last_hidden_state).abs().max().item() if check else None
    return feats, diff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--vision-id", default="google/siglip-base-patch16-224")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    from transformers import SiglipVisionModel

    scratch = Path(os.environ.get("PA3_SCRATCH", "/tmp/pa3_scratch"))
    feat_dir = scratch / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    keys_all = json.loads((output_root() / "data" / "images.json").read_text())
    keys_all = keys_all["train"] + keys_all["val"]
    keys = keys_all[a.shard::a.nshards]
    done = feat_dir / f"shard{a.shard}of{a.nshards}.json"
    if done.exists() and json.loads(done.read_text()).get("keys") == keys:
        print(f"[features] shard {a.shard} already cached", flush=True)
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    pp = load_preproc(a.vision_id)
    model, info = SiglipVisionModel.from_pretrained(a.vision_id, dtype=torch.float32, output_loading_info=True)
    assert not info["missing_keys"], f"SigLIP vision weights missing from checkpoint: {sorted(info['missing_keys'])[:5]}"
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    grid = model.config.image_size // model.config.patch_size
    n_tok, dim = grid * grid, model.config.hidden_size
    assert (n_tok, dim) == (196, 768), (n_tok, dim)
    arr = np.lib.format.open_memmap(feat_dir / f"shard{a.shard}of{a.nshards}.npy", mode="w+", dtype=np.float16,
                                    shape=(len(keys), n_tok, dim))
    dl = torch.utils.data.DataLoader(_DS(keys, scratch / "images", pp), batch_size=a.batch, shuffle=False,
                                     num_workers=a.workers, pin_memory=dev.type == "cuda")
    j, max_abs, ln_diff, n_nonfinite = 0, 0.0, None, 0
    for b, pix in enumerate(dl):
        feats, d = encode(model, pix.to(dev), check=(b == 0))
        if d is not None:
            ln_diff = d
            assert d < 1e-3, f"post_layernorm(hidden_states[-1]) != last_hidden_state (max diff {d})"
        f16 = feats.half()
        n_nonfinite += int((~torch.isfinite(f16)).sum())
        max_abs = max(max_abs, feats.abs().max().item())
        arr[j:j + len(f16)] = f16.cpu().numpy()
        j += len(f16)
        if b % 20 == 0:
            print(f"[features] shard {a.shard}: {j}/{len(keys)} images", flush=True)
    arr.flush()
    assert j == len(keys) and n_nonfinite == 0, (j, n_nonfinite)
    secs = time.time() - t0
    write_json(done, {"keys": keys, "n": len(keys), "shape": [len(keys), n_tok, dim], "dtype": "float16"})
    write_json(output_root() / "features" / f"cache_shard{a.shard}.json", {
        "n_images": len(keys), "seconds": secs, "images_per_s": len(keys) / max(secs, 1e-9),
        "max_abs_feature": max_abs, "ln_check_max_diff": ln_diff, "nonfinite_fp16": n_nonfinite,
        "preprocessing": pp, "encoder_dtype": "float32", "store_dtype": "float16",
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu", "memory": gpu_mem_stats(dev),
        "bytes": int(arr.nbytes)})
    print(f"[features] shard {a.shard} done: {len(keys)} images in {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
