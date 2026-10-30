"""SigLIP feature cache on CPU with a 1-layer random SigLIP vision tower (real 224/16 geometry, width 768):
final LayerNorm applied exactly once, FP16 memmap layout, deterministic preprocessing, 2x2 pooling."""
from __future__ import annotations

import json

import numpy as np
import torch

from conftest import run_module


def test_feature_cache(tmp_path, monkeypatch):
    from PIL import Image
    from transformers import SiglipConfig, SiglipModel, SiglipVisionModel

    from pa3vlm.features.siglip_cache import load_preproc, preprocess
    from pa3vlm.features.store import FeatureStore, pool2x2

    vdir = tmp_path / "siglip_tiny"
    torch.manual_seed(0)
    # full SigLIP checkpoint (vision_model.* + text_model.* keys), like the hub one
    full = SiglipModel(SiglipConfig(
        vision_config=dict(image_size=224, patch_size=16, hidden_size=768, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2),
        text_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2,
                         vocab_size=99, bos_token_id=1, eos_token_id=2, pad_token_id=1)))
    with torch.no_grad():   # non-trivial final LayerNorm so applying it twice would be detected
        full.vision_model.post_layernorm.weight.uniform_(0.5, 2.0)
        full.vision_model.post_layernorm.bias.uniform_(-1, 1)
    full.save_pretrained(vdir)
    m = SiglipVisionModel.from_pretrained(vdir).eval()
    ln = getattr(m, "vision_model", m).post_layernorm
    assert torch.equal(ln.weight, full.vision_model.post_layernorm.weight)
    (vdir / "preprocessor_config.json").write_text(json.dumps({
        "do_resize": True, "size": {"height": 224, "width": 224}, "resample": 3, "do_rescale": True,
        "rescale_factor": 1 / 255, "do_normalize": True, "image_mean": [0.5] * 3, "image_std": [0.5] * 3}))
    root, scratch = tmp_path / "out", tmp_path / "scratch"
    keys = ["train2014:1", "train2014:2", "val2014:3"]
    rng = np.random.default_rng(0)
    for k in keys:
        split, i = k.split(":")
        p = scratch / "images" / split / f"COCO_{split}_{int(i):012d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "L" if i == "3" else "RGB"       # grayscale COCO images exist: must be converted to RGB
        arr = rng.integers(0, 255, (300, 410, 3) if mode == "RGB" else (300, 410), dtype=np.uint8)
        Image.fromarray(arr, mode).save(p)
    (root / "data").mkdir(parents=True)
    (root / "data" / "images.json").write_text(json.dumps({"train": keys[:2], "val": keys[2:]}))
    monkeypatch.setenv("PA3_SCRATCH", str(scratch))
    for s in (0, 1):
        run_module(root, "pa3vlm.features.siglip_cache", "--shard", str(s), "--nshards", "2", "--vision-id",
                   str(vdir), "--workers", "0")
    st = FeatureStore(scratch / "features", 1)
    f = st.get(keys)
    assert f.shape == (3, 196, 768)
    pp = load_preproc(str(vdir))
    x = torch.from_numpy(np.stack([preprocess(scratch / "images" / "train2014" / f"COCO_train2014_{1:012d}.jpg", pp)]))
    assert x.shape == (1, 3, 224, 224)
    with torch.no_grad():
        out = m(pixel_values=x, output_hidden_states=True)
        once = ln(out.hidden_states[-1])
    assert torch.allclose(once, out.last_hidden_state, atol=1e-5)
    assert torch.allclose(f[0], once[0].half().float(), atol=1e-6)
    twice = ln(once)
    assert not torch.allclose(f[0], twice[0], atol=1e-2)
    meta = json.loads((root / "features" / "cache_shard0.json").read_text())
    assert meta["nonfinite_fp16"] == 0 and meta["ln_check_max_diff"] < 1e-4
    assert torch.allclose(FeatureStore(scratch / "features", 2).get(keys[:1]), pool2x2(f[:1]))
