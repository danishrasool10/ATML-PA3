"""Compute OPLoRA singular bases once (FP32) for every targeted projection; record timing/storage."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[2] / "common")]
from pa3.config import TARGET_MODULES  # noqa: E402
from pa3.models.load import load_model  # noqa: E402
from pa3.models.oplora import compute_bases, save_bases  # noqa: E402
from pa3.utils.common import find_artifact, output_root, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--rel", default="bases/oplora_bases_k128.safetensors")
    a = ap.parse_args()
    if find_artifact(a.rel) is not None:
        print(f"bases already exist: {find_artifact(a.rel)}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.model_id, torch.float32)  # exact upcast of bf16 starting weights
    tensors, meta = compute_bases(model, a.k, TARGET_MODULES, device)
    out = output_root() / a.rel
    meta["file_bytes"] = save_bases(tensors, meta, out)
    meta["model_id"] = a.model_id
    write_json(out.with_suffix(".json"), meta)
    print(meta)


if __name__ == "__main__":
    main()
